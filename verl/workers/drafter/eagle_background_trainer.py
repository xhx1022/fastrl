import asyncio
import logging
import os
import time
from collections import deque
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import SmoothL1Loss
from torch.nn import functional as F

from verl.utils.data_buffer import DataBuffer
from verl.utils.device import is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import (
    get_device_id,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class EagleBackgroundTrainer:
    """FSDP2-capable background trainer for Eagle drafter model."""

    def __init__(
        self,
        drafter_module_fsdp,
        drafter_optimizer,
        drafter_lr_scheduler,
        drafter_train_config,
        drafter_device_mesh,
        model_config=None,
    ):
        self.model = drafter_module_fsdp
        self.optimizer = drafter_optimizer
        self.lr_scheduler = drafter_lr_scheduler
        self.config = drafter_train_config
        self.training_device_mesh = drafter_device_mesh
        self.model_config = model_config

        self.is_offload_param = self.config.get("is_offload_param", False)
        self.is_offload_optimizer = self.config.get("is_offload_optimizer", False)

        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        self._training_initialized = False
        self._training_active = False
        self.training_steps = 0

        self.collected_data = deque(maxlen=int(self.config.get("buffer_max_samples", 2000)))
        self.shared_data_buffer = None
        self.batch_size = int(self.config.get("batch_size_per_gpu", 32))

        # Initialize DataBuffer for storing data across RL steps
        buffer_max_size = int(self.config.get("data_buffer_max_size", 10000))
        # Only store hidden states in buffer if we're collecting them during generation
        collect_hidden_states_from_sgl = bool(self.config.get("collect_hidden_states_from_sgl", False))
        self.data_buffer = DataBuffer(max_size=buffer_max_size, store_hidden_states=collect_hidden_states_from_sgl)

        self.criterion = SmoothL1Loss(reduction="none")

        self.eagle_model_path = self.config.get("eagle_model_path", self.config.get("spec_model_path"))
        self.checkpoint_dir = self.config.get("checkpoint_path")
        self._last_ckpt_step = -1
        # New: optional per-step barrier (default False to avoid stalls)
        self.enable_mesh_barrier = bool(self.config.get("enable_step_barrier", False))

        # Track the last pending async checkpoint save future
        self._pending_checkpoint_future = None
        self._frozen_param_names = {"model.embed_tokens.weight", "lm_head.weight"}

        # Ulysses Sequence Parallelism configuration
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        if torch.distributed.get_rank() == 0:
            print(
                f"EagleBackgroundTrainer use_ulysses_sp={self.use_ulysses_sp} "
                f"(sp_size={self.ulysses_sequence_parallel_size})"
            )

    def _get_model_class(self, model_type: str):
        if model_type.lower() == "llama":
            from verl.workers.drafter.model.llama_eagle import LlamaForCausalLMEagle

            return LlamaForCausalLMEagle
        if model_type.lower() == "qwen2":
            from verl.workers.drafter.model.qwen2_eagle import Qwen2ForCausalLMEagle

            return Qwen2ForCausalLMEagle
        raise ValueError(f"Unsupported model type: {model_type}")

    def _get_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Get state dict excluding frozen layers (embed_tokens, lm_head)."""
        full_state_dict = self.model.state_dict()
        trainable_state_dict = {}

        for name, param in full_state_dict.items():
            # Skip frozen parameters
            if any(frozen_name in name for frozen_name in self._frozen_param_names):
                logger.debug(f"Skipping frozen parameter: {name}")
                continue
            trainable_state_dict[name] = param

        return trainable_state_dict

    def _save_checkpoint_async(self, step: int, is_final: bool = False):
        """Asynchronously save checkpoint using DCP's async_save.

        Args:
            step: Current training step
            is_final: Whether this is the final checkpoint during cleanup

        Returns:
            Future object from dcp.async_save that can be awaited or checked for completion
        """
        if not self.checkpoint_dir:
            return None

        try:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"eagle_step_{step}")
            os.makedirs(checkpoint_path, exist_ok=True)

            # Get trainable state dict (excluding frozen layers)
            model_state_dict = self._get_trainable_state_dict()
            optimizer_state_dict = self.optimizer.state_dict() if self.optimizer else {}

            state_dict = {"model": model_state_dict, "optimizer": optimizer_state_dict, "step": step}

            # Use DCP async_save - returns a future that can be checked later
            future = dcp.async_save(
                state_dict=state_dict,
                checkpoint_id=checkpoint_path,
                process_group=self.training_device_mesh.get_group(),
            )
            return future

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Async checkpoint save failed on rank {self.rank}: {e}")
            return None

    async def activate_training_model(
        self, device_mesh: DeviceMesh, training_ranks: list[int], base_model=None
    ) -> bool:
        start_ts = time.time()
        try:
            logger.warning(
                f"[EagleTrainer rank {getattr(self, 'rank', -1)}] activate_training_model enter "
                f"training_ranks={training_ranks}"
            )

            first_param = next(self.model.parameters(), None)
            param_device = first_param.device.type if first_param is not None else None

            if self.is_offload_param or param_device != "cuda":
                load_fsdp_model_to_gpu(self.model)
                logger.debug("Loaded drafter model to GPU for training")

            if self.optimizer is not None:
                load_fsdp_optimizer(optimizer=self.optimizer, device_id=get_device_id())
                logger.debug("Loaded drafter optimizer to GPU for training")
                # if self.is_offload_optimizer:
                #     load_fsdp_optimizer(optimizer=self.optimizer, device_id=get_device_id())
                #     logger.info("Loaded drafter optimizer to GPU for training")
                # else:
                #     # Ensure optimizer state tensors reside on the local CUDA device before stepping
                #     opt_state_param = next(iter(self.optimizer.state.values()), None)
                #     if opt_state_param:
                #         for key, value in opt_state_param.items():
                #             if isinstance(value, torch.Tensor) and value.device.type != "cuda":
                #                 load_fsdp_optimizer(optimizer=self.optimizer, device_id=get_device_id())
                #                 logger.info("Moved drafter optimizer state to GPU for training")
                #                 break

            # Store the device mesh but don't rely on it for distributed operations
            # that might fail due to workers leaving/joining at different times
            self.training_device_mesh = device_mesh

            # Check if we can actually use the device mesh for coordination
            # NOTE: We skip the barrier here because workers may join at slightly different times
            # and the barrier could cause deadlock. FSDP2 will handle its own synchronization.
            if device_mesh.size() > 1:
                logger.debug(f"Training with device_mesh={device_mesh} (skipping init barrier to avoid deadlock)")

            self._training_initialized = True
            self._training_active = True

            logger.debug(f"Drafter training activated with device_mesh={device_mesh}, training_ranks={training_ranks}")
            logger.debug(
                f"[EagleTrainer rank {getattr(self, 'rank', -1)}] activate_training_model success "
                f"elapsed={time.time() - start_ts:.2f}s"
            )
            return True

        except Exception as e:
            logger.error(f"[EagleTrainer rank {getattr(self, 'rank', -1)}] activate_training_model failed: {e}")
            return False

    def collect_online_data(self, batch: dict[str, torch.Tensor], hidden_states: list[torch.Tensor]):
        """Collect online data from inference for Eagle training.

        This method collects data both to the local collected_data deque (for immediate use)
        and to the DataBuffer (for cross-step data accumulation).
        """
        input_ids = batch.get("input_ids")
        responses = batch.get("responses")
        prompts = batch.get("prompts")
        pad_token_id = getattr(self.model_config, "pad_token_id", 0) if self.model_config else 0

        if input_ids is None or input_ids.dim() != 2:
            logger.warning(
                f"[Rank {self.rank}] Non-batched data or wrong dimensions. input_ids dim: {input_ids.dim() if input_ids is not None else None}"
            )
            return

        # Add batch to DataBuffer for cross-step accumulation
        self.data_buffer.add_batch(batch, hidden_states)

        batch_size = input_ids.size(0)
        for i in range(batch_size):
            # Sample hidden states
            if isinstance(hidden_states, list) and len(hidden_states) > i:
                h_state = hidden_states[i]
                if h_state.dim() == 1:
                    h_state = h_state.unsqueeze(0)
                elif h_state.dim() > 2:
                    h_state = h_state.view(-1, h_state.size(-1))
            else:
                # logger.debug(f"Missing hidden states for sample {i}")
                continue

            seq = input_ids[i]

            # Loss mask: 1.0 over response tokens
            loss_mask = torch.zeros_like(seq, dtype=torch.float32)
            if prompts is not None and responses is not None:
                prompt_len = prompts[i].size(0)
                response_len = responses[i].size(0)
                for j in range(response_len):
                    if responses[i][j] != pad_token_id:
                        loss_mask[prompt_len + j] = 1.0
            elif responses is not None:
                response_start = seq.size(0) - responses[i].size(0)
                response_mask = (responses[i] != pad_token_id).float()
                loss_mask[response_start:] = response_mask

            # Ensure hidden states align with sequence length
            seq_len = seq.size(0)
            if h_state.size(0) < seq_len:
                pad_len = seq_len - h_state.size(0)
                padding = torch.zeros(pad_len, h_state.size(1), dtype=h_state.dtype)
                h_state = torch.cat([h_state, padding], dim=0)
            elif h_state.size(0) > seq_len:
                h_state = h_state[:seq_len]

            item = {
                "input_ids": seq.detach().cpu(),
                "loss_mask": loss_mask.detach().cpu(),
                "hidden_states": h_state.detach().cpu(),
            }
            self.collected_data.append(item)

    def _prepare_training_batch(
        self, use_buffer_data: bool = True, buffer_steps: int = 2
    ) -> Optional[dict[str, torch.Tensor]]:
        """Prepare a batch for training using Ulysses SP to remove padding.

        Args:
            use_buffer_data: If True, use data from DataBuffer (across multiple RL steps)
            buffer_steps: Number of recent RL steps to include data from (only used if use_buffer_data=True)

        Returns:
            Dictionary containing batch tensors for training
        """
        effective_batch_size = min(self.batch_size, 4)

        # Determine data source: DataBuffer (cross-step) or collected_data (current step only)
        if use_buffer_data and len(self.data_buffer) > 0:
            # Use data from last N RL steps via DataBuffer
            available_data = self.data_buffer.get_data_from_last_n_steps(buffer_steps)
            if len(available_data) < effective_batch_size:
                if 0 < len(available_data) >= min(2, effective_batch_size // 2):
                    items = available_data
                else:
                    return None
            else:
                # Randomly sample from available data to ensure diversity
                import random

                items = random.sample(available_data, min(len(available_data), effective_batch_size))
        else:
            # Fall back to current step data only
            if len(self.collected_data) < effective_batch_size:
                if 0 < len(self.collected_data) >= min(2, effective_batch_size // 2):
                    items = list(self.collected_data)
                else:
                    return None
            else:
                items = list(self.collected_data)[:effective_batch_size]

        # Filter out items without hidden_states (defensive check)
        items = [item for item in items if "hidden_states" in item]
        if len(items) == 0:
            logger.warning(f"[Rank {self.rank}] No items with hidden_states found, cannot prepare batch")
            return None
        elif len(items) < min(2, effective_batch_size // 2):
            logger.warning(
                f"[Rank {self.rank}] Only {len(items)} items with hidden_states found "
                f"(need at least {min(2, effective_batch_size // 2)}), cannot prepare batch"
            )
            return None

        pad_id = int(getattr(self.model_config, "pad_token_id", 0) or 0)
        dev = next(self.model.parameters()).device

        # Collect sequences to concatenate (removing padding)
        input_ids_list = []
        loss_mask_list = []
        hidden_states_list = []

        for item in items:
            full_len = item["input_ids"].numel()

            # Compute loss_mask if not present (for DataBuffer items)
            if "loss_mask" not in item:
                item_loss_mask = torch.zeros_like(item["input_ids"], dtype=torch.float32)
                if "prompts" in item and "responses" in item:
                    prompt_len = item["prompts"].size(0)
                    response_len = item["responses"].size(0)
                    for j in range(response_len):
                        if item["responses"][j] != pad_id:
                            item_loss_mask[prompt_len + j] = 1.0
                elif "responses" in item:
                    response_start = full_len - item["responses"].size(0)
                    response_mask = (item["responses"] != pad_id).float()
                    item_loss_mask[response_start:] = response_mask
                else:
                    # If no response info, assume all tokens are valid
                    item_loss_mask[:] = 1.0
            else:
                item_loss_mask = item["loss_mask"]

            # Limit sequence length to 512 max
            max_len = min(full_len, 512)

            # Select window around response tokens
            nonzero = torch.nonzero(item_loss_mask).flatten().cpu()
            if nonzero.numel() > 0:
                resp_start_idx = int(nonzero[0].item())
                resp_end_idx = int(nonzero[-1].item()) + 1
                window_span = max_len
                start = max(0, min(resp_start_idx, full_len - window_span))
                if resp_end_idx - start > window_span:
                    start = resp_end_idx - window_span
                end = min(full_len, start + window_span)
            else:
                start = max(0, full_len - max_len)
                end = full_len

            # Extract the window
            seq_input_ids = item["input_ids"][start:end].to(dev, non_blocking=True)
            seq_loss_mask = item_loss_mask[start:end].to(dev, non_blocking=True)

            h_states = item["hidden_states"].to(dev, dtype=torch.bfloat16, non_blocking=True)

            # Extract hidden states for the window
            h_seq_len = h_states.size(0)
            window_len = end - start
            if h_seq_len < window_len:
                # Pad hidden states if needed
                if h_seq_len > 0:
                    pad_len = window_len - h_seq_len
                    padding = torch.zeros(pad_len, h_states.size(-1), dtype=h_states.dtype, device=dev)
                    seq_hidden_states = torch.cat([h_states, padding], dim=0)
                else:
                    continue  # Skip this item if no hidden states
            else:
                # Extract the corresponding window from hidden states
                if start < h_seq_len:
                    actual_end = min(h_seq_len, start + window_len)
                    seq_hidden_states = h_states[start:actual_end]
                    if seq_hidden_states.size(0) < window_len:
                        # Pad if extracted less than window_len
                        pad_len = window_len - seq_hidden_states.size(0)
                        padding = torch.zeros(pad_len, h_states.size(-1), dtype=h_states.dtype, device=dev)
                        seq_hidden_states = torch.cat([seq_hidden_states, padding], dim=0)
                else:
                    # Use the last window_len tokens
                    seq_hidden_states = h_states[-window_len:]

            input_ids_list.append(seq_input_ids)
            loss_mask_list.append(seq_loss_mask)
            hidden_states_list.append(seq_hidden_states)

        if len(input_ids_list) == 0:
            return None

        # Concatenate all sequences into a single sequence (removing padding between samples)
        input_ids_concat = torch.cat(input_ids_list, dim=0).unsqueeze(0)  # (1, total_seq_len)
        loss_mask_concat = torch.cat(loss_mask_list, dim=0).unsqueeze(0)  # (1, total_seq_len)
        hidden_states_concat = torch.cat(hidden_states_list, dim=0).unsqueeze(0)  # (1, total_seq_len, hidden_dim)

        # Create attention mask (all 1s since no padding)
        total_seq_len = input_ids_concat.size(1)
        attn_mask = torch.ones((1, total_seq_len), dtype=torch.long, device=dev)

        # Use Ulysses SP to pad and slice if needed
        if self.use_ulysses_sp:
            # Pad to be divisible by SP size and slice across ranks
            input_ids_concat, _, pad_size = ulysses_pad_and_slice_inputs(
                input_ids_concat, position_ids_rmpad=None, sp_size=self.ulysses_sequence_parallel_size
            )
            # Pad loss_mask and hidden_states to match
            if pad_size > 0:
                loss_mask_concat = torch.nn.functional.pad(loss_mask_concat, (0, pad_size), value=0.0)
                hidden_states_concat = torch.nn.functional.pad(hidden_states_concat, (0, 0, 0, pad_size), value=0.0)
                attn_mask = torch.nn.functional.pad(attn_mask, (0, pad_size), value=0)

            # Slice for this rank
            from verl.utils.ulysses import slice_input_tensor

            loss_mask_concat = slice_input_tensor(loss_mask_concat, dim=1, padding=False)
            hidden_states_concat = slice_input_tensor(hidden_states_concat, dim=1, padding=False)
            attn_mask = slice_input_tensor(attn_mask, dim=1, padding=False)

            # Store pad_size for later gathering
            self._current_pad_size = pad_size
        else:
            self._current_pad_size = 0

        # Shift for next token prediction
        target = hidden_states_concat[:, 1:].contiguous()
        loss_mask = loss_mask_concat[:, 1:].contiguous()
        input_ids = input_ids_concat[:, :-1].contiguous()
        attn_mask = attn_mask[:, :-1].contiguous()
        base_h = hidden_states_concat[:, :-1].contiguous()

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "hidden_states": base_h,
            "target": target,
            "loss_mask": loss_mask,
        }

    async def training_step(self, step: int) -> bool:
        try:
            with torch.enable_grad():
                return await self._training_step_impl(step)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Training step {step} failed with error: {e}")
            return False

    async def _training_step_impl(self, step: int) -> bool:
        """Execute a single training step."""
        if not self.model:
            logger.warning("No model available for training")
            return False

        # Skip training if we're not collecting hidden states (since we can't train without them)
        collect_hidden_states_from_sgl = bool(self.config.get("collect_hidden_states_from_sgl", False))
        if not collect_hidden_states_from_sgl:
            logger.debug(
                f"[EagleTrainer rank {self.rank}] Skipping training step {step} "
                f"because collect_hidden_states_from_sgl=False"
            )
            return False

        batch = self._prepare_training_batch()
        if batch is None:
            logger.debug(
                f"[EagleTrainer rank {self.rank}] Not enough data at step {step} "
                f"(have={len(self.collected_data)} need≥{min(self.batch_size, 4)})"
            )
            return False

        # Optional barrier only if explicitly enabled
        if self.enable_mesh_barrier and self.training_device_mesh is not None and self.training_device_mesh.size() > 1:
            try:
                logger.debug(f"[EagleTrainer rank {self.rank}] Barrier enter (step={step})")
                torch.distributed.barrier(self.training_device_mesh.get_group())
                logger.debug(f"[EagleTrainer rank {self.rank}] Barrier exit (step={step})")
            except Exception as e:
                logger.warning(f"Training barrier failed at step {step}: {e}")

        self.model.train()
        self.optimizer.zero_grad()

        bsz, seqlen = batch["input_ids"].shape

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                base_model_hidden_states=batch["hidden_states"],
                output_hidden_states=True,
            )

        logits = getattr(outputs, "logits", None)
        hs = getattr(outputs, "hidden_states", None)
        hidden_states = hs[-1] if (hs is not None and len(hs) > 0) else None
        if hidden_states is None:
            hidden_states = batch["hidden_states"]

        # If logits are None, compute from hidden states
        if logits is None:
            logits = self.model.lm_head(hidden_states)  # type: ignore[attr-defined]

        # Gather outputs if using Ulysses SP
        if self.use_ulysses_sp:
            # Gather hidden_states and logits from all SP ranks
            hidden_states = gather_outputs_and_unpad(
                hidden_states.squeeze(0),
                gather_dim=0,
                unpad_dim=0,
                padding_size=self._current_pad_size,
            ).unsqueeze(0)

            logits = gather_outputs_and_unpad(
                logits.squeeze(0), gather_dim=0, unpad_dim=0, padding_size=self._current_pad_size
            ).unsqueeze(0)

            # Also gather target and loss_mask
            target = gather_outputs_and_unpad(
                batch["target"].squeeze(0),
                gather_dim=0,
                unpad_dim=0,
                padding_size=self._current_pad_size,
            ).unsqueeze(0)

            loss_mask_2d = gather_outputs_and_unpad(
                batch["loss_mask"].squeeze(0),
                gather_dim=0,
                unpad_dim=0,
                padding_size=self._current_pad_size,
            ).unsqueeze(0)
        else:
            target = batch["target"]
            loss_mask_2d = batch["loss_mask"]

        # Ensure correct shapes
        if loss_mask_2d.dim() != 2:
            loss_mask_2d = loss_mask_2d.view(bsz, -1)

        # Compute losses
        out_logp = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            target_logits = self.model.lm_head(target)  # type: ignore[attr-defined]
            target_p = F.softmax(target_logits, dim=-1).detach()

        plogp = target_p * out_logp
        num_valid_tokens = torch.sum(loss_mask_2d).item()
        if num_valid_tokens > 0:
            ploss = -torch.sum(loss_mask_2d * torch.sum(plogp, dim=-1)) / num_valid_tokens
        else:
            ploss = torch.sum(out_logp * 0.0)

        vloss = self.criterion(hidden_states, target)  # [B,T,H]
        if num_valid_tokens > 0:
            vloss = torch.sum(loss_mask_2d * torch.mean(vloss, dim=-1)) / num_valid_tokens
        else:
            vloss = torch.sum(hidden_states * 0.0)

        w_v = float(self.config.get("vloss_weight", 0.5))
        w_p = float(self.config.get("ploss_weight", 0.5))
        loss = w_v * vloss + w_p * ploss

        loss.backward()
        # Extra debug: gradient presence (first param)
        first_p = next(self.model.parameters(), None)
        if first_p is not None and first_p.grad is None:
            logger.debug(f"[EagleTrainer rank {self.rank}] step={step} first param grad is None")
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.training_steps += 1
        if self.training_steps % 10 == 0:
            logger.info(
                f"Step {self.training_steps}: loss={float(loss.item()):.4f}, vloss={float(vloss.item()):.4f}, ploss={float(ploss.item()):.4f}"
            )

        if self.checkpoint_dir and (step // 100) > self._last_ckpt_step:
            # Wait for previous checkpoint to complete before starting a new one
            # This avoids queuing multiple checkpoints and excessive memory usage
            if self._pending_checkpoint_future is not None:
                try:
                    self._pending_checkpoint_future.result()
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Previous checkpoint save failed: {e}")

            # Launch async checkpoint save without blocking training
            self._pending_checkpoint_future = self._save_checkpoint_async(step, is_final=False)
            self._last_ckpt_step = step // 100

        return True

    def get_model_state_dict(self) -> Optional[dict[str, torch.Tensor]]:
        """Get trainable model state dict (excluding frozen layers)."""
        if not self.model:
            return None
        trainable_state = self._get_trainable_state_dict()
        return {k: v.detach().cpu() for k, v in trainable_state.items() if v.requires_grad}

    def increment_rl_step(self):
        """Increment the RL step counter in the data buffer.

        Should be called at the end of each RL training step to mark the boundary.
        """
        self.data_buffer.increment_step()
        logger.debug(
            f"[Rank {self.rank}] DataBuffer RL step incremented to {self.data_buffer.get_current_step()}, "
            f"total samples: {len(self.data_buffer)}"
        )

    async def cleanup_training(self):
        # First set training as inactive to prevent further steps
        self._training_active = False

        # Wait for any pending async checkpoint save to complete
        if self._pending_checkpoint_future is not None:
            logger.debug(f"[Rank {self.rank}] Waiting for pending checkpoint save to complete...")
            try:
                # Run the blocking .result() call in executor to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._pending_checkpoint_future.result)
                logger.debug(f"[Rank {self.rank}] Pending checkpoint save completed")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Pending checkpoint save failed: {e}")
            self._pending_checkpoint_future = None

        # Save final checkpoint and wait for it to complete
        if self.checkpoint_dir and self.model is not None:
            final_future = self._save_checkpoint_async(self.training_steps, is_final=True)
            if final_future is not None:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, final_future.result)
                    logger.info(f"[Rank {self.rank}] Final checkpoint save completed")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Final checkpoint save failed: {e}")

        # Clean up distributed resources gracefully
        if self.training_device_mesh is not None:
            try:
                # Give a moment for any pending operations to complete
                await asyncio.sleep(0.1)
                if self.training_device_mesh.size() > 1:
                    # Try to destroy the process group if possible
                    try:
                        # Run barrier with timeout to avoid hanging
                        await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, lambda: torch.distributed.barrier(self.training_device_mesh.get_group())
                            ),
                            timeout=5.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Rank {self.rank} barrier timeout during cleanup, continuing anyway")
                    except Exception:
                        pass  # Ignore barrier errors during cleanup
            except Exception as e:
                logger.debug(f"Process group cleanup error (expected): {e}")

        if self.model is not None:
            try:
                offload_fsdp_model_to_cpu(self.model)
                logger.debug("Offloaded drafter model to CPU after training")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to offload drafter model during cleanup: {e}")

        if self.optimizer is not None:
            try:
                offload_fsdp_optimizer(self.optimizer)
                logger.debug("Offloaded drafter optimizer state to CPU after training")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to offload drafter optimizer during cleanup: {e}")

        self.collected_data.clear()
        self.data_buffer.clear()  # Clear the cross-step data buffer
        self.training_device_mesh = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._training_initialized = False
        self.training_steps = 0

    @property
    def is_training_initialized(self) -> bool:
        return self._training_initialized

    @property
    def is_training_active(self) -> bool:
        return self._training_active

    def set_training_active(self, active: bool):
        self._training_active = active