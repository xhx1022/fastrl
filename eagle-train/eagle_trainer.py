import os
import logging
import json
import gc
import deepspeed
import argparse

from safetensors.torch import safe_open
from accelerate.utils import set_seed
from tqdm import tqdm
from transformers import AutoConfig

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model.llama_eagle import LlamaForCausalLMEagle
from model.qwen2_eagle import Qwen2ForCausalLMEagle
from utils import Tracking

set_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger(__file__)

def add_args():
    parser = argparse.ArgumentParser(description="Eagle Trainer DeepSpeed")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--project_name", type=str, default="Eagle-Trainer", help="WandB project name")
    parser.add_argument("--experiment_name", type=str, default="eagle-deepspeed", help="WandB experiment name")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Number of warmup steps")
    parser.add_argument("--grad_clip", type=float, default=0.5, help="Gradient clipping value")
    parser.add_argument("--value_weight", type=float, default=1.0, help="Weight for value loss")
    parser.add_argument("--prob_weight", type=float, default=0.1, help="Weight for probability loss")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--load_optimizer", action="store_true", help="Whether to load optimizer states")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16"], help="Training precision type")
    parser = deepspeed.add_config_arguments(parser)
    return parser


class EagleDataset(Dataset):
    def __init__(self, datapath, transform=None, max_len=2048, dataset_max_len=None, hidden_states_idx=None):
        """Initialize EagleDataset to load pre-processed data"""
        self.datapath = datapath
        self.transform = transform
        self.max_len = max_len
        self.global_max_seq_len = dataset_max_len
        self.hidden_states_idx = hidden_states_idx
        self.failed_indices = set()  # Track failed loads

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, idx):
        if idx in self.failed_indices:
            return self.__getitem__((idx + 1) % len(self.datapath))

        try:
            data = torch.load(self.datapath[idx], weights_only=True)
            data["loss_mask"][-1] = 0
            processed_item = {
                "input_ids": data["input_ids"][1:],
                "hidden_states": data["hidden_state"],
                "target": data["hidden_state"][1:],  # Shift right by 1 for target
                "loss_mask": data["loss_mask"],
                "max_seq_len": self.global_max_seq_len,
            }

            if self.transform:
                processed_item = self.transform(processed_item)

            return processed_item
        except Exception as e:
            self.failed_indices.add(idx)
            if dist.get_rank() == 0:
                logger.warning(f"Error loading file {self.datapath[idx]}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self.datapath))


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_states"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_states"] = noisy_tensor
        return data


class EagleDataCollator:
    def padding_tensor(self, intensors, N):
        assert len(intensors.shape) == 1 or len(intensors.shape) == 2
        intensors = intensors.unsqueeze(0)
        B, n = intensors.shape[:2]
        padding_shape = [B, N - n]
        if len(intensors.shape) == 3:
            padding_shape.append(intensors.shape[2])
        padding_tensor = torch.zeros(padding_shape, dtype=intensors.dtype, device=intensors.device)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features):
        # Determine max length from the batch if max_seq_len is not set
        max_length = features[0]["max_seq_len"]
        if max_length is None:
            max_length = max(len(item["input_ids"]) for item in features)
            logger.warning(f"max_seq_len not set in dataset, using batch maximum length: {max_length}")

        device = features[0]["input_ids"].device

        batch_attention_mask = torch.cat(
            [
                torch.cat(
                    [
                        torch.ones(len(item["input_ids"]) + 1, device=device),
                        torch.zeros(max_length - len(item["input_ids"]) - 1, device=device),
                    ]
                ).unsqueeze(0)
                for item in features
            ]
        )
        batch_input_ids = torch.cat([self.padding_tensor(item["input_ids"], max_length) for item in features])
        batch_hidden_states = torch.cat([self.padding_tensor(item["hidden_states"], max_length) for item in features])
        batch_target = torch.cat([self.padding_tensor(item["target"], max_length) for item in features])
        batch_loss_mask = torch.cat(
            [
                torch.cat([item["loss_mask"], torch.zeros(max_length - len(item["loss_mask"]), device=device)]).unsqueeze(0)
                for item in features
            ]
        )

        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "loss_mask": batch_loss_mask,
            "attention_mask": batch_attention_mask,
        }
        return batch


class EagleTrainerDeepSpeed:
    def __init__(self, args):
        self.args = args
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.start_epoch = 0  # Track starting epoch for resume

        # Initialize model and datasets
        self._build_dataloader()
        self._build_model()
        self._initialize_deepspeed()

        self._load_checkpoint()

        # Initialize loss functions
        self.criterion = nn.SmoothL1Loss(reduction="none")

    def _load_checkpoint(self):
        if os.path.exists(os.path.join(self.args.output_dir, "latest")):
            load_path, client_state = self.model_engine.load_checkpoint(
                self.args.output_dir,
                load_optimizer_states=self.args.load_optimizer,
                load_lr_scheduler_states=True,
                load_module_only=False,
            )

            self.start_epoch = client_state["epoch"] + 1
            steps_per_epoch = len(self.train_loader)
            self.start_epoch = client_state["step"] // steps_per_epoch + 1

            if self.rank == 0:
                logger.info(f"Successfully loaded checkpoint from: {load_path}")
                logger.info(f"Resuming training from epoch {self.start_epoch}")
                if client_state:
                    logger.info(f"Client state keys: {list(client_state.keys())}")

    def _build_model(self):
        # Load model config
        config = AutoConfig.from_pretrained(self.args.base_model_path, trust_remote_code=True)
        config.num_hidden_layers = 1
        config.dtype = torch.bfloat16 if self.args.precision == "bf16" else torch.float16
        config._attn_implementation = "flash_attention_2"

        # Determine model class based on config
        model_class = None
        if hasattr(config, "model_type"):
            if config.model_type.lower() == "llama":
                model_class = LlamaForCausalLMEagle
            elif config.model_type.lower() == "qwen2":
                model_class = Qwen2ForCausalLMEagle
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
        else:
            if "llama" in self.args.base_model_path.lower():
                model_class = LlamaForCausalLMEagle
            elif "qwen" in self.args.base_model_path.lower():
                model_class = Qwen2ForCausalLMEagle
            else:
                raise ValueError("Could not determine model type")

        self.model = model_class(config=config)

        # Load embeddings and LM head from base model
        self._load_base_model_weights()
        self.model.to(dtype=config.dtype)

    def _load_base_model_weights(self):
        base_path = self.args.base_model_path
        device = f"cuda:{self.local_rank}"
        dtype = torch.bfloat16 if self.args.precision == "bf16" else torch.float16

        try:
            # Try loading from safetensors first
            with open(os.path.join(base_path, "model.safetensors.index.json"), "r") as f:
                index_json = json.loads(f.read())
                head_path = index_json["weight_map"]["lm_head.weight"]
                embed_path = index_json["weight_map"]["model.embed_tokens.weight"]

            with safe_open(os.path.join(base_path, head_path), framework="pt", device=device) as f:
                tensor_slice = f.get_slice("lm_head.weight")
                vocab_size, hidden_dim = tensor_slice.get_shape()
                lm_head_weight = tensor_slice[:, :hidden_dim].to(dtype)

            with safe_open(os.path.join(base_path, embed_path), framework="pt", device=device) as f:
                tensor_slice = f.get_slice("model.embed_tokens.weight")
                vocab_size, hidden_dim = tensor_slice.get_shape()
                embed_weight = tensor_slice[:, :hidden_dim].to(dtype)
        except:
            # Fallback to pytorch model files
            with open(os.path.join(base_path, "pytorch_model.bin.index.json"), "r") as f:
                index_json = json.loads(f.read())
                head_path = index_json["weight_map"]["lm_head.weight"]
                embed_path = index_json["weight_map"]["model.embed_tokens.weight"]

            head_weights = torch.load(os.path.join(base_path, head_path))
            embed_weights = torch.load(os.path.join(base_path, embed_path))
            lm_head_weight = head_weights["lm_head.weight"].to(dtype).to(device)
            embed_weight = embed_weights["model.embed_tokens.weight"].to(dtype).to(device)

        # Copy weights to model
        self.model.lm_head.weight.data.copy_(lm_head_weight)
        self.model.model.embed_tokens.weight.data.copy_(embed_weight)

        # Freeze embedding layer and LM head
        for param in self.model.model.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.model.lm_head.parameters():
            param.requires_grad = False

        del embed_weight, lm_head_weight
        torch.cuda.empty_cache()
        gc.collect()

    def _build_dataloader(self):
        # First, check if the data path exists
        if not os.path.exists(self.args.data_path):
            raise ValueError(f"Data path does not exist: {self.args.data_path}")

        logger.info(f"Searching for data files in: {self.args.data_path}")

        # List all subdirectories
        dir_list = [d for d in os.listdir(self.args.data_path) if os.path.isdir(os.path.join(self.args.data_path, d))]
        dir_list.sort()
        logger.info(f"Found directories: {dir_list}")

        # Determine max sequence length from directory name if possible
        dataset_max_len = 2048  # Default max length
        for dirname in dir_list:
            if "-" in dirname:
                try:
                    max_k_value = dirname.split("-")[1].strip()
                    if max_k_value.endswith("K"):
                        dataset_max_len = int(float(max_k_value[:-1]) * 1024)
                    else:
                        dataset_max_len = int(max_k_value)
                    break
                except:
                    continue

        logger.info(f"Using max sequence length: {dataset_max_len}")

        # Collect all .pt files
        datapath = []
        for root, dirs, files in os.walk(self.args.data_path):
            for file in files:
                if file.endswith(".ckpt") or file.endswith(".pt"):
                    file_path = os.path.join(root, file)
                    datapath.append(file_path)

        if not datapath:
            raise ValueError(f"No .pt files found in {self.args.data_path} or its subdirectories")

        logger.info(f"Total number of .pt files found: {len(datapath)}")

        # Split into train and validation
        train_size = int(len(datapath) * 0.95)
        train_files = datapath[:train_size]
        val_files = datapath[train_size:]

        # Create datasets with proper max_len
        aug = AddUniformNoise(std=0.2)
        self.train_dataset = EagleDataset(train_files, transform=aug, dataset_max_len=dataset_max_len)
        self.val_dataset = EagleDataset(val_files, dataset_max_len=dataset_max_len)

        if self.rank == 0:
            logger.info(f"Found {len(train_files)} training files and {len(val_files)} validation files")
            # Log some example file paths for verification
            logger.info("Example training file paths:")
            for path in train_files[:3]:
                logger.info(f"  {path}")
            if val_files:
                logger.info("Example validation file paths:")
                for path in val_files[:3]:
                    logger.info(f"  {path}")

        # Verify data loading by trying to load the first file
        if train_files:
            try:
                first_file = train_files[0]
                data = torch.load(first_file, weights_only=True)
                logger.info(f"Successfully loaded first data file: {first_file}")
                logger.info(f"Data keys: {list(data.keys())}")
                # Also verify sequence lengths
                if "input_ids" in data:
                    logger.info(f"First file input_ids length: {len(data['input_ids'])}")
            except Exception as e:
                logger.error(f"Failed to load first data file {first_file}: {str(e)}")
                raise

    def _initialize_deepspeed(self):
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        # Initialize DeepSpeed engine
        self.model_engine, self.optimizer, self.train_loader, _ = deepspeed.initialize(
            args=self.args,
            model=self.model,
            model_parameters=parameters,
            training_data=self.train_dataset,
            collate_fn=EagleDataCollator(),
        )

        # Create validation dataloader
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=EagleDataCollator(), num_workers=4
        )

    def _compute_loss(self, batch):
        outputs = self.model_engine(
            batch["hidden_states"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            use_cache=False, 
        )

        # Extract hidden states from model output
        predict = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs[0]

        with torch.no_grad():
            # Ensure target is in the same dtype as model
            target = batch["target"].to(dtype=self.model.config.dtype)
            target_head = self.model.lm_head(target)
            target_p = nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()

        out_head = self.model.lm_head(predict)
        out_logp = nn.LogSoftmax(dim=2)(out_head)

        loss_mask = batch["loss_mask"][:, :, None]

        # Compute policy loss
        plogp = target_p * out_logp
        ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.shape[0] * loss_mask.shape[1])

        # Compute value loss
        vloss = self.criterion(predict, target)
        vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.shape[0] * loss_mask.shape[1])

        # Combined loss
        loss = self.args.value_weight * vloss + self.args.prob_weight * ploss

        return loss, vloss, ploss, out_head, target_head

    def _compute_metrics(self, out_head, target_head, loss_mask):
        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)

            correct = ((predicted == target) * loss_mask.squeeze()).sum().item()
            total = loss_mask.sum().item()

            # Calculate top-k accuracy
            out_head_flat = out_head.view(-1, out_head.size(-1))[loss_mask.view(-1) == 1]
            target_flat = target.view(-1)[loss_mask.view(-1) == 1]

            maxk = 3
            _, pred = out_head_flat.topk(maxk, 1, True, True)
            pred = pred.t()
            correct_k = pred.eq(target_flat.view(1, -1).expand_as(pred))

            top_k_acc = []
            for k in range(1, maxk + 1):
                top_k_acc.append(correct_k[:k].reshape(-1).float().sum(0, keepdim=True).item())

        return correct, total, top_k_acc

    def train(self):
        if self.rank == 0:
            tracking = Tracking(
                project_name=self.args.project_name, experiment_name=self.args.experiment_name, default_backend="wandb"
            )
        else:
            tracking = None

        for epoch in range(self.start_epoch, self.args.epochs):
            self.model_engine.train()

            if self.rank == 0:
                train_iter = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            else:
                train_iter = self.train_loader

            for batch_idx, batch in enumerate(train_iter):
                batch = {k: v.cuda() for k, v in batch.items()}
                batch["hidden_states"] = batch["hidden_states"].to(dtype=self.model.config.dtype)
                batch["target"] = batch["target"].to(dtype=self.model.config.dtype)

                loss, vloss, ploss, out_head, target_head = self._compute_loss(batch)

                self.model_engine.backward(loss)
                self.model_engine.step()

                correct, total, top_k_acc = self._compute_metrics(out_head, target_head, batch["loss_mask"])

                if self.rank == 0:
                    global_step = epoch * len(self.train_loader) + batch_idx
                    metrics = {
                        "train/loss": loss.item(),
                        "train/vloss": vloss.item(),
                        "train/ploss": ploss.item(),
                        "train/acc": correct / (total + 1e-5),
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                    }

                    for k, acc in enumerate(top_k_acc):
                        metrics[f"train/top_{k+1}_acc"] = acc / (total + 1e-5)

                    tracking.log(metrics, step=global_step)
                    train_iter.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total):.2%}", "epoch": epoch})

            # self.validate(epoch, tracking)

            client_state = {
                "epoch": epoch,
                "step": epoch * len(self.train_loader) + len(self.train_loader) - 1,
            }
            self.model_engine.save_checkpoint(self.args.output_dir, client_state=client_state, exclude_frozen_parameters=False)

            if self.rank == 0:
                logger.info(f"Checkpoint saved at epoch {epoch}: {self.args.output_dir}")

    def validate(self, epoch, tracking):
        self.model_engine.eval()
        val_metrics = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.cuda() for k, v in batch.items()}
                batch["hidden_states"] = batch["hidden_states"].to(dtype=self.model.config.dtype)
                batch["target"] = batch["target"].to(dtype=self.model.config.dtype)
                loss, vloss, ploss, out_head, target_head = self._compute_loss(batch)
                correct, total, top_k_acc = self._compute_metrics(out_head, target_head, batch["loss_mask"])

                val_metrics.append(
                    {
                        "val/loss": loss.item(),
                        "val/vloss": vloss.item(),
                        "val/ploss": ploss.item(),
                        "val/acc": correct / (total + 1e-5),
                        "val/top_1_acc": top_k_acc[0] / (total + 1e-5),
                        "val/top_2_acc": top_k_acc[1] / (total + 1e-5),
                        "val/top_3_acc": top_k_acc[2] / (total + 1e-5),
                    }
                )

        if self.rank == 0 and val_metrics:
            # Average validation metrics
            avg_metrics = {k: sum(d[k] for d in val_metrics) / len(val_metrics) for k in val_metrics[0].keys()}
            if tracking is not None:
                tracking.log(avg_metrics, step=epoch)
            logger.info(f"Validation metrics at epoch {epoch}: {avg_metrics}")


def main():
    parser = add_args()
    args = parser.parse_args()

    # Initialize distributed training
    deepspeed.init_distributed()

    # Create trainer and start training
    trainer = EagleTrainerDeepSpeed(args)
    trainer.train()


if __name__ == "__main__":
    main()
