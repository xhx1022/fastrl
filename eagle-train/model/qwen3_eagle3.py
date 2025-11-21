from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Model as Qwen3ModelTF,
    Qwen3RotaryEmbedding,
    Qwen3MLP,
    Qwen3Attention,
    Qwen3RMSNorm,
    Qwen3Config,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        # NOTE: Override the qkv projection for Eagle-3, Qwen 3 disenables bias by default
        self.self_attn.q_proj = nn.Linear(
            config.hidden_size * 2, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.self_attn.k_proj = nn.Linear(
            config.hidden_size * 2, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.self_attn.v_proj = nn.Linear(
            config.hidden_size * 2, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # NOTE: Add a hidden_norm for Eagle-3
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        input_embeds = self.input_layernorm(input_embeds)
        hidden_states = self.hidden_norm(hidden_states)

        # NOTE: Concatenate the input_embeds and hidden_states for Eagle-3
        hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen3ModelEagle3(Qwen3ModelTF):
    def __init__(self, config: Qwen3Config):
        # super().__init__(config)
        nn.Module.__init__(self)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)

        d2t = torch.zeros((config.draft_vocab_size), dtype=torch.long)
        t2d = torch.zeros((config.vocab_size), dtype=torch.bool)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)

        # NOTE: Add a midlayer, fc for Eagle-3
        self.midlayer = Qwen3DecoderLayer(config, 0)
        if hasattr(config, "target_hidden_size"):
            self.fc = torch.nn.Linear(config.target_hidden_size * 3, config.hidden_size, bias=False)
        else:
            self.fc = torch.nn.Linear(config.hidden_size * 3, config.hidden_size, bias=False)

        self.gradient_checkpointing = False

    @torch.no_grad()
    def _padding(self, tensor, left=True):
        """Utility function to pad tensors as used in Eagle3"""
        zeropadding = torch.zeros_like(tensor[:, -1:])
        if left:
            tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
        else:
            tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
        return tensor

    def forward(
        self,
        base_model_hidden_states: torch.Tensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # Eagle 3 Args: Test-Time Scaling
        prediction_length: Optional[int] = 1,
        target: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + target.shape[1], device=target.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = base_model_hidden_states

        if self.training and self.gradient_checkpointing and not hidden_states.requires_grad:
            hidden_states.requires_grad = True

        hidden_states = self.fc(hidden_states)

        batch_size, seq_length, _ = hidden_states.shape

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        loss_list = []
        accuracy_list = []

        for idx in range(prediction_length):
            inputs_embeds = self.embed_tokens(input_ids)
            if self.training and self.gradient_checkpointing and not inputs_embeds.requires_grad:
                inputs_embeds.requires_grad = True
            inputs_embeds = inputs_embeds.to(base_model_hidden_states.dtype)

            # Reset cache for each prediction step to prevent accumulation
            step_past_key_values = DynamicCache() if use_cache else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    self.midlayer.__call__,
                    inputs_embeds,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    step_past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = self.midlayer(
                    inputs_embeds,
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=step_past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

                hidden_states_out = layer_outputs[0]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = hidden_states_out
            hidden_states_out = self.norm(hidden_states_out)

            with torch.no_grad():
                target_head = target
                target_max_token = target_head.argmax(-1)
                target_mask = self.t2d[target_max_token]
                target_mask = target_mask[..., None].int()
                position_mask = target_mask * loss_mask
                target_head = target_head[..., self.t2d]
                target_head = target_head.float()
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()

            logits = self.lm_head(hidden_states_out)
            logits = logits.float()
            out_logp = nn.LogSoftmax(dim=2)(logits)
            plogp = target_p * out_logp
            loss = -torch.sum(position_mask * plogp, 2).mean()
            loss_list.append(loss)
            with torch.no_grad():
                accuracy_list.append(
                    ((logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)).sum().item()
                    / (loss_mask.sum().item() + 1e-6)
                )

            if idx < prediction_length - 1:
                input_ids = self._padding(input_ids, left=False)
                target = self._padding(target, left=False)
                loss_mask = self._padding(loss_mask, left=False)
                attention_mask = self._padding(attention_mask, left=False)

            # Clean up step cache to prevent accumulation
            del step_past_key_values

        return loss_list, accuracy_list
