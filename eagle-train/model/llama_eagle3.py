import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm
from transformers.models.llama.modeling_llama import LlamaModel as LlamaModelTF


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaEagle3Attention(nn.Module):
    """Eagle3 specific attention module with custom cache mechanism"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # NOTE: Override the qkv projection for Eagle-3 (input is 2x hidden_size)
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

        # self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                from transformers.models.llama.modeling_llama import LlamaLinearScalingRotaryEmbedding

                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                from transformers.models.llama.modeling_llama import LlamaDynamicNTKScalingRotaryEmbedding

                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[list] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        """
        Eagle3 specific attention forward pass with custom cache mechanism
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        lck = len(cache_hidden[0]) if (cache_hidden and len(cache_hidden) >= 2 and len(cache_hidden[0]) > 0) else 0

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        total_seq_len = q_len + lck
        if total_seq_len > self.max_position_embeddings:
            total_seq_len = self.max_position_embeddings

        cos, sin = self.rotary_emb(query_states, seq_len=total_seq_len)
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)

        if position_ids is not None:
            max_pos = cos.shape[2] - 1
            safe_position_ids = torch.clamp(position_ids + lck, 0, max_pos)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, safe_position_ids)
        else:
            default_pos_ids = torch.arange(q_len, device=query_states.device, dtype=torch.long).unsqueeze(0)
            safe_position_ids = torch.clamp(default_pos_ids + lck, 0, cos.shape[2] - 1)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, safe_position_ids)

        # Repeat key and value states for grouped query attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if cache_hidden is not None:
            if len(cache_hidden) < 2:
                cache_hidden.extend([[] for _ in range(2 - len(cache_hidden))])

            cache_hidden[0].append(key_states)
            cache_hidden[1].append(value_states)

            cache_k = cache_hidden[0]
            cache_v = cache_hidden[1]
            num_caches = len(cache_k)

            if num_caches == 1:
                k0 = cache_k[0]
                v0 = cache_v[0]

                # query_states: [batch, heads, seq_len, head_dim]
                # k0: [batch, heads, seq_len, head_dim]
                attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(self.head_dim)

                if attention_mask is not None:
                    if attention_mask.dim() == 4:
                        attn_weights = attn_weights + attention_mask
                    elif attention_mask.dim() == 2:
                        expanded_mask = attention_mask[:, None, None, :].expand(bsz, 1, q_len, q_len)
                        attn_weights = attn_weights + expanded_mask

                # Apply softmax
                attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

                # Compute attention output
                attn_output = torch.matmul(attn_weights, v0)

            else:
                k0 = cache_k[0]
                v0 = cache_v[0]

                attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(self.head_dim)

                if attention_mask is not None:
                    if attention_mask.dim() == 4:
                        attn_weights = attn_weights + attention_mask
                    elif attention_mask.dim() == 2:
                        expanded_mask = attention_mask[:, None, None, :].expand(bsz, 1, q_len, q_len)
                        attn_weights = attn_weights + expanded_mask

                for i in range(1, num_caches):
                    ki = cache_k[i]
                    qi = query_states

                    if qi.shape == ki.shape:
                        attn_weightsi = (qi * ki).sum(-1) / math.sqrt(self.head_dim)
                        attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)

                # Apply softmax
                attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights0 = attn_weights[..., :q_len]

                # Compute attention output
                attn_output = torch.matmul(attn_weights0, v0)

                for i in range(1, num_caches):
                    vi = cache_v[i]
                    attn_weightsi = attn_weights[..., q_len + i - 1]
                    attn_outputi = attn_weightsi[..., None] * vi
                    attn_output = attn_output + attn_outputi

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

            return attn_output
        else:
            raise ValueError("cache_hidden must be provided for Eagle3")


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        # Use Eagle3 specific attention
        self.self_attn = LlamaEagle3Attention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # NOTE: Add a hidden_norm for Eagle-3
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[list] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        input_embeds = self.input_layernorm(input_embeds)
        hidden_states = self.hidden_norm(hidden_states)

        # NOTE: Concatenate the input_embeds and hidden_states for Eagle-3
        hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)

        # Self Attention with Eagle3 custom cache
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


class LlamaModelEagle3(LlamaModelTF):
    def __init__(self, config: LlamaConfig):
        nn.Module.__init__(self)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)

        d2t = torch.zeros((config.draft_vocab_size), dtype=torch.long)
        t2d = torch.zeros((config.vocab_size), dtype=torch.bool)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)

        # NOTE: Add a midlayer, fc for Eagle-3
        self.midlayer = LlamaDecoderLayer(config, 0)
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

    def _prepare_4d_attention_mask(self, attention_mask, input_shape, device, dtype):
        """
        Create 4D attention mask from 2D mask for causal attention
        """
        batch_size, seq_length = input_shape

        # Create causal mask
        causal_mask = torch.full((seq_length, seq_length), torch.finfo(dtype).min, device=device, dtype=dtype)
        mask_cond = torch.arange(seq_length, device=device)
        causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(seq_length, 1), 0)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_length)

        if attention_mask is not None:
            # Expand 2D attention mask to 4D
            if attention_mask.dim() == 2:
                expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length)
                # Invert mask (1 -> 0, 0 -> large negative)
                inverted_mask = (1.0 - expanded_mask).to(dtype)
                inverted_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
                causal_mask = causal_mask + inverted_mask

        return causal_mask

    def forward(
        self,
        base_model_hidden_states: torch.Tensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[object] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        prediction_length: Optional[int] = 1,
        target: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        hidden_states = base_model_hidden_states

        if self.training and self.gradient_checkpointing and not hidden_states.requires_grad:
            hidden_states.requires_grad = True

        hidden_states = self.fc(hidden_states)

        batch_size, seq_length, _ = hidden_states.shape

        loss_list = []
        accuracy_list = []

        cache_hidden = [[], []]

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        for idx in range(prediction_length):
            inputs_embeds = self.embed_tokens(input_ids)
            if self.training and self.gradient_checkpointing and not inputs_embeds.requires_grad:
                inputs_embeds.requires_grad = True
            inputs_embeds = inputs_embeds.to(base_model_hidden_states.dtype)

            current_seq_len = input_ids.shape[1]
            if attention_mask is not None:
                attention_mask_4d = self._prepare_4d_attention_mask(
                    attention_mask, (batch_size, current_seq_len), input_ids.device, inputs_embeds.dtype
                )
            else:
                attention_mask_4d = None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    self.midlayer.__call__,
                    inputs_embeds,
                    hidden_states,
                    cache_hidden,
                    attention_mask_4d,
                    position_ids,
                )
            else:
                layer_outputs = self.midlayer(
                    input_embeds=inputs_embeds,
                    hidden_states=hidden_states,
                    cache_hidden=cache_hidden,
                    attention_mask=attention_mask_4d,
                    position_ids=position_ids,
                )

            hidden_states_out = layer_outputs[0]
            hidden_states = hidden_states_out
            hidden_states_out = self.norm(hidden_states_out)

            # Loss computation (same as original)
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
                # Apply Eagle3 specific transformations
                input_ids = self._padding(input_ids, left=False)
                target = self._padding(target, left=False)
                loss_mask = self._padding(loss_mask, left=False)

                if attention_mask is not None:
                    attention_mask = self._padding(attention_mask, left=False)

                position_ids = position_ids + 1
                max_pos = self.config.max_position_embeddings - 1
                position_ids = torch.clamp(position_ids, 0, max_pos)

        return loss_list, accuracy_list
