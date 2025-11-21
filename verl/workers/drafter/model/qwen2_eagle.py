from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer as Qwen2DecoderLayerTF
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as Qwen2ModelTF
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging

logger = logging.get_logger(__name__)


class Qwen2DecoderLayer(Qwen2DecoderLayerTF):
    def __init__(
        self,
        config: Qwen2Config,
        layer_id: int = 0,
    ) -> None:
        super().__init__(config, layer_id)

        # NOTE: Skip the input_layernorm for Eagle-2. No need for Eagle-3
        if layer_id == 0:
            del self.input_layernorm
            self.input_layernorm = nn.Identity()


class Qwen2Model(Qwen2ModelTF):
    def __init__(self, config: Qwen2Config):
        # super().__init__(config)
        nn.Module.__init__(self)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.fc = torch.nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        # self.post_init()

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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Handle causal mask - compatibility for different transformers versions
        if hasattr(self, '_update_causal_mask'):
            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )
        else:
            # Fallback for older transformers versions
            # Create a simple causal mask
            batch_size, seq_length = inputs_embeds.shape[:2]
            device = inputs_embeds.device
            dtype = inputs_embeds.dtype
            
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.bool)
            
            # Create causal mask
            causal_mask = None
            if seq_length > 1:
                causal_mask = torch.triu(
                    torch.ones(seq_length, seq_length, device=device, dtype=torch.bool),
                    diagonal=1
                )
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)

        hidden_states = inputs_embeds

        hidden_states = self.fc(torch.cat((hidden_states, base_model_hidden_states), dim=-1))

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                # Only pass flash_attn_kwargs if using flash attention
                if hasattr(self.config, '_attn_implementation') and self.config._attn_implementation == 'flash_attention_2':
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # NOTE: Skip the final norm for Eagle-2. No need for Eagle-3
        # hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class KwargsForCausalLM(FlashAttentionKwargs, TransformersKwargs): ...


class Qwen2ForCausalLMEagle(Qwen2ForCausalLM):
    def __init__(self, config: Qwen2Config) -> None:
        # Disable flash attention to avoid device-side assert issues
        if not hasattr(config, '_attn_implementation'):
            config._attn_implementation = 'eager'
        elif config._attn_implementation == 'flash_attention_2':
            config._attn_implementation = 'eager'
        super().__init__(config)
        self.model = Qwen2Model(config)
        # self.post_init()

    def forward(
        self,
        base_model_hidden_states: torch.Tensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            base_model_hidden_states=base_model_hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        if logits_to_keep == 0:
            # If logits_to_keep is 0, compute logits for all positions
            logits = self.lm_head(hidden_states)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
