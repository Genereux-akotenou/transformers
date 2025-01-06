"""PyTorch BERT model. """

import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.autograd as autograd

from .utils.activations import gelu, gelu_new, swish
from .configuration_bert import BertConfig
from .utils.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .utils.modeling_utils import PreTrainedModel, prune_linear_layer
logger = logging.getLogger(__name__)
from typing import List, Optional, Tuple, Union
from ..utils.compressive_memory import CompressiveMemory

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))
ACT2FN = {
    "gelu": gelu, 
    "relu": torch.nn.functional.relu, 
    "swish": swish, 
    "gelu_new": gelu_new, 
    "mish": mish
}

BertLayerNorm = torch.nn.LayerNorm

class BertEmbeddings(nn.Module):
    """
    1. Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
        
class BertSelfAttention(nn.Module):
    """
    2. Construct self attention class.
    """
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query   = nn.Linear(config.hidden_size, self.all_head_size)
        self.key     = nn.Linear(config.hidden_size, self.all_head_size)
        self.value   = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer   = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

class BertSdpaSelfAttention(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.dropout_prob = config.attention_probs_dropout_prob
        #self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
            # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once implemented.
            logger.warning_once(
                "BertSdpaSelfAttention is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
                "non-absolute `position_embedding_type` or `output_attentions=True` or `head_mask`. Falling back to "
                "the manual attention implementation, but specifying the manual implementation will be required from "
                "Transformers version v5.0.0 onwards. This warning can be removed using the argument "
                '`attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

        bsz, tgt_len, _ = hidden_states.size()

        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # If this is instantiated as a cross-attention module, the keys and values come from an encoder; the attention
        # mask needs to be such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        attention_mask = encoder_attention_mask if is_cross_attention else attention_mask

        # Check `seq_length` of `past_key_value` == `len(current_states)` to support prefix tuning
        if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
            key_layer, value_layer = past_key_value
        else:
            key_layer = self.transpose_for_scores(self.key(current_states))
            value_layer = self.transpose_for_scores(self.value(current_states))
            if past_key_value is not None and not is_cross_attention:
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()` here. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        if self.require_contiguous_qkv and query_layer.device.type == "cuda" and attention_mask is not None:
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create
        # a causal mask in case tgt_len == 1.
        is_causal = (
            True if self.is_decoder and not is_cross_attention and attention_mask is None and tgt_len > 1 else False
        )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)

        outputs = (attn_output,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class BertSdpaSelfInfiniAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.position_embedding_type = config.position_embedding_type
        self.dropout_prob = config.attention_probs_dropout_prob
        # for infinite attention
        self.segments = None
        self.d_key = self.d_value = config.hidden_size
        self.d_model = config.hidden_size
        self.segment_size = config.segment_size
        self.memory = CompressiveMemory(config.batch_size, self.d_key, self.d_value, config.segment_size)
        self.beta = nn.Parameter(torch.randn((1, 1, 1)))
        self.require_contiguous_qkv = False #version.parse(get_torch_version()) < version.parse("2.2.0")
        #self.o_proj  = nn.Linear(self.d_model, self.d_model)


    def split(self, tensor):
        batch_size, length, d_tensor = tensor.size()
        segments = torch.split(tensor, self.segment_size, dim=1)
        return segments
    
    def split_mask(self, mask):
        segments = torch.split(mask, self.segment_size, dim=-1)
        return segments
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        # if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
        #     # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once implemented.
        #     logger.warning_once(
        #         "BertSdpaSelfAttention is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
        #         "non-absolute `position_embedding_type` or `output_attentions=True` or `head_mask`. Falling back to "
        #         "the manual attention implementation, but specifying the manual implementation will be required from "
        #         "Transformers version v5.0.0 onwards. This warning can be removed using the argument "
        #         '`attn_implementation="eager"` when loading the model.'
        #     )
        #     return super().forward(
        #         hidden_states,
        #         attention_mask,
        #         head_mask,
        #         encoder_hidden_states,
        #         encoder_attention_mask,
        #         past_key_value,
        #         output_attentions,
        #     )

        bsz, tgt_len, _ = hidden_states.size()
        # If this is instantiated as a cross-attention module, the keys and values come from an encoder; the attention
        # mask needs to be such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        attention_mask = encoder_attention_mask if is_cross_attention else attention_mask

        # 1. Split the input into segments
        q_segments = self.split(hidden_states)
        k_segments = self.split(current_states)
        v_segments = self.split(current_states)
        mask_segments = self.split_mask(attention_mask) if attention_mask is not None else None
        #print(len(q_segments))

        output_segments = []
        for q_seg, k_seg, v_seg, mask_seg in zip(q_segments, k_segments, v_segments, mask_segments):
            # 2: dot-product attention within each segment.
            query_layer = self.transpose_for_scores(self.query(q_seg))

            # Check `seq_length` of `past_key_value` == `len(current_states)` to support prefix tuning
            if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
                key_layer, value_layer = past_key_value
            else:
                key_layer = self.transpose_for_scores(self.key(k_seg))
                value_layer = self.transpose_for_scores(self.value(v_seg))
                if past_key_value is not None and not is_cross_attention:
                    key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                    value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

            if self.is_decoder:
                # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
                # Further calls to cross_attention layer can then reuse all cross-attention
                # key/value_states (first "if" case)
                # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
                # all previous decoder key/value_states. Further calls to uni-directional self-attention
                # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
                # if encoder bi-directional self-attention `past_key_value` is always `None`
                past_key_value = (key_layer, value_layer)

            # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
            # attn_mask, so we need to call `.contiguous()` here. This was fixed in torch==2.2.0.
            # Reference: https://github.com/pytorch/pytorch/issues/112577
            if self.require_contiguous_qkv and query_layer.device.type == "cuda" and mask_seg is not None:
                query_layer = query_layer.contiguous()
                key_layer = key_layer.contiguous()
                value_layer = value_layer.contiguous()

            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create
            # a causal mask in case tgt_len == 1.
            is_causal = (
                True if self.is_decoder and not is_cross_attention and mask_seg is None and tgt_len > 1 else False
            )

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=mask_seg,
                dropout_p=self.dropout_prob if self.training else 0.0,
                is_causal=is_causal,
            )

            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
            A_dot = attn_output

            # 3: pull Amem from compressive memory using current segment’s query
            A_mem = self.memory.retrieve(q_seg.detach())

            # 4: combine local context with the long-term context
            beta =  nn.Sigmoid()(self.beta)
            A = beta * A_mem + (1 - beta) * A_dot

            # 5: Update compressive memory by adding KV
            self.memory.update(q_seg.detach(), v_seg.detach())

            # 6: discard the previous segment's attention states pass updated memory to next segment
            #output = self.o_proj(A)
            output = A
            output_segments.append(output)

        # concat along sequence dimension
        attn_output = torch.cat(output_segments, dim=1)

        # 7. Free compressive memory
        self.memory.free()

        # 8. visualize attention map
        # TODO: implement visualization

        outputs = (attn_output,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs    
            
    
class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        #self.self = BertSdpaSelfInfiniAttention(config)
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )
        # print(hidden_states.shape)
        # print(attention_mask.shape)
        # print(head_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


"""
def __init__(self, config):
        super().__init__(config)
        self.position_embedding_type = config.position_embedding_type
        self.dropout_prob = config.attention_probs_dropout_prob
        # for infinite attention
        self.segments = None
        self.d_key = self.d_value = config.hidden_size
        self.d_model = config.hidden_size
        self.segment_size = config.segment_size
        self.memory = CompressiveMemory(config.batch_size, self.d_key, self.d_value, config.segment_size)
        self.beta = nn.Parameter(torch.randn((1, 1, 1)))
        self.require_contiguous_qkv = False #version.parse(get_torch_version()) < version.parse("2.2.0")
        #self.o_proj  = nn.Linear(self.d_model, self.d_model)


    def split(self, tensor):
        batch_size, length, d_tensor = tensor.size()
        segments = torch.split(tensor, self.segment_size, dim=1)
        return segments
    
    def split_mask(self, mask):
        segments = torch.split(mask, self.segment_size, dim=-1)
        return segments
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        # if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
        #     # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once implemented.
        #     logger.warning_once(
        #         "BertSdpaSelfAttention is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
        #         "non-absolute `position_embedding_type` or `output_attentions=True` or `head_mask`. Falling back to "
        #         "the manual attention implementation, but specifying the manual implementation will be required from "
        #         "Transformers version v5.0.0 onwards. This warning can be removed using the argument "
        #         '`attn_implementation="eager"` when loading the model.'
        #     )
        #     return super().forward(
        #         hidden_states,
        #         attention_mask,
        #         head_mask,
        #         encoder_hidden_states,
        #         encoder_attention_mask,
        #         past_key_value,
        #         output_attentions,
        #     )

        bsz, tgt_len, _ = hidden_states.size()
        # If this is instantiated as a cross-attention module, the keys and values come from an encoder; the attention
        # mask needs to be such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        attention_mask = encoder_attention_mask if is_cross_attention else attention_mask

        # 1. Split the input into segments
        q_segments = self.split(hidden_states)
        k_segments = self.split(current_states)
        v_segments = self.split(current_states)
        mask_segments = self.split_mask(attention_mask) if attention_mask is not None else None
        #print(len(q_segments))

        output_segments = []
        for q_seg, k_seg, v_seg, mask_seg in zip(q_segments, k_segments, v_segments, mask_segments):
            # 2: dot-product attention within each segment.
            query_layer = self.transpose_for_scores(self.query(q_seg))

            # Check `seq_length` of `past_key_value` == `len(current_states)` to support prefix tuning
            if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
                key_layer, value_layer = past_key_value
            else:
                key_layer = self.transpose_for_scores(self.key(k_seg))
                value_layer = self.transpose_for_scores(self.value(v_seg))
                if past_key_value is not None and not is_cross_attention:
                    key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                    value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

            if self.is_decoder:
                # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
                # Further calls to cross_attention layer can then reuse all cross-attention
                # key/value_states (first "if" case)
                # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
                # all previous decoder key/value_states. Further calls to uni-directional self-attention
                # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
                # if encoder bi-directional self-attention `past_key_value` is always `None`
                past_key_value = (key_layer, value_layer)

            # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
            # attn_mask, so we need to call `.contiguous()` here. This was fixed in torch==2.2.0.
            # Reference: https://github.com/pytorch/pytorch/issues/112577
            if self.require_contiguous_qkv and query_layer.device.type == "cuda" and mask_seg is not None:
                query_layer = query_layer.contiguous()
                key_layer = key_layer.contiguous()
                value_layer = value_layer.contiguous()

            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create
            # a causal mask in case tgt_len == 1.
            is_causal = (
                True if self.is_decoder and not is_cross_attention and mask_seg is None and tgt_len > 1 else False
            )

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=mask_seg,
                dropout_p=self.dropout_prob if self.training else 0.0,
                is_causal=is_causal,
            )

            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
            A_dot = attn_output

            # 3: pull Amem from compressive memory using current segment’s query
            A_mem = self.memory.retrieve(q_seg.detach())

            # 4: combine local context with the long-term context
            beta =  nn.Sigmoid()(self.beta)
            A = beta * A_mem + (1 - beta) * A_dot

            # 5: Update compressive memory by adding KV
            self.memory.update(q_seg.detach(), v_seg.detach())

            # 6: discard the previous segment's attention states pass updated memory to next segment
            #output = self.o_proj(A)
            output = A
            output_segments.append(output)

        # concat along sequence dimension
        attn_output = torch.cat(output_segments, dim=1)

        # 7. Free compressive memory
        self.memory.free()

        # 8. visualize attention map
        # TODO: implement visualization

        outputs = (attn_output,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs    
            
"""
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# ----------------------------------------
# ----------------------------------------
class BertModel(PreTrainedModel):
    config_class = BertConfig
    pretrained_model_archive_map = None
    load_tf_weights = None
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

        # for infinite attention
        self.segments = None
        self.d_key = self.d_value = config.hidden_size
        self.d_model = config.hidden_size
        self.segment_size = config.segment_size
        self.memory = CompressiveMemory(config.batch_size, self.d_key, self.d_value, config.segment_size)
        self.beta = nn.Parameter(torch.randn((1, 1, 1)))

    def split(self, tensor):
        batch_size, length, d_tensor = tensor.size()
        segments = torch.split(tensor, self.segment_size, dim=1)
        return segments
    
    def split_mask(self, mask):
        segments = torch.split(mask, self.segment_size, dim=-1)
        return segments

    def split(self, tensor, segment_size):
        segments = torch.split(tensor, segment_size, dim=1)
        return segments

    def split_mask(self, mask, segment_size):
        segments = torch.split(mask, segment_size, dim=-1)
        return segments

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            
            for param in module.parameters():
                if len(param.shape) >= 2:
                    torch.nn.init.xavier_normal_(param.data)
                else:
                    torch.nn.init.normal_(param.data)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params,trainable_params

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extened_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        # 1. Split the input into segments
        inputs_ids_segments = self.split(input_ids, segment_size=self.segment_size)
        position_ids_segments = self.split(position_ids, segment_size=self.segment_size) if position_ids is not None else None
        token_type_ids_segments = self.split(token_type_ids, segment_size=self.segment_size) if token_type_ids is not None else None
        extended_attention_mask_segments = self.split_mask(extended_attention_mask,segment_size=self.segment_size) if extended_attention_mask is not None else None

        output_segments = []
        for inputs_ids_seg, extended_attention_mask_seg, token_type_ids_seg in zip(inputs_ids_segments, extended_attention_mask_segments, token_type_ids_segments):
            # 2: dot-product attention within each segment.
         
        
            embedding_output = self.embeddings(
                input_ids=inputs_ids_seg, position_ids=position_ids, token_type_ids=token_type_ids_seg, inputs_embeds=inputs_embeds
            )
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask_seg,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
            )
            A_dot = sequence_output = encoder_outputs[0]

            # 3: pull Amem from compressive memory using current segment’s query
            A_mem = self.memory.retrieve(embedding_output.detach())

            # 4: combine local context with the long-term context
            beta =  nn.Sigmoid()(self.beta)
            A = beta * A_mem + (1 - beta) * A_dot

            # 5: Update compressive memory by adding KV
            self.memory.update(embedding_output.detach(), embedding_output.detach())

            # 6: discard the previous segment's attention states pass updated memory to next segment
            output = A
            output_segments.append(output)

        # concat along sequence dimension
        attn_output = torch.cat(output_segments, dim=1)

        # 7. Free compressive memory
        self.memory.free()

        # 8. visualize attention map
        # TODO: implement visualization

        pooled_output = self.pooler(attn_output)
        outputs = (attn_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
    

        # embedding_output = self.embeddings(
        #     input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        # )
        # encoder_outputs = self.encoder(
        #     embedding_output,
        #     attention_mask=extended_attention_mask,
        #     head_mask=head_mask,
        #     encoder_hidden_states=encoder_hidden_states,
        #     encoder_attention_mask=encoder_extended_attention_mask,
        # )
        # sequence_output = encoder_outputs[0]
        # pooled_output = self.pooler(sequence_output)

        # outputs = (sequence_output, pooled_output,) + encoder_outputs[
        #     1:
        # ]  # add hidden_states and attentions if they are here
        # return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertForSequenceClassification(PreTrainedModel):
    config_class = BertConfig
    pretrained_model_archive_map = None
    load_tf_weights = None
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            
            for param in module.parameters():
                if len(param.shape) >= 2:
                    torch.nn.init.xavier_normal_(param.data)
                else:
                    torch.nn.init.normal_(param.data)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


# ----------------------------------------
# ----------------------------------------








# class BertPredictionHeadTransform(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         if isinstance(config.hidden_act, str):
#             self.transform_act_fn = ACT2FN[config.hidden_act]
#         else:
#             self.transform_act_fn = config.hidden_act
#         self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

#     def forward(self, hidden_states):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.transform_act_fn(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states)
#         return hidden_states


# class BertLMPredictionHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.transform = BertPredictionHeadTransform(config)

#         # The output weights are the same as the input embeddings, but there is
#         # an output-only bias for each token.
#         self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         self.bias = nn.Parameter(torch.zeros(config.vocab_size))

#         # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
#         self.decoder.bias = self.bias

#     def forward(self, hidden_states):
#         hidden_states = self.transform(hidden_states)
#         hidden_states = self.decoder(hidden_states) + self.bias
#         return hidden_states


# class BertOnlyMLMHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.predictions = BertLMPredictionHead(config)

#     def forward(self, sequence_output):
#         prediction_scores = self.predictions(sequence_output)
#         return prediction_scores


# class BertOnlyNSPHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.seq_relationship = nn.Linear(config.hidden_size, 2)

#     def forward(self, pooled_output):
#         seq_relationship_score = self.seq_relationship(pooled_output)
#         return seq_relationship_score


# class BertPreTrainingHeads(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.predictions = BertLMPredictionHead(config)
#         self.seq_relationship = nn.Linear(config.hidden_size, 2)

#     def forward(self, sequence_output, pooled_output):
#         prediction_scores = self.predictions(sequence_output)
#         seq_relationship_score = self.seq_relationship(pooled_output)
#         return prediction_scores, seq_relationship_score





# BERT_START_DOCSTRING = r"""
#     This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
#     Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
#     usage and behavior.

#     Parameters:
#         config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
#             Initializing with a config file does not load the weights associated with the model, only the configuration.
#             Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
# """

# BERT_INPUTS_DOCSTRING = r"""
#     Args:
#         input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
#             Indices of input sequence tokens in the vocabulary.

#             Indices can be obtained using :class:`transformers.BertTokenizer`.
#             See :func:`transformers.PreTrainedTokenizer.encode` and
#             :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

#             `What are input IDs? <../glossary.html#input-ids>`__
#         attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
#             Mask to avoid performing attention on padding token indices.
#             Mask values selected in ``[0, 1]``:
#             ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

#             `What are attention masks? <../glossary.html#attention-mask>`__
#         token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
#             Segment token indices to indicate first and second portions of the inputs.
#             Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
#             corresponds to a `sentence B` token

#             `What are token type IDs? <../glossary.html#token-type-ids>`_
#         position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
#             Indices of positions of each input sequence tokens in the position embeddings.
#             Selected in the range ``[0, config.max_position_embeddings - 1]``.

#             `What are position IDs? <../glossary.html#position-ids>`_
#         head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
#             Mask to nullify selected heads of the self-attention modules.
#             Mask values selected in ``[0, 1]``:
#             :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
#         inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
#             Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
#             This is useful if you want more control over how to convert `input_ids` indices into associated vectors
#             than the model's internal embedding lookup matrix.
#         encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
#             Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
#             if the model is configured as a decoder.
#         encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
#             Mask to avoid performing attention on the padding token indices of the encoder input. This mask
#             is used in the cross-attention if the model is configured as a decoder.
#             Mask values selected in ``[0, 1]``:
#             ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
# """



# @add_start_docstrings(
#     """Bert Model with two heads on top as done during the pre-training: a `masked language modeling` head and
#     a `next sentence prediction (classification)` head. """,
#     BERT_START_DOCSTRING,
# )
# class BertForPreTraining(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)

#         self.bert = BertModel(config)
#         self.cls = BertPreTrainingHeads(config)

#         self.init_weights()

#     def get_output_embeddings(self):
#         return self.cls.predictions.decoder

#     @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         masked_lm_labels=None,
#         next_sentence_label=None,
#     ):
#         r"""
#         masked_lm_labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`, defaults to :obj:`None`):
#             Labels for computing the masked language modeling loss.
#             Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
#             Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
#             in ``[0, ..., config.vocab_size]``
#         next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`, defaults to :obj:`None`):
#             Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see :obj:`input_ids` docstring)
#             Indices should be in ``[0, 1]``.
#             ``0`` indicates sequence B is a continuation of sequence A,
#             ``1`` indicates sequence B is a random sequence.

#     Returns:
#         :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
#         loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
#             Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
#         prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
#             Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
#         seq_relationship_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, 2)`):
#             Prediction scores of the next sequence prediction (classification) head (scores of True/False
#             continuation before SoftMax).
#         hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
#             Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
#             of shape :obj:`(batch_size, sequence_length, hidden_size)`.

#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#             :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.


#     Examples::

#         from transformers import BertTokenizer, BertForPreTraining
#         import torch

#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         model = BertForPreTraining.from_pretrained('bert-base-uncased')

#         input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
#         outputs = model(input_ids)

#         prediction_scores, seq_relationship_scores = outputs[:2]

#         """

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         sequence_output, pooled_output = outputs[:2]
#         prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

#         outputs = (prediction_scores, seq_relationship_score,) + outputs[
#             2:
#         ]  # add hidden states and attention if they are here

#         if masked_lm_labels is not None and next_sentence_label is not None:
#             loss_fct = CrossEntropyLoss()
#             masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
#             next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
#             total_loss = masked_lm_loss + next_sentence_loss
#             outputs = (total_loss,) + outputs

#         return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


# @add_start_docstrings("""Bert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING)
# class BertForMaskedLM(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)

#         self.bert = BertModel(config)
#         self.cls = BertOnlyMLMHead(config)

#         self.init_weights()

#     def get_output_embeddings(self):
#         return self.cls.predictions.decoder

#     @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         masked_lm_labels=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         lm_labels=None,
#     ):
#         r"""
#         masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
#             Labels for computing the masked language modeling loss.
#             Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
#             Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
#             in ``[0, ..., config.vocab_size]``
#         lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
#             Labels for computing the left-to-right language modeling loss (next word prediction).
#             Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
#             Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
#             in ``[0, ..., config.vocab_size]``

#     Returns:
#         :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
#         masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
#             Masked language modeling loss.
#         ltr_lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_labels` is provided):
#                 Next token prediction loss.
#         prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
#             Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
#         hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
#             of shape :obj:`(batch_size, sequence_length, hidden_size)`.

#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#             :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.

#         Examples::

#             from transformers import BertTokenizer, BertForMaskedLM
#             import torch

#             tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#             model = BertForMaskedLM.from_pretrained('bert-base-uncased')

#             input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
#             outputs = model(input_ids, masked_lm_labels=input_ids)

#             loss, prediction_scores = outputs[:2]

#         """

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#         )

#         sequence_output = outputs[0]
#         prediction_scores = self.cls(sequence_output)

#         outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

#         # Although this may seem awkward, BertForMaskedLM supports two scenarios:
#         # 1. If a tensor that contains the indices of masked labels is provided,
#         #    the cross-entropy is the MLM cross-entropy that measures the likelihood
#         #    of predictions for masked words.
#         # 2. If `lm_labels` is provided we are in a causal scenario where we
#         #    try to predict the next token for each input in the decoder.
#         if masked_lm_labels is not None:
#             loss_fct = CrossEntropyLoss()  # -100 index = padding token
#             masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
#             outputs = (masked_lm_loss,) + outputs

#         if lm_labels is not None:
#             # we are doing next-token prediction; shift prediction scores and input ids by one
#             prediction_scores = prediction_scores[:, :-1, :].contiguous()
#             lm_labels = lm_labels[:, 1:].contiguous()
#             loss_fct = CrossEntropyLoss()
#             ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
#             outputs = (ltr_lm_loss,) + outputs

#         return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)


# @add_start_docstrings(
#     """Bert Model with a `next sentence prediction (classification)` head on top. """, BERT_START_DOCSTRING,
# )
# class BertForNextSentencePrediction(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)

#         self.bert = BertModel(config)
#         self.cls = BertOnlyNSPHead(config)

#         self.init_weights()

#     @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         next_sentence_label=None,
#     ):
#         r"""
#         next_sentence_label (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#             Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
#             Indices should be in ``[0, 1]``.
#             ``0`` indicates sequence B is a continuation of sequence A,
#             ``1`` indicates sequence B is a random sequence.

#     Returns:
#         :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
#         loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`next_sentence_label` is provided):
#             Next sequence prediction (classification) loss.
#         seq_relationship_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, 2)`):
#             Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
#         hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
#             of shape :obj:`(batch_size, sequence_length, hidden_size)`.

#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#             :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.

#     Examples::

#         from transformers import BertTokenizer, BertForNextSentencePrediction
#         import torch

#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

#         input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
#         outputs = model(input_ids)

#         seq_relationship_scores = outputs[0]

#         """

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         pooled_output = outputs[1]

#         seq_relationship_score = self.cls(pooled_output)

#         outputs = (seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
#         if next_sentence_label is not None:
#             loss_fct = CrossEntropyLoss()
#             next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
#             outputs = (next_sentence_loss,) + outputs

#         return outputs  # (next_sentence_loss), seq_relationship_score, (hidden_states), (attentions)


# @add_start_docstrings(
#     """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
#     the pooled output) e.g. for GLUE tasks. """,
#     BERT_START_DOCSTRING,
# )
# class BertForSequenceClassification(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

#         self.init_weights()

#     @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#             Labels for computing the sequence classification/regression loss.
#             Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
#             If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
#             If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

#     Returns:
#         :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
#         loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
#             Classification (or regression if config.num_labels==1) loss.
#         logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
#             Classification (or regression if config.num_labels==1) scores (before SoftMax).
#         hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
#             of shape :obj:`(batch_size, sequence_length, hidden_size)`.

#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#             :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.

#     Examples::

#         from transformers import BertTokenizer, BertForSequenceClassification
#         import torch

#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

#         input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
#         labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
#         outputs = model(input_ids, labels=labels)

#         loss, logits = outputs[:2]

#         """
    
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )
        
#         pooled_output = outputs[1]

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

#         if labels is not None:
#             if self.num_labels == 1:
#                 #  We are doing regression
#                 loss_fct = MSELoss()
#                 loss = loss_fct(logits.view(-1), labels.view(-1))
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs = (loss,) + outputs

#         return outputs  # (loss), logits, (hidden_states), (attentions)



#<><>

# @add_start_docstrings(
#     """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
#     the pooled output) e.g. for GLUE tasks. Especially designed for sequences longer than 512. """,
#     BERT_START_DOCSTRING,
# )
# class BertForLongSequenceClassification(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.split = config.split
#         self.rnn_type = config.rnn
#         self.num_rnn_layer = config.num_rnn_layer
#         self.hidden_size = config.hidden_size
#         self.rnn_dropout = config.rnn_dropout
#         self.rnn_hidden = config.rnn_hidden

#         self.bert = BertModel(config)
#         if self.rnn_type == "lstm":
#             self.rnn = nn.LSTM(input_size=self.hidden_size,hidden_size=self.hidden_size, bidirectional=True, num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
#         elif self.rnn_type == "gru":
#             self.rnn = nn.GRU(input_size=self.hidden_size,hidden_size=self.hidden_size, bidirectional=True, num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
#         else:
#             raise ValueError
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(self.hidden_size, self.config.num_labels)

#         self.init_weights()
    

#     @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         overlap=100,
#         max_length_per_seq=500,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#             Labels for computing the sequence classification/regression loss.
#             Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
#             If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
#             If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

#     Returns:
#         :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
#         loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
#             Classification (or regression if config.num_labels==1) loss.
#         logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
#             Classification (or regression if config.num_labels==1) scores (before SoftMax).
#         hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
#             of shape :obj:`(batch_size, sequence_length, hidden_size)`.

#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#             :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.

#     Examples::

#         from transformers import BertTokenizer, BertForSequenceClassification
#         import torch

#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

#         input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
#         labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
#         outputs = model(input_ids, labels=labels)

#         loss, logits = outputs[:2]

#         """
#         # batch_size = input_ids.shape[0]
#         # sequence_length = input_ids.shape[1]
#         # starts = []
#         # start = 0
#         # while start + max_length_per_seq <= sequence_length:
#         #     starts.append(start)
#         #     start += (max_length_per_seq-overlap)
#         # last_start = sequence_length-max_length_per_seq
#         # if last_start > starts[-1]:
#         #     starts.append(last_start)
        

#         # new_input_ids = torch.zeros([len(starts)*batch_size, max_length_per_seq], dtype=input_ids.dtype, device=input_ids.device)
#         # new_attention_mask = torch.zeros([len(starts)*batch_size, max_length_per_seq], dtype=attention_mask.dtype, device=attention_mask.device)
#         # new_token_type_ids = torch.zeros([len(starts)*batch_size, max_length_per_seq], dtype=token_type_ids.dtype, device=token_type_ids.device)

#         # for j in range(batch_size):
#         #     for i, start in enumerate(starts):
#         #         new_input_ids[i] = input_ids[j,start:start+max_length_per_seq]
#         #         new_attention_mask[i] = attention_mask[j,start:start+max_length_per_seq]
#         #         new_token_type_ids[i] = token_type_ids[j,start:start+max_length_per_seq]

#         # if batch_size == 1:
#         #     pooled_output = outputs[1].mean(dim=0)
#         #     pooled_output = pooled_output.reshape(1, pooled_output.shape[0])
#         # else:
#         #     pooled_output = torch.zeros([batch_size, outputs[1].shape[1]], dtype=outputs[1].dtype)
#         #     for i in range(batch_size):
#         #         pooled_output[i] = outputs[1][i*batch_size:(i+1)*batch_size].mean(dim=0)
        
#         batch_size = input_ids.shape[0]
#         input_ids = input_ids.view(self.split*batch_size, 512)
#         attention_mask = attention_mask.view(self.split*batch_size, 512)
#         token_type_ids = None
        

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         # lstm
#         if self.rnn_type == "lstm":
#             # random
#             # h0 = autograd.Variable(torch.randn([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device))/100.0
#             # c0 = autograd.Variable(torch.randn([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device))/100.0
#             # self.hidden = (h0, c0)
#             # self.rnn.flatten_parameters()
#             # pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
#             # _, (ht, ct) = self.rnn(pooled_output, self.hidden)

#             # orth
#             # h0 = torch.empty([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device)
#             # nn.init.orthogonal_(h0)
#             # h0 = autograd.Variable(h0)
#             # c0 = torch.empty([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device)
#             # nn.init.orthogonal_(c0)
#             # c0 = autograd.Variable(c0)
#             # self.hidden = (h0, c0)
#             # pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
#             # _, (ht, ct) = self.rnn(pooled_output, self.hidden)

#             # zero
#             pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
#             _, (ht, ct) = self.rnn(pooled_output)
#         elif self.rnn_type == "gru":
#             # h0 = autograd.Variable(torch.randn([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device))
#             # pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
#             # _, ht = self.rnn(pooled_output, h0)

#             # h0 = torch.empty([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device)
#             # nn.init.orthogonal_(h0)
#             # h0 = autograd.Variable(h0)
#             # pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
#             # _, ht = self.rnn(pooled_output, h0)

#             pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
#             _, ht = self.rnn(pooled_output)
#         else:
#             raise ValueError


    
#         output = self.dropout(ht.squeeze(0).sum(dim=0))
#         logits = self.classifier(output)
#         outputs = (logits,) + outputs[2:]   # add hidden states and attention if they are here

        
#         if labels is not None:
#             if self.num_labels == 1:
#                 loss_fct = MSELoss()
#                 loss = loss_fct(logits.view(-1), labels.view(-1))
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs = (loss,) + outputs

#         return outputs  # (loss), logits, (hidden_states), (attentions)



# @add_start_docstrings(
#     """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
#     the pooled output) e.g. for GLUE tasks. Especially designed for sequences longer than 512. Use non-overlapped concatenation """,
#     BERT_START_DOCSTRING,
# )
# class BertForLongSequenceClassificationCat(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.split = config.split

#         self.bert = BertModel(config)
#         self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
#         self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        
#         self.classifier1 = nn.Linear(config.hidden_size*self.split, config.hidden_size)
#         self.relu = torch.nn.ReLU()
#         self.classifier2 = nn.Linear(config.hidden_size, self.config.num_labels)

#         self.init_weights()

#     @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         overlap=100,
#         max_length_per_seq=500,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#             Labels for computing the sequence classification/regression loss.
#             Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
#             If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
#             If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

#     Returns:
#         :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
#         loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
#             Classification (or regression if config.num_labels==1) loss.
#         logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
#             Classification (or regression if config.num_labels==1) scores (before SoftMax).
#         hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
#             of shape :obj:`(batch_size, sequence_length, hidden_size)`.

#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#             :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.

#     Examples::

#         from transformers import BertTokenizer, BertForSequenceClassification
#         import torch

#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

#         input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
#         labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
#         outputs = model(input_ids, labels=labels)

#         loss, logits = outputs[:2]

#         """
        

#         batch_size = input_ids.shape[0]
#         input_ids = input_ids.view(self.split*batch_size, 512)
#         attention_mask = attention_mask.view(self.split*batch_size, 512)
        

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=None,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         # hidden_states = outputs[1].unsqueeze(0)
#         # attention_mask = torch.ones([1,1,1,10], device=hidden_states.device)


#         # attended_output = self.atten1(hidden_states=hidden_states, attention_mask=attention_mask, head_mask=None)
#         # attended_output = self.atten2(hidden_states=attended_output[0], attention_mask=attention_mask, head_mask=None)
#         # attended_output = self.atten3(hidden_states=attended_output[0], attention_mask=attention_mask, head_mask=None)
#         # attended_output = self.atten4(hidden_states=attended_output[0], attention_mask=attention_mask, head_mask=None)
#         # attended_output = self.atten5(hidden_states=attended_output[0], attention_mask=attention_mask, head_mask=None)
#         # attended_output = self.atten6(hidden_states=attended_output[0], attention_mask=attention_mask, head_mask=None)

#         # max pooling
#         # pooled_output, _ = torch.max(attended_output[0],dim=1)

#         # concatenate
#         # pooled_output = attended_output[0].view(batch_size,-1)

#         pooled_output = outputs[1].view(batch_size,-1)
#         pooled_output = self.dropout1(pooled_output)
#         pooled_output = self.classifier1(pooled_output)
#         pooled_output = self.relu(pooled_output)
#         pooled_output = self.dropout2(pooled_output)
#         logits = self.classifier2(pooled_output)
#         outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

#         if labels is not None:
#             if self.num_labels == 1:
#                 #  We are doing regression
#                 loss_fct = MSELoss()
#                 loss = loss_fct(logits.view(-1), labels.view(-1))
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs = (loss,) + outputs

#         return outputs  # (loss), logits, (hidden_states), (attentions)






# @add_start_docstrings(
#     """Bert Model with a multiple choice classification head on top (a linear layer on top of
#     the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
#     BERT_START_DOCSTRING,
# )
# class BertForMultipleChoice(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)

#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, 1)

#         self.init_weights()

#     @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#             Labels for computing the multiple choice classification loss.
#             Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
#             of the input tensors. (see `input_ids` above)

#     Returns:
#         :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
#         loss (:obj:`torch.FloatTensor`` of shape ``(1,)`, `optional`, returned when :obj:`labels` is provided):
#             Classification loss.
#         classification_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
#             `num_choices` is the second dimension of the input tensors. (see `input_ids` above).

#             Classification scores (before SoftMax).
#         hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
#             of shape :obj:`(batch_size, sequence_length, hidden_size)`.

#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#             :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.

#     Examples::

#         from transformers import BertTokenizer, BertForMultipleChoice
#         import torch

#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
#         choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]

#         input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
#         labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
#         outputs = model(input_ids, labels=labels)

#         loss, classification_scores = outputs[:2]

#         """
#         num_choices = input_ids.shape[1]

#         input_ids = input_ids.view(-1, input_ids.size(-1))
#         attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
#         token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
#         position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         pooled_output = outputs[1]

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#         reshaped_logits = logits.view(-1, num_choices)

#         outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(reshaped_logits, labels)
#             outputs = (loss,) + outputs

#         return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


# @add_start_docstrings(
#     """Bert Model with a token classification head on top (a linear layer on top of
#     the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
#     BERT_START_DOCSTRING,
# )
# class BertForTokenClassification(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)

#         self.init_weights()

#     @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
#             Labels for computing the token classification loss.
#             Indices should be in ``[0, ..., config.num_labels - 1]``.

#     Returns:
#         :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
#         loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
#             Classification loss.
#         scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
#             Classification scores (before SoftMax).
#         hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
#             of shape :obj:`(batch_size, sequence_length, hidden_size)`.

#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#             :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.

#     Examples::

#         from transformers import BertTokenizer, BertForTokenClassification
#         import torch

#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         model = BertForTokenClassification.from_pretrained('bert-base-uncased')

#         input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
#         labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
#         outputs = model(input_ids, labels=labels)

#         loss, scores = outputs[:2]

#         """
#         batch_size = input_ids.shape[0]

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         sequence_output = outputs[0]
#         sequence_output = sequence_output[:,200:300,:]

#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)

#         outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
#         if labels is not None:
#             labels = labels[:,200:300]
#             # print(labels.shape)
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.reshape([labels.shape[0]*labels.shape[1]]))
#             # Only keep active parts of the loss
#             # if attention_mask is not None:
#             #     active_loss = attention_mask.view(-1) == 1
#             #     active_logits = logits.view(-1, self.num_labels)[active_loss]
#             #     active_labels = labels.view(-1)[active_loss]
#             #     loss = loss_fct(active_logits, active_labels)
#             # else:
#             #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs = (loss,) + outputs

#         return outputs  # (loss), scores, (hidden_states), (attentions)


# @add_start_docstrings(
#     """Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
#     layers on top of the hidden-states output to compute `span start logits` and `span end logits`). """,
#     BERT_START_DOCSTRING,
# )
# class BertForQuestionAnswering(BertPreTrainedModel):
#     def __init__(self, config):
#         super(BertForQuestionAnswering, self).__init__(config)
#         self.num_labels = config.num_labels

#         self.bert = BertModel(config)
#         self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

#         self.init_weights()

#     @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         start_positions=None,
#         end_positions=None,
#     ):
#         r"""
#         start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#             Labels for position (index) of the start of the labelled span for computing the token classification loss.
#             Positions are clamped to the length of the sequence (`sequence_length`).
#             Position outside of the sequence are not taken into account for computing the loss.
#         end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#             Labels for position (index) of the end of the labelled span for computing the token classification loss.
#             Positions are clamped to the length of the sequence (`sequence_length`).
#             Position outside of the sequence are not taken into account for computing the loss.

#     Returns:
#         :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
#         loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
#             Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
#         start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
#             Span-start scores (before SoftMax).
#         end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
#             Span-end scores (before SoftMax).
#         hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
#             of shape :obj:`(batch_size, sequence_length, hidden_size)`.

#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#             :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.

#     Examples::

#         from transformers import BertTokenizer, BertForQuestionAnswering
#         import torch

#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

#         question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
#         input_ids = tokenizer.encode(question, text)
#         token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
#         start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

#         all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
#         answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

#         assert answer == "a nice puppet"

#         """

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         sequence_output = outputs[0]

#         logits = self.qa_outputs(sequence_output)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)

#         outputs = (start_logits, end_logits,) + outputs[2:]
#         if start_positions is not None and end_positions is not None:
#             # If we are on multi-GPU, split add a dimension
#             if len(start_positions.size()) > 1:
#                 start_positions = start_positions.squeeze(-1)
#             if len(end_positions.size()) > 1:
#                 end_positions = end_positions.squeeze(-1)
#             # sometimes the start/end positions are outside our model inputs, we ignore these terms
#             ignored_index = start_logits.size(1)
#             start_positions.clamp_(0, ignored_index)
#             end_positions.clamp_(0, ignored_index)

#             loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = (start_loss + end_loss) / 2
#             outputs = (total_loss,) + outputs

#         return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)