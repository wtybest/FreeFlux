import torch
from typing import Callable, List, Optional, Tuple, Union
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

class FluxAttnProcessor2_0_Probing:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, start_step=4, start_layer=0, layer_idx=None, step_idx=None, total_layers=57, total_steps=50):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0_Probing requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.total_steps = total_steps
        self.total_layers = total_layers
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("RoPE Change at denoising steps: ", self.step_idx)
        print("RoPE Change at U-Net layers: ", self.layer_idx)

        self.cur_step = 0
        self.cur_att_layer = 0
        self.num_attn_layers = total_layers

    def after_step(self):
        pass

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        given_v_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        out = self.attn_forward(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
            given_v_rotary_emb,
        )
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_attn_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.cur_step %= self.total_steps
            # after step
            self.after_step()
        return out


    def attn_forward(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        given_v_rotary_emb: Optional[torch.Tensor] = None,
    ):

        cur_transformer_layer = self.cur_att_layer

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads # head_dim=128, attn.heads=24

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            if self.cur_step in self.step_idx and cur_transformer_layer in self.layer_idx:
                if given_v_rotary_emb is not None:
                    key = apply_rotary_emb(key, given_v_rotary_emb)
            else:
                key = apply_rotary_emb(key, image_rotary_emb)


        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

def register_probing_control(model, **attn_args):
    attn_procs = FluxAttnProcessor2_0_Probing(**attn_args)
    model.transformer.set_attn_processor(attn_procs)
    print(f"Model {model.transformer.__class__.__name__} is registered attention processor: FluxAttnProcessor2_0_Probing")
