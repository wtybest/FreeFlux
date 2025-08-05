import torch
from typing import Callable, List, Optional, Tuple, Union, List
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

class FluxAttnProcessor2_0_Bg_Replace:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, start_step=4, start_layer=0, layer_idx=None, step_idx=None, total_layers=57, total_steps=50):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0_Bg_Replace requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.total_steps = total_steps
        self.total_layers = total_layers
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("Background Replacement at denoising steps: ", self.step_idx)
        print("Background Replacement at U-Net layers: ", self.layer_idx)

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
        src_tar_inpaint_idx_list: List = None,
    ) -> torch.FloatTensor:

        out = self.attn_forward(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
            src_tar_inpaint_idx_list,
        )
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_attn_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.cur_step %= self.total_steps
            # after step
            self.after_step()
        return out


    def masactrl_forward(
        self,
        query,
        key,
        value,
        src_tar_inpaint_idx_list,
    ):
        """
        Rearrange the key and value for mutual self-attention control
        """
        kc_src, kc_tgt = key.chunk(2)
        vc_src, vc_tgt = value.chunk(2)

        src_idx = [x + 512 for x in src_tar_inpaint_idx_list[0]]
        tar_idx = [x + 512 for x in src_tar_inpaint_idx_list[1]]
        remain_idx = [x + 512 for x in src_tar_inpaint_idx_list[3]]

        vc_tgt_modified = vc_tgt.clone()
        vc_tgt_modified[:,:,tar_idx,:] = vc_src[:,:,src_idx,:]

        k_rearranged = torch.cat([kc_src, kc_tgt])
        v_rearranged = torch.cat([vc_src, vc_tgt_modified])

        return query, k_rearranged, v_rearranged


    def attn_forward(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        src_tar_inpaint_idx_list: List = None,
    ):

        cur_transformer_layer = self.cur_att_layer

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1] #3072
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
            key = apply_rotary_emb(key, image_rotary_emb)

        if self.cur_step in self.step_idx and cur_transformer_layer in self.layer_idx:
            query, key, value = self.masactrl_forward(query, key, value, src_tar_inpaint_idx_list)

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


def register_bg_replace_attention_control(model, **attn_args):
    attn_procs = FluxAttnProcessor2_0_Bg_Replace(**attn_args)
    model.transformer.set_attn_processor(attn_procs)
    print(f"Model {model.transformer.__class__.__name__} is registered attention processor: FluxAttnProcessor2_0_Bg_Replace")


class FluxAttnProcessor2_0_Reasoning:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, start_step=4, start_layer=0, layer_idx=None, step_idx=None, total_layers=57, total_steps=50):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0_Reasoning requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.total_steps = total_steps
        self.total_layers = total_layers
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("Reasoning at denoising steps: ", self.step_idx)
        print("Reasoning at U-Net layers: ", self.layer_idx)

        self.cur_step = 0
        self.cur_att_layer = 0
        self.num_attn_layers = total_layers
        self.global_store = {f"block_{i}": [[],[]] for i in range(57)}

    def after_step(self):
        pass

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        out = self.attn_forward(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
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
    ):

        cur_transformer_layer = self.cur_att_layer

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1] #3072
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
            key = apply_rotary_emb(key, image_rotary_emb)

        #use 1.0 approach to get the attention score of current layer
        q_tgt = query.clone()
        k_tgt = key.clone()

        q_tgt = q_tgt.transpose(1, 2).contiguous().view(1, -1, inner_dim)
        k_tgt = k_tgt.transpose(1, 2).contiguous().view(1, -1, inner_dim)
        q_tgt = attn.head_to_batch_dim(q_tgt)
        k_tgt = attn.head_to_batch_dim(k_tgt)
        attention_probs = torch.mean(attn.get_attention_scores(q_tgt, k_tgt), dim=0)
        cross_attn_1 = attention_probs[0:512, 512::].detach().cpu()
        cross_attn_2 = attention_probs[512::, 0:512].detach().cpu()
        self.global_store[f'block_{cur_transformer_layer}'][0].append(cross_attn_1)
        self.global_store[f'block_{cur_transformer_layer}'][1].append(cross_attn_2)


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



def register_reasoning_attention_control(model, **attn_args):
    attn_procs = FluxAttnProcessor2_0_Reasoning(**attn_args)
    model.transformer.set_attn_processor(attn_procs)
    print(f"Model {model.transformer.__class__.__name__} is registered attention processor: FluxAttnProcessor2_0_Reasoning")