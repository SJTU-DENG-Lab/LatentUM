import torch
import torch.nn as nn
from typing import Optional, Callable
from einops import rearrange


def create_mot_attention_forward(original_forward):
    """
    创建新的 attention forward 方法
    根据 vision_token_mask 选择使用不同的 qkvo projection
    - 视觉 token 使用 q/k/v/o_proj_vision
    - 文本 token 使用原本的 q/k/v/o_proj
    """
    def mot_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor],
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward, ALL_ATTENTION_FUNCTIONS
        
        # 从 kwargs 获取 vision_token_mask
        vision_token_mask = kwargs.pop('vision_token_mask', None)
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        if hasattr(self, 'q_proj_vision') and vision_token_mask is not None:
            # 获取 mask: (B, N) -> (B, N, 1)
            mask = vision_token_mask.unsqueeze(-1).to(hidden_states.dtype)
            
            # 分别计算文本和视觉的 Q, K, V
            q_text = self.q_proj(hidden_states)
            k_text = self.k_proj(hidden_states)
            v_text = self.v_proj(hidden_states)
            
            q_vision = self.q_proj_vision(hidden_states)
            k_vision = self.k_proj_vision(hidden_states)
            v_vision = self.v_proj_vision(hidden_states)
            
            # 根据 mask 混合：vision token 用 vision proj，text token 用 text proj
            q_states = mask * q_vision + (1 - mask) * q_text
            k_states = mask * k_vision + (1 - mask) * k_text
            v_states = mask * v_vision + (1 - mask) * v_text
            
            # 应用 q_norm 和 k_norm (也需要分开处理)
            q_states = q_states.view(hidden_shape)
            k_states = k_states.view(hidden_shape)
            
            q_norm_text = self.q_norm(q_states)
            k_norm_text = self.k_norm(k_states)
            q_norm_vision = self.q_norm_vision(q_states)
            k_norm_vision = self.k_norm_vision(k_states)
            
            # mask 需要扩展到 (B, N, num_heads, head_dim)
            mask_expanded = mask.unsqueeze(-1)  # (B, N, 1, 1)
            query_states = (mask_expanded * q_norm_vision + (1 - mask_expanded) * q_norm_text).transpose(1, 2)
            key_states = (mask_expanded * k_norm_vision + (1 - mask_expanded) * k_norm_text).transpose(1, 2)
            value_states = v_states.view(hidden_shape).transpose(1, 2)
        else:
            # 原始逻辑
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        
        # Output projection with MoT
        if hasattr(self, 'o_proj_vision') and vision_token_mask is not None:
            mask = vision_token_mask.unsqueeze(-1).to(attn_output.dtype)
            o_text = self.o_proj(attn_output)
            o_vision = self.o_proj_vision(attn_output)
            attn_output = mask * o_vision + (1 - mask) * o_text
        else:
            attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights
    
    return mot_attention_forward


def create_mot_decoder_forward(original_forward):
    """
    创建新的 decoder layer forward 方法
    根据 vision_token_mask 选择使用不同的 mlp
    """
    def mot_decoder_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple] = None,
        **kwargs,
    ) -> torch.Tensor:
        # 从 kwargs 获取 vision_token_mask（保留它以便传给 attention）
        vision_token_mask = kwargs.get('vision_token_mask', None)
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention (vision_token_mask 会通过 kwargs 传递)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected with MoT
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if hasattr(self, 'mlp_vision') and vision_token_mask is not None:
            mask = vision_token_mask.unsqueeze(-1).to(hidden_states.dtype)
            mlp_output = self.mlp(hidden_states)
            mlp_vision_output = self.mlp_vision(hidden_states)
            hidden_states = mask * mlp_vision_output + (1 - mask) * mlp_output
        else:
            hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        return hidden_states
    
    return mot_decoder_forward


def make_internvl_mot(internvl):
    """
    将 pretrained InternVL 转换为 MoT (Mixture of Transformers) 模型：
    1. 给每个 Qwen3Attention 添加 q/k/v/o_proj_vision 和 q/k_norm_vision
    2. 给每个 Qwen3DecoderLayer 添加 mlp_vision
    3. 冻结所有原有参数，只有新添加的参数可训练
    4. 修改 forward 逻辑：视觉 token 使用 vision 版本，文本 token 使用原版
    
    使用方式：在调用 language_model 时传入 vision_token_mask 参数
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm
    import types
    
    # 首先冻结所有原有参数
    for param in internvl.parameters():
        param.requires_grad = False
    
    # 获取 LLM 的配置
    llm_config = internvl.language_model.config
    
    # 获取所有的 decoder layers
    layers = internvl.language_model.model.layers
    
    # 为每个 layer 添加 vision 版本的模块并替换 forward
    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        
        # 添加 q/k/v/o_proj_vision
        attn.q_proj_vision = nn.Linear(
            llm_config.hidden_size,
            llm_config.num_attention_heads * attn.head_dim,
            bias=llm_config.attention_bias
        )
        attn.k_proj_vision = nn.Linear(
            llm_config.hidden_size,
            llm_config.num_key_value_heads * attn.head_dim,
            bias=llm_config.attention_bias
        )
        attn.v_proj_vision = nn.Linear(
            llm_config.hidden_size,
            llm_config.num_key_value_heads * attn.head_dim,
            bias=llm_config.attention_bias
        )
        attn.o_proj_vision = nn.Linear(
            llm_config.num_attention_heads * attn.head_dim,
            llm_config.hidden_size,
            bias=llm_config.attention_bias
        )
        
        # 添加 q/k_norm_vision
        attn.q_norm_vision = Qwen3RMSNorm(attn.head_dim, eps=llm_config.rms_norm_eps)
        attn.k_norm_vision = Qwen3RMSNorm(attn.head_dim, eps=llm_config.rms_norm_eps)
        
        # 用原有权重初始化
        attn.q_proj_vision.load_state_dict(attn.q_proj.state_dict())
        attn.k_proj_vision.load_state_dict(attn.k_proj.state_dict())
        attn.v_proj_vision.load_state_dict(attn.v_proj.state_dict())
        attn.o_proj_vision.load_state_dict(attn.o_proj.state_dict())
        attn.q_norm_vision.load_state_dict(attn.q_norm.state_dict())
        attn.k_norm_vision.load_state_dict(attn.k_norm.state_dict())
        
        # 设置为可训练
        for name in ['q_proj_vision', 'k_proj_vision', 'v_proj_vision', 'o_proj_vision', 'q_norm_vision', 'k_norm_vision']:
            for param in getattr(attn, name).parameters():
                param.requires_grad = True
        
        # 替换 attention forward
        attn.forward = types.MethodType(create_mot_attention_forward(attn.forward), attn)
        
        # 添加 mlp_vision
        mlp_vision = Qwen3MLP(llm_config)
        mlp_vision.load_state_dict(layer.mlp.state_dict())
        for param in mlp_vision.parameters():
            param.requires_grad = True
        layer.mlp_vision = mlp_vision
        
        # 替换 decoder layer forward
        layer.forward = types.MethodType(create_mot_decoder_forward(layer.forward), layer)
    
    return internvl
