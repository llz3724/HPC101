"""
Qwen3 Transformer 层实现
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import Qwen3Config
from .attention import Qwen3Attention, RMSNorm
from .mlp import Qwen3MLP


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 Decoder 层"""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        device = "cpu" if device is None else device
        dtype = torch.float32 if dtype is None else dtype

        # 注意力模块
        self.self_attn = Qwen3Attention(
            config=config, layer_idx=layer_idx, device=device, dtype=dtype
        )

        # 前馈网络
        self.mlp = Qwen3MLP(config, device=device, dtype=dtype)

        # 层归一化
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码
            position_ids: 位置编码索引
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states
