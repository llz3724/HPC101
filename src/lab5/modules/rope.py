"""
RoPE (Rotary Position Embedding) 实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将张量的后半部分旋转到前半部分，并取负号"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用旋转位置编码到查询和键张量

    Args:
        q: 查询张量 [batch_size, num_heads, seq_len, head_dim]
        k: 键张量 [batch_size, num_heads, seq_len, head_dim]
        cos: 余弦值 [1, 1, seq_len, head_dim]
        sin: 正弦值 [1, 1, seq_len, head_dim]

    Returns:
        应用RoPE后的查询和键张量
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """旋转位置编码模块"""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 计算频率的倒数
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 缓存 cos 和 sin 值
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        """设置余弦和正弦值的缓存"""
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device).type_as(
            self.inv_freq
        )

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量
            seq_len: 序列长度

        Returns:
            cos 和 sin 张量
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        # 如果序列长度超过缓存长度，重新计算
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
