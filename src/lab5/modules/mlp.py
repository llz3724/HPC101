"""
Feed Forward Network (FFN) 实现
"""

import torch
import torch.nn as nn
from typing import Optional
from .config import Qwen3Config


class Qwen3MLP(nn.Module):
    """Qwen3 前馈网络模块"""

    def __init__(
        self, config: Qwen3Config, device=None, dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        if dtype is None:
            dtype = torch.float32

        # 门控投影和上投影
        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        # 下投影
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 根据实验文档 2.9 节完成 FFN 模块的实现

        gate_proj_res = self.gate_proj(x)
        up_proj_res = self.up_proj(x)
        gate_output = torch.nn.functional.silu(gate_proj_res) * up_proj_res
        down_proj_res = self.down_proj(gate_output)
        return down_proj_res
