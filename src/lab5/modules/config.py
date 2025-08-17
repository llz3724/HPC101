"""
Qwen3 模型配置类
"""

import json
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Qwen3Config:
    """Qwen3 模型配置类"""

    # 模型架构参数
    vocab_size: int = 152064
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    max_position_embeddings: int = 32768

    # 激活函数和归一化
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    # 注意力机制参数
    attention_dropout: float = 0.0

    # 其他参数
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    use_cache: bool = True

    # 8B 模型特定参数
    def __post_init__(self):
        """初始化后处理，确保参数一致性"""
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Qwen3Config":
        """从字典创建配置"""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> "Qwen3Config":
        """从JSON文件加载配置"""
        with open(json_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_act": self.hidden_act,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "attention_dropout": self.attention_dropout,
            "initializer_range": self.initializer_range,
            "tie_word_embeddings": self.tie_word_embeddings,
            "use_cache": self.use_cache,
        }
