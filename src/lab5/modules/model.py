"""
Qwen3 模型实现
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Dict, Any
import math
import os
import json
from safetensors import safe_open
from pathlib import Path

from .config import Qwen3Config
from .layer import Qwen3DecoderLayer
from .attention import RMSNorm


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    创建因果注意力掩码
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    扩展注意力掩码
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


class Qwen3Model(nn.Module):
    """
    Qwen3 基础模型
    """

    def __init__(
        self,
        config: Qwen3Config,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.vocab_size - 1  # 通常使用最后一个token作为padding
        self.vocab_size = config.vocab_size

        if device is None:
            device = "cpu"
        if dtype is None:
            dtype = torch.float32

        # Token 嵌入
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            device=device,
            dtype=dtype,
        )

        # Transformer 层
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_idx, device=device, dtype=dtype)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # 最终层归一化
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "Qwen3Model":
        """
        从预训练模型加载权重

        Args:
            model_path: 模型路径（包含config.json和权重文件）
            device: 设备
            torch_dtype: 数据类型
        """
        model_path_obj = Path(model_path)

        # 加载配置
        config_path = model_path_obj / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        # 创建Qwen3Config
        qwen3_config = cls._convert_hf_config_to_qwen3(config_dict)

        # 创建模型
        model = cls(qwen3_config, device=device, dtype=torch_dtype)

        # 加载权重
        model._load_hf_weights(model_path_obj, torch_dtype)

        return model

    @staticmethod
    def _convert_hf_config_to_qwen3(hf_config: Dict[str, Any]) -> Qwen3Config:
        """将HuggingFace配置转换为Qwen3Config"""
        # HF配置到Qwen3配置的映射
        config_mapping = {
            "vocab_size": "vocab_size",
            "hidden_size": "hidden_size",
            "intermediate_size": "intermediate_size",
            "num_hidden_layers": "num_hidden_layers",
            "num_attention_heads": "num_attention_heads",
            "num_key_value_heads": "num_key_value_heads",
            "max_position_embeddings": "max_position_embeddings",
            "hidden_act": "hidden_act",
            "rms_norm_eps": "rms_norm_eps",
            "rope_theta": "rope_theta",
            "attention_dropout": "attention_dropout",
            "initializer_range": "initializer_range",
            "tie_word_embeddings": "tie_word_embeddings",
            "use_cache": "use_cache",
        }

        qwen3_config_dict = {}
        for hf_key, qwen3_key in config_mapping.items():
            if hf_key in hf_config:
                qwen3_config_dict[qwen3_key] = hf_config[hf_key]

        return Qwen3Config(**qwen3_config_dict)

    def _load_hf_weights(
        self, model_path: Path, torch_dtype: Optional[torch.dtype] = None
    ):
        """加载HuggingFace格式的权重"""
        # 查找权重文件
        safetensor_files = list(model_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"在 {model_path} 中未找到 .safetensors 文件")

        # 创建权重映射
        weight_map = self._create_weight_mapping()

        # 加载所有权重
        all_weights = {}
        for safetensor_file in safetensor_files:
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if torch_dtype is not None:
                        tensor = tensor.to(torch_dtype)
                    all_weights[key] = tensor

        # 创建状态字典
        state_dict = {}
        for hf_key, model_key in weight_map.items():
            if hf_key in all_weights:
                state_dict[model_key] = all_weights[hf_key]
            else:
                print(f"警告: 权重 {hf_key} 未在模型文件中找到")

        # 加载权重到模型
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"缺失的权重: {missing_keys}")
        if unexpected_keys:
            print(f"意外的权重: {unexpected_keys}")

    def _create_weight_mapping(self) -> Dict[str, str]:
        """创建HuggingFace权重名称到模型权重名称的映射"""
        weight_map = {}

        # 嵌入层
        weight_map["model.embed_tokens.weight"] = "embed_tokens.weight"

        # Transformer层
        for i in range(self.config.num_hidden_layers):
            layer_prefix = f"model.layers.{i}"
            model_prefix = f"layers.{i}"

            # 注意力层
            weight_map[f"{layer_prefix}.self_attn.q_proj.weight"] = (
                f"{model_prefix}.self_attn.q_proj.weight"
            )
            weight_map[f"{layer_prefix}.self_attn.k_proj.weight"] = (
                f"{model_prefix}.self_attn.k_proj.weight"
            )
            weight_map[f"{layer_prefix}.self_attn.v_proj.weight"] = (
                f"{model_prefix}.self_attn.v_proj.weight"
            )
            weight_map[f"{layer_prefix}.self_attn.o_proj.weight"] = (
                f"{model_prefix}.self_attn.o_proj.weight"
            )

            # MLP层
            weight_map[f"{layer_prefix}.mlp.gate_proj.weight"] = (
                f"{model_prefix}.mlp.gate_proj.weight"
            )
            weight_map[f"{layer_prefix}.mlp.up_proj.weight"] = (
                f"{model_prefix}.mlp.up_proj.weight"
            )
            weight_map[f"{layer_prefix}.mlp.down_proj.weight"] = (
                f"{model_prefix}.mlp.down_proj.weight"
            )

            # 层归一化
            weight_map[f"{layer_prefix}.input_layernorm.weight"] = (
                f"{model_prefix}.input_layernorm.weight"
            )
            weight_map[f"{layer_prefix}.post_attention_layernorm.weight"] = (
                f"{model_prefix}.post_attention_layernorm.weight"
            )

        # 最终层归一化
        weight_map["model.norm.weight"] = "norm.weight"

        return weight_map

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def post_init(self):
        """初始化权重"""
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, torch.Tensor]:

        # 获取输入形状
        batch_size, seq_length = input_ids.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0

        # 嵌入
        inputs_embeds = self.embed_tokens(input_ids)

        # 注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds
        # 通过每个decoder层
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)

        return hidden_states

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        """准备decoder的注意力掩码"""
        # 创建因果掩码
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


class Qwen3ForCausalLM(nn.Module):
    """
    用于因果语言建模的Qwen3模型
    """

    def __init__(
        self,
        config: Qwen3Config,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.model = Qwen3Model(config, device=device, dtype=dtype)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "Qwen3ForCausalLM":
        """
        从预训练模型加载权重

        Args:
            model_path: 模型路径（包含config.json和权重文件）
            device: 设备
            torch_dtype: 数据类型
        """
        model_path_obj = Path(model_path)

        # 加载配置
        config_path = model_path_obj / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        # 创建Qwen3Config
        qwen3_config = Qwen3Model._convert_hf_config_to_qwen3(config_dict)

        # 创建模型
        model = cls(qwen3_config, device=device, dtype=torch_dtype)

        # 加载权重
        model._load_hf_weights(model_path_obj, torch_dtype)

        return model

    def _load_hf_weights(
        self, model_path: Path, torch_dtype: Optional[torch.dtype] = None
    ):
        """加载HuggingFace格式的权重（包含lm_head）"""
        # 查找权重文件
        safetensor_files = list(model_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"在 {model_path} 中未找到 .safetensors 文件")

        # 创建权重映射
        weight_map = self._create_weight_mapping()

        # 加载所有权重
        all_weights = {}
        for safetensor_file in safetensor_files:
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if torch_dtype is not None:
                        tensor = tensor.to(torch_dtype)
                    all_weights[key] = tensor

        # 创建状态字典
        state_dict = {}
        for hf_key, model_key in weight_map.items():
            if hf_key in all_weights:
                state_dict[model_key] = all_weights[hf_key]
            else:
                print(f"警告: 权重 {hf_key} 未在模型文件中找到")

        # 加载权重到模型
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"缺失的权重: {missing_keys}")
        if unexpected_keys:
            print(f"意外的权重: {unexpected_keys}")

    def _create_weight_mapping(self) -> Dict[str, str]:
        """创建HuggingFace权重名称到模型权重名称的映射（包含lm_head）"""
        weight_map = {}

        # 嵌入层
        weight_map["model.embed_tokens.weight"] = "model.embed_tokens.weight"

        # Transformer层
        for i in range(self.model.config.num_hidden_layers):
            layer_prefix = f"model.layers.{i}"
            model_prefix = f"model.layers.{i}"

            # 注意力层
            weight_map[f"{layer_prefix}.self_attn.q_proj.weight"] = (
                f"{model_prefix}.self_attn.q_proj.weight"
            )
            weight_map[f"{layer_prefix}.self_attn.k_proj.weight"] = (
                f"{model_prefix}.self_attn.k_proj.weight"
            )
            weight_map[f"{layer_prefix}.self_attn.v_proj.weight"] = (
                f"{model_prefix}.self_attn.v_proj.weight"
            )
            weight_map[f"{layer_prefix}.self_attn.o_proj.weight"] = (
                f"{model_prefix}.self_attn.o_proj.weight"
            )
            weight_map[f"{layer_prefix}.self_attn.q_norm.weight"] = (
                f"{model_prefix}.self_attn.q_norm.weight"
            )
            weight_map[f"{layer_prefix}.self_attn.k_norm.weight"] = (
                f"{model_prefix}.self_attn.k_norm.weight"
            )

            # MLP层
            weight_map[f"{layer_prefix}.mlp.gate_proj.weight"] = (
                f"{model_prefix}.mlp.gate_proj.weight"
            )
            weight_map[f"{layer_prefix}.mlp.up_proj.weight"] = (
                f"{model_prefix}.mlp.up_proj.weight"
            )
            weight_map[f"{layer_prefix}.mlp.down_proj.weight"] = (
                f"{model_prefix}.mlp.down_proj.weight"
            )

            # 层归一化
            weight_map[f"{layer_prefix}.input_layernorm.weight"] = (
                f"{model_prefix}.input_layernorm.weight"
            )
            weight_map[f"{layer_prefix}.post_attention_layernorm.weight"] = (
                f"{model_prefix}.post_attention_layernorm.weight"
            )

        # 最终层归一化
        weight_map["model.norm.weight"] = "model.norm.weight"

        # 语言模型头
        weight_map["lm_head.weight"] = "lm_head.weight"

        return weight_map

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def post_init(self):
        """初始化权重"""
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        std = self.model.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, torch.Tensor]:

        # decoder输出包含(dec_features, layer_state, dec_hidden, dec_attn)
        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # 计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return (loss, logits) if loss is not None else logits
