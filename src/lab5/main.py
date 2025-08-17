import torch
import torch.nn.functional as F
from modules import Qwen3Config, Qwen3ForCausalLM
from transformers import AutoTokenizer
import time
import os
from utils.utils import generate_text


def main():
    MODEL_PATH = "/ocean/model/Qwen3-8B"
    PROMPT = "为什么西瓜比苹果甜？<think>"  # 示例提示词，添加<think>标记以触发思考模式
    max_length = 200

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen3ForCausalLM.from_pretrained(
        model_path=MODEL_PATH,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=False
    )
    model.eval()

    generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=PROMPT,
        max_length=max_length,
        temperature=0,
        device=device,
    )


if __name__ == "__main__":
    main()
