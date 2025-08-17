import torch
import numpy as np
import gc
import unittest
from transformers import AutoTokenizer, AutoModelForCausalLM
from modules import Qwen3Config, Qwen3ForCausalLM
from utils.utils import generate_text


# ANSI颜色代码
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"  # 结束颜色


def load_custom_model():
    """只加载自定义模型"""
    MODEL_PATH = "/ocean/model/Qwen3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    torch_dtype = torch.bfloat16

    custom_model = Qwen3ForCausalLM.from_pretrained(
        model_path=MODEL_PATH,
        device=device,
        torch_dtype=torch_dtype,
    )
    custom_model.eval()
    return custom_model, device, torch_dtype


def load_baseline_model():
    """只加载基线模型"""
    MODEL_PATH = "/ocean/model/Qwen3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    torch_dtype = torch.bfloat16

    baseline_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    baseline_model.model.to(device)
    baseline_model.eval()
    return baseline_model, device, torch_dtype


def load_tokenizer():
    """加载tokenizer"""
    MODEL_PATH = "/ocean/model/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=False
    )
    return tokenizer


def cleanup_model(model):
    """清理模型并释放内存"""
    if model is not None:
        del model

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 强制垃圾回收
    gc.collect()


def generate_with_custom_model(model, tokenizer, prompt, max_length=20, device="cpu"):
    """使用自定义模型生成文本"""
    return generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        temperature=0,
        device=device,
    )


def generate_with_baseline_model(model, tokenizer, prompt, max_length=20, device="cpu"):
    """使用transformers基线模型生成文本"""
    return generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        temperature=0,
        device=device,
    )


def compare_model_outputs():
    """比较两个模型的输出一致性"""
    # 加载tokenizer（两个模型共用）
    tokenizer = load_tokenizer()

    # 测试用的提示词
    test_prompts = [
        "Once upon a time in a land far, far away, there lived a dragon who",
        "The quick brown fox",
        "在一个阳光明媚的早晨",
        "To be or not to be",
    ]

    max_length = 10  # 较短的生成长度以便快速测试
    results = []
    custom_outputs = []
    baseline_outputs = []

    # ================ 测试自定义模型 ================
    print(f"{Colors.BLUE}正在加载目标模型...{Colors.END}")
    custom_model, device, torch_dtype = load_custom_model()
    print(f"{Colors.BLUE}目标模型生成中...{Colors.END}")
    for i, prompt in enumerate(test_prompts):
        custom_output = generate_with_custom_model(
            custom_model, tokenizer, prompt, max_length, device
        )
        custom_outputs.append(custom_output)
    # 清理自定义模型
    del custom_model
    print(f"{Colors.GREEN}目标模型测试完成{Colors.END}")

    # ================ 测试基线模型 ================
    print(f"{Colors.BLUE}正在加载基线模型...{Colors.END}")
    baseline_model, device, torch_dtype = load_baseline_model()
    print(f"{Colors.BLUE}基线模型生成中...{Colors.END}")
    for i, prompt in enumerate(test_prompts):
        baseline_output = generate_with_baseline_model(
            baseline_model, tokenizer, prompt, max_length, device
        )
        baseline_outputs.append(baseline_output)
    # 清理基线模型
    cleanup_model(baseline_model)
    print(f"{Colors.GREEN}基线模型测试完成{Colors.END}")

    # 比较输出
    print("\n" + Colors.BOLD + "=" * 60 + Colors.END)
    print(Colors.BOLD + Colors.MAGENTA + " " * 20 + "测试结果对比" + Colors.END)
    print(Colors.BOLD + "=" * 60 + Colors.END)

    for i, prompt in enumerate(test_prompts):
        is_consistent = custom_outputs[i] == baseline_outputs[i]
        results.append(
            {
                "prompt": prompt,
                "custom_output": custom_outputs[i],
                "baseline_output": baseline_outputs[i],
                "is_consistent": is_consistent,
            }
        )

        # 美化输出
        if is_consistent:
            status_icon = "✅"
            status_text = f"{Colors.GREEN}通过{Colors.END}"
        else:
            status_icon = "❌"
            status_text = f"{Colors.RED}失败{Colors.END}"

        print(f"\n{Colors.BOLD}测试点 {i+1}: {status_icon} {status_text}{Colors.END}")
        print(f"{Colors.CYAN}提示词:{Colors.END} {prompt}")
        print(f"{Colors.YELLOW}目标模型:{Colors.END} {custom_outputs[i]}")
        print(f"{Colors.YELLOW}基线模型:{Colors.END} {baseline_outputs[i]}")
        if not is_consistent:
            print(f"{Colors.RED}⚠️  输出不一致{Colors.END}")
        print(Colors.WHITE + "-" * 50 + Colors.END)

    print(f"{Colors.GREEN}所有测试完成{Colors.END}")

    return results


def main():
    """主函数"""
    print(f"{Colors.BOLD}{Colors.CYAN}开始模型一致性测试...{Colors.END}")

    try:
        results = compare_model_outputs()

        print("\n" + Colors.BOLD + "=" * 60 + Colors.END)
        print(Colors.BOLD + Colors.MAGENTA + " " * 20 + "最终测试汇总" + Colors.END)
        print(Colors.BOLD + "=" * 60 + Colors.END)

        consistent_count = sum(1 for r in results if r["is_consistent"])
        total_count = len(results)

        print(
            f"{Colors.BOLD}一致性测试结果: {Colors.CYAN}{consistent_count}/{total_count}{Colors.END} {Colors.BOLD}通过{Colors.END}"
        )

        if consistent_count == total_count:
            print(
                f"{Colors.GREEN}{Colors.BOLD}🎉 恭喜！所有测试都通过了！模型实现正确！{Colors.END}"
            )
        else:
            print(
                f"{Colors.RED}{Colors.BOLD}⚠️  有 {total_count - consistent_count} 个测试失败，请检查模型实现{Colors.END}"
            )

    except Exception as e:
        print(f"{Colors.RED}测试过程中出现错误: {e}{Colors.END}")
    finally:
        # 最终清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
