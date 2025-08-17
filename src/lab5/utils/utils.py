import time
import torch
import torch.nn.functional as F


def generate_text(
    model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50, device="cpu"
):
    """生成文本"""
    print(f"\n输入提示: {prompt}")
    print("=" * 50)

    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    print(f"输入 token 数量: {input_ids.shape[1]}")

    # 生成设置
    generated_ids = input_ids.clone()

    start_time = time.time()

    with torch.no_grad():
        for step in range(max_length):

            # 前向传播
            results = model(input_ids=generated_ids, attention_mask=attention_mask)
            if hasattr(results, "logits"):
                logits = results.logits
            else:
                logits = results

            if temperature == 0:
                # 直接取最大值作为下一个 token
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            else:
                # 获取下一个 token 的概率分布
                next_token_logits = logits[:, -1, :] / temperature

                # Top-k 采样
                if top_k > 0:
                    # 获取 top-k 个最高的分数
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    # 创建掩码，只保留 top-k
                    next_token_logits = torch.full_like(
                        next_token_logits, float("-inf")
                    )
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)

                # 应用 softmax 并采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # 添加到生成的序列中
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # 更新注意力掩码
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        device=device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=-1,
            )

            # 检查是否生成了结束 token
            if next_token.item() == tokenizer.eos_token_id:
                break

    end_time = time.time()

    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print(f"\n生成完成!")
    print(f"生成时间: {end_time - start_time:.2f} 秒")
    print(f"生成 token 数量: {generated_ids.shape[1] - input_ids.shape[1]}")
    print(
        f"生成速度: {(generated_ids.shape[1] - input_ids.shape[1]) / (end_time - start_time):.2f} tokens/秒"
    )
    print("=" * 50)
    print(f"完整输出: {generated_text}")

    return generated_text
