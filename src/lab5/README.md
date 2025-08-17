# 实验五: TinyLLM

## 实验框架结构

实验框架的目录结构如下：

```text
.
├── main.py             # 简单推理脚本
├── modules
│   ├── attention.py    # TODO: 完成 Qwen3Attention 的实现
│   ├── config.py       # Qwen3 模型配置类
│   ├── __init__.py
│   ├── layer.py        # Qwen3 Decoder Layer 的实现
│   ├── mlp.py          # TODO: 完成 Qwen3FFN 的实现
│   ├── model.py        # Qwen3 模型的实现
│   └── rope.py         # RoPE 位置编码的实现
├── README.md
├── simple_test.py      # 单元测试脚本
└── utils
```

你需要完成 `modules/attention.py` 中的 `Qwen3Attention` 类和 `modules/mlp.py` 中的 `Qwen3FFN` 类的实现。

我们提供了一个简单的推理脚本 `main.py`，你可以通过运行该脚本来测试你的模型是否能够输出正常的人类可读文本。你也可以通过修改 `main.py` 中的生成配置来探索不同的生成效果（如调整不同的参数、使用 `<think>` 标签强制模型思考等），欢迎同学们将自己的尝试和结果分享在实验报告中。

我们也提供了测试用例，你需要确保你的实现能够通过测试。你可以使用以下命令来运行测试：

```bash
python simple_test.py
```

## 开发环境设置

本次实验使用 [UV](https://docs.astral.sh/uv/) 进行环境管理。UV 是一个现代化的 Python 环境管理工具，支持多种 Python 版本和依赖包管理。

首先，你需要安装 UV。参考 UV 文档中的 [安装手册](https://docs.astral.sh/uv/getting-started/installation/) 使用以下命令来安装 UV：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

本次实验提供了一个预配置的 UV 环境，你可以通过以下命令来加载环境:

```bash
source /river/hpc101/2025/lab5/env/bin/activate
```

加载后直接使用 `python main.py` 命令即可运行实验代码。

## 提交作业

本实验需要在 GPU 平台上运行，你可以参考 [提交作业 - 提交 GPU 任务](https://hpc101.zjusct.io/guide/job/#%E6%8F%90%E4%BA%A4gpu%E4%BB%BB%E5%8A%A1) 来申请 GPU 资源并提交你的任务。

需要注意的是，你需要确保在运行前加载 UV 环境。你可以在 SBATCH 提交脚本中执行加载环境的命令，或者在使用 `srun` 命令前确保已经加载了 UV 环境。
