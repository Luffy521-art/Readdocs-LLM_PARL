# LLM_PARL 项目自述文件 (README)

## 项目简介

**LLM_PARL** 是一个面向 SUMO 路网 XML 生成任务的 PPO (Proximal Policy Optimization) 训练与进化实验项目。项目旨在通过大语言模型（主要使用 Qwen2.5 系列模型）自动生成、修改和优化交通仿真软件 SUMO 所需的路网 XML 配置文件（如节点文件、边文件及连接文件等），并通过外部评分器与强化学习不断迭代提升生成质量。

---

## 核心功能与流程

* **数据驱动的提示词机制**：支持从结构化 JSON 数据集（如 SUMO 路网分步任务数据）中读取 Prompt 和标准标签。
* **多阶段大模型生成 (Rollout)**：利用 Qwen Actor 模型根据路网编辑需求自动生成对应的候选 XML 响应文本。
* **多维度智能评分与奖励计算**：
* 支持结合三维约束（结构完整性、边删除硬约束、计数对齐）的本地/外部评分器。
* 支持通过 OpenAI 兼容的 Chat Completions API（如 DeepSeek 等异步打分服务）进行高并发质量评分。


* **PPO 强化学习对齐更新**：结合 Actor、Critic 以及 Reference Model，构建完整的 PPO 训练循环，对模型的 Token 级或轨迹级表现进行策略优化。
* **PARL 奖励对齐框架**：引入基于 ModernBERT 的奖励模型（SUMORewardModel）及隐式梯度/Fisher 损失约束，实现更稳定高效的对齐训练。
* **自动化监控与日志记录**：
* 内置多卡 GPU 自动探测与守护启动脚本（支持显存轮询与容灾重启）。
* 训练过程中自动写入评估指标 CSV、样本明细、Checkpoint 断点，并支持通过 Weights & Biases (W&B) 和 QQ 邮件进行监控提醒。



---

## 项目代码结构

* `configs/`：训练、评分、数据路径等 YAML 配置文件。
* `src/llm/`：包含 Qwen 模型的 Actor/Critic 架构、PPO 逻辑以及基于 ModernBERT 的奖励模型定义 (`qwen_module.py`, `reward_model.py`)。
* `src/pipeline/`：核心训练步骤流水线，包括生成收集、奖励计算、PPO 数据构造以及 PARL 训练流程 (`runner_LLM_new.py`, `steps_LLM.py`, `steps_parl.py`)。
* `src/metrics/`：外部 API 异步评分及规则校验逻辑。
* `src/evolution/`：进化策略与种群接口。
* `test/`：各类集成测试脚本、验证脚本以及日志分析工具。
* `data/`：SUMO 任务相关的数据集文件。

---

## 快速上手与运行

### 1. 环境准备

确保在 Linux 环境下，配置好 Python 3.10 及对应的虚拟环境（推荐项目已有的 `env_openrlhf` 运行环境），并安装 `requirements.txt` 中的基础依赖。

```bash
# 激活环境
conda activate /zjl/xmf/env_openrlhf
cd /zjl/xmf/LLM_PARL

# 安装依赖
pip install -r requirements.txt

```

### 2. 设置环境变量

配置项目根目录及评分器 API 密钥（以环境变量形式传入，避免写入配置文件）：

```bash
export MASDIFF_ROOT=/zjl/xmf
export HF_ENDPOINT=https://hf-mirror.com
export SCORER_API_KEY="你的评分器API密钥"

```

### 3. 启动训练

可以通过通用配置文件直接运行主流程：

```bash
python run.py --config configs/default.yaml

```

或者使用自动多卡 GPU 守护启动脚本运行：

```bash
python auto_gpu_launcher.py

```
