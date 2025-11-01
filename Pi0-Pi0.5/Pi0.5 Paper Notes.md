# Paper-Pi05: a Vision-Language-Action Model with Open-World Generalization

[Paper arxiv](https://arxiv.org/abs/2504.16054)

[Local PDF](papers/Paper-Pi05.pdf)

## Abstract（摘要）

**目前问题**：虽然VLA模型在端到端机器人控制上表现优异，但**开放世界泛化** (open-world generalization) 仍然是未解决的核心挑战。

**解决方案**：π0.5基于π0，使用**异构任务联合训练** (co-training) 实现广泛泛化。关键在于"**异构数据** (heterogeneous data)" - 来自不同类型、不同来源的任务数据。

**技术路线**：

- **多源数据融合**：多机器人数据 + 高层语义预测 + 网络数据等
- **混合多模态样本**：图像观察 + 语言指令 + 物体检测 + 语义子任务预测 + 底层动作  
- **知识迁移验证**：证明知识迁移对有效泛化的重要性

**突破性成果**：**首次实现**端到端学习驱动的机器人系统在全新家庭环境中执行长时程精细操作技能（如厨房和卧室清理）。

![alt text](<Pi0.5 Paper Notes.assets/image.png>)

## I. INTRODUCTION（引言）

### 问题本质：从"能力不足"到"泛化不足"

当前机器人的根本问题不是技术能力（敏捷性、灵活性），而是**泛化能力**。大多数VLA模型在实验室表现优秀，但一到真实世界就失效。

**重要观点**：机器人需要的不是更强的单项能力，而是在**多个抽象层面**同时泛化的能力。

### 泛化的三个层次（以厨房清理为例）

| 泛化层次     | 难度 | 例子                               | 解决方案                 |
| ------------ | ---- | ---------------------------------- | ------------------------ |
| **感知泛化** | 低   | 识别不同外观的盘子、刀具           | 足够的视觉多样性         |
| **技能泛化** | 中   | 用新的方式或序列组合已有动作       | 跨环境、跨任务的技能迁移 |
| **语义泛化** | 高   | 理解"哪个是晾衣架"、"该开哪个抽屉" | 常识知识和推理能力       |

**关键发现**：不同层次的泛化需要**不同类型的数据源**，无法通过单一数据类型解决。

### 人类学习的启发：多源知识融合

人类解决新问题的能力来自**知识的组合运用**：

- **间接知识**：书本、他人经验 → 对应网络数据的语义知识
- **类比迁移**：其他任务的经验 → 对应跨机器人平台的技能迁移  
- **直接经验**：目标任务练习 → 对应目标平台的直接数据

**重要发现**：**知识多样性 > 数据相关性**。97.6%的"不相关"数据通过合理配方能产生强泛化。

### VLA的技术突破：统一建模框架

传统方法的问题：异构数据源无法有效整合，需要设计复杂的多模型系统。

**VLA的突破**：**序列建模框架** (sequence modeling) 将所有模态（视觉、语言、动作）统一为token序列，使异构数据的联合训练成为可能。

**技术优势**：

1. **统一表示**：不同数据类型可以在同一模型中处理
2. **端到端优化**：避免多模型系统的累积误差
3. **知识共享**：不同数据源的知识可以相互促进

**设计哲学转变**：从"收集更多直接数据"到"设计更好的数据配方"。

## II. RELATED WORK（相关工作）

### 现有工作的进展与局限

**通用机器人操作策略的发展**：

近期工作（BridgeData V2、Open X-Embodiment等）证明将训练数据从单任务扩展到多场景、多任务的多样化数据集能显著提高泛化能力。VLA模型（如RT-2、OpenVLA）通过微调预训练视觉-语言模型，能够利用网络规模预训练获得的语义知识，结合高表达力的动作解码机制（flow matching、diffusion、高级action tokenization），在真实世界执行复杂操作任务。

**根本局限**：尽管语言遵循能力impressive，现有VLA仍然主要在**与训练数据密切匹配的环境**中评估。即使一些研究表明简单技能（如抓取物体、开抽屉）可以通过在更广泛环境中收集数据来泛化，但将同样方法应用到更复杂的长时程任务（如清理厨房）仍然困难，通过暴力扩展收集数据来覆盖所有可能场景是不可行的。

### π0.5的技术突破点

| 技术方向                 | 现有方法局限                       | π0.5的创新                                       |
| ------------------------ | ---------------------------------- | ------------------------------------------------ |
| **非机器人数据联合训练** | 主要局限于VLM数据混合              | 设计更广泛的机器人相关监督源联合训练系统         |
| **高层推理与规划**       | 使用两个独立模型（VLM + 低层策略） | **同一模型**进行高低层推理，类似chain-of-thought |
| **开放世界泛化**         | 限制在基本原语，任务相对简单       | 长时多阶段任务（10-15分钟），在全新家庭环境      |
| **数据利用策略**         | 主要依赖目标任务相关数据           | 97.6%"不相关"数据通过co-training实现强泛化       |

### 非机器人数据联合训练的探索

许多先前工作尝试使用多样化的非机器人数据来改进机器人策略的泛化：

- **视觉编码器初始化**：从计算机视觉数据集初始化视觉编码器
- **VLM数据混合**：RT-2等展示了与VLM训练数据联合训练能改善泛化，特别是与新物体或未见过的场景背景交互时
- **现成规划器**：利用现成的任务规划器辅助机器人控制

**π0.5的超越**：超越了VLM数据联合训练，设计了一个系统来与**更广泛的机器人相关监督源**进行联合训练，包括其他机器人数据、高层语义子任务预测、语言指导演示等。实验表明这种特定的数据源组合能够让移动机器人在全新环境中执行复杂长时程行为。

### 高层推理与语言规划

许多工作表明用高层推理增强端到端策略可以显著改善长时程任务性能，特别是当高层子任务推理可以利用大型预训练LLM和VLM时。

**现有方法**：通常使用**两个独立模型** - VLM预测语义步骤，独立的低层策略执行这些步骤。

**π0.5的方法**：也使用两阶段推理（首先推理高层语义子任务，然后基于此预测动作），但使用**同一个模型**进行高层和低层推理，配方更接近chain-of-thought或test-time compute方法。关键区别是高层推理仍然以低于低层动作推理的频率运行。

**技术优势**：统一模型使得高低层推理能够共享内部表示，不仅通过文本还通过hidden states传递信息，避免了两模型系统的理解不一致和优化困难问题。

### 开放世界泛化的评估

**现有工作的局限**：

大多数机器人学习系统在与训练数据密切匹配的环境中评估。当机器人任务限制在更窄的基本原语集合（如抓取）时，一些允许任务特定假设的方法（如grasp prediction、model-based planning and control）已经显示能够广泛泛化甚至到全新家庭，但这些方法不容易泛化到通用机器人可能需要执行的全部任务范围。

近期大规模数据集工作显示简单的端到端学习任务能够泛化到新环境，但这些演示中的任务仍然相对简单，通常少于1分钟长度，成功率往往较低。

**π0.5的突破**：

展示π0.5能够执行长时多阶段任务，如把所有盘子放进水槽或把所有衣物从新卧室地板上拾起，同时泛化到**完全新的家庭环境**（entirely new homes）。这是首次端到端学习驱动的机器人系统在全新环境中执行如此复杂和长时程的操作技能（10-15分钟）。

## III. PRELIMINARIES（预备知识）

### VLA的训练目标

VLA模型通过模仿学习训练，目标是最大化给定观察和语言指令下，动作序列的对数似然：

$$
\max_\theta \mathbb{E}_{(a_{t:t+H}, o_t, \ell) \sim D} \left[ \log \pi_\theta(a_{t:t+H} | o_t, \ell) \right]
$$

**公式含义**：

- **θ**：模型参数（神经网络的权重），这是训练的目标。
- **D**：机器人演示数据集，包含以下的三元组的数据。
- **数据三元组（下标）**：
  - $a_{t:t+H}$：动作序列（action chunk），从时刻t到t+H这一段时间内的连续动作（如50步×7维=350个数值）。
  - $o_t$：时刻t的观察（由对环境的外部感知以及对自身状态的内部感知组成，详见[观察的组成结构](#观察的组成结构)）
  - $ℓ$：语言指令
- **期望 E**：对所有训练数据求平均
- **log π_θ(...)**：模型预测该动作序列的对数概率。取对数在保持优化目标不变的同时把乘法变成加法，避免数值计算问题。

**公式整体含义**：我们想要训练一组模型的参数θ，使得模型对于给定数据集D中的所有三元组（动作序列、观察、语言指令），能根据当前的观察和语言指令，预测出正确动作序列的对数概率的平均值最大化。

**训练过程**：通过调整参数θ，让模型在所有训练数据上预测正确动作序列的平均对数概率最大化。实际训练时用随机梯度下降，每次采样一小批数据计算loss并更新参数。

### 观察的组成结构

观察 $o_t$ 不是单一数据，而是**外部感知 + 内部感知**的组合：

#### 1. 外部感知 (Exteroception)

多视角图像 $I_1^t, ..., I_n^t$：

- 通常3个摄像头（手腕视角、头部视角、外部视角，具体数量根据具体任务和机器人平台而定）
- 每张图像如224×224×3的RGB图像
- 提供环境、物体位置等信息

#### 2. 内部感知 (Proprioception)

本体感知状态 $q_t$：

- **关节角度**：每个关节当前的角度值
- **夹爪状态**：开合程度（0到1）
- **底盘状态**（移动机器人）：位置、朝向、速度

**为什么需要两者结合？**

- 只有图像：看到物体但不知道自己手臂在哪
- 只有本体感知：知道姿势但不知道周围环境
- 两者结合：能够规划"如何从当前姿势到达目标"

**代码实现**：

**源代码位置**：[`pi0_pytorch.py` - `_preprocess_observation()` 方法（L161-L170）](../Repo/openpi/src/openpi/models_pytorch/pi0_pytorch.py#L161-L170)

观察预处理的核心代码：

```python
def _preprocess_observation(self, observation, *, train=True):
    """将原始观察转换为模型输入格式"""
    observation = preprocess_observation_pytorch(observation, train=train)
    return (
        list(observation.images.values()),        # 多个相机图像
        list(observation.image_masks.values()),   # 图像mask
        observation.tokenized_prompt,             # 文本tokens
        observation.tokenized_prompt_mask,        # 文本mask
        observation.state,                        # 本体感知状态
    )
```

预处理后在模型中的使用：

```python
# 在forward()中调用
images, img_masks, lang_tokens, lang_masks, state = \
    self._preprocess_observation(observation, train=True)
```

### VLA架构设计

**基本架构**：遵循现代视觉-语言模型设计

| 组件                   | 作用                       | 技术                            |
| ---------------------- | -------------------------- | ------------------------------- |
| **模态特定tokenizers** | 将不同模态转成token表示    | 图像patches、文本分词、动作编码 |
| **自回归transformer**  | 从输入tokens生成输出tokens | 标准transformer架构             |
| **权重初始化**         | 从预训练VLM初始化          | 利用网络规模的语义知识          |

**统一表示机制**：

不同模态的数据（图像、文本、动作）含义完全不同，无法直接拼接。解决方案是通过**Embedding层**将所有模态转成相同维度的向量：

| 模态     | 原始格式          | Encoder           | 统一表示  |
| -------- | ----------------- | ----------------- | --------- |
| 文本     | "pick" → ID 1024  | Embedding Matrix  | 768维向量 |
| 图像     | 16×16×3 像素patch | Linear Projection | 768维向量 |
| 本体感知 | [θ1, θ2, ..., θ7] | Linear Layer      | 768维向量 |
| 动作     | 离散ID或连续值    | Embedding/Linear  | 768维向量 |

**统一到相同维度后**，所有tokens可以拼接成一个序列输入transformer，通过Attention机制互相交互。

**自回归生成**：模型逐个token生成输出，每次生成依赖前面已生成的tokens，类似GPT生成文本的方式。

### 动作表示的两种方法

**为什么需要特殊的动作表示？**

动作是连续值（如关节角度0.523 rad），与离散的文本token不同。需要特殊方法将动作转成token或向量表示。

| 方法                     | 表示形式      | 训练         | 推理             | 精度             |
| ------------------------ | ------------- | ------------ | ---------------- | ---------------- |
| **离散化 (FAST)**        | 8个离散tokens | 快速、稳定   | 自回归解码（慢） | 中（有量化误差） |
| **连续 (Flow Matching)** | 350维连续向量 | 需要迭代训练 | 迭代去噪（10步） | 高               |

**π0.5的策略**：

- **Pre-training**：使用FAST离散tokens（训练效率高，和文本统一处理）
- **Post-training**：切换到Flow Matching（精度高，适合实时控制）

### FAST编码原理

**问题**：50步×7维 = 350个连续值，如何高效编码？

**FAST的解决方案**：利用**时间相关性**压缩

**时间相关性**：机器人动作是平滑的，相邻时刻的动作非常相似。就像视频压缩利用相邻帧的相似性，FAST利用相邻时刻动作的相似性。

**压缩类比**：动作的"关键帧"

```text
原始: 50步完整动作序列（350个数）
  ↓ Encoder
8个关键"模式"（8个离散tokens）
  ↓ Decoder  
重建: 50步完整动作序列
```

**技术实现**：Vector Quantization (VQ)

1. **学习Codebook（动作模式库）**：包含256个codes，每个code是一个抽象的动作特征向量
2. **编码**：将350维动作映射到8个code IDs
3. **解码**：从8个codes通过神经网络decoder重建350维动作

**Code的本质**：

类似Language Model中的token：

- 离散的符号（ID）
- 对应抽象的向量表示
- 含义通过神经网络学习
- 不是人类可直接解释的"指令"，而是网络内部的抽象编码

**压缩效果**：350个数 → 8个tokens，压缩比约44倍

| 优势                | 劣势                      |
| :------------------ | :------------------------ |
| 序列短，训练快速    | 有量化误差                |
| 和文本token统一处理 | 推理需要8步自回归（较慢） |

**代码实现**：

**源代码位置**：[`fsq_tokenizer.py` - `FsqAttentionTokenizer` 类](../Repo/openpi/src/openpi/models/utils/fsq_tokenizer.py)

FAST编码使用FSQ（Finite Scalar Quantization）实现：

```python
class FsqAttentionTokenizer:
    """动作序列的压缩编码器
    
    架构：Encoder -> FSQ Codebook -> Decoder
    - 输入：[batch, 50 steps, 7 dims] = 350个连续值
    - 输出：8个离散tokens
    """
    def tokenize(self, action, obs=None, train=False):
        """编码：350维动作 -> 8个tokens"""
        x = self.proj(action)                    # 投影到embed_dim
        x = self.encoder(x, state_conditioning=obs)  # Transformer编码
        return self.codebook.encode(x)           # FSQ量化 -> 8 tokens
    
    def detokenize(self, tokens, obs=None):
        """解码：8个tokens -> 350维动作"""
        x = self.codebook.decode(tokens)         # tokens -> 连续向量
        x = self.decoder(x, state_conditioning=obs)  # Transformer解码
        mean = self.proj_mean(x) * self.out_scale
        return mean  # [batch, 50, 7]
```

**FSQ vs VQ的区别**：

- **VQ（Vector Quantization）**：需要学习codebook embeddings
- **FSQ（Finite Scalar Quantization）**：使用固定的离散bins，无需学习codebook
- π0/π0.5使用FSQ，更简单稳定

注：代码为JAX版本，PyTorch版本在训练时使用，但推理时π0.5不使用FAST（只用FM）

### Flow Matching for 连续动作（在Post-training阶段使用）

**核心思想**：学习从噪声到真实动作的"流动路径"

**数学表示**：

从噪声 $x_0 \sim \mathcal{N}(0,I)$ 到真实动作 $x_1 = a_{t:t+H}$ 建立线性插值路径：

$$
x_\tau = (1-\tau) \cdot x_0 + \tau \cdot x_1, \quad \tau \in [0,1]
$$

路径的速度场（方向）：

$$
v_\tau = \frac{dx}{d\tau} = x_1 - x_0
$$

**训练目标**：

训练神经网络 $v_\theta(x, \tau)$ 预测速度场：

$$
L = \mathbb{E}_{\tau, x_0, x_1} \left[ ||v_\theta(x_\tau, \tau) - (x_1 - x_0)||^2 \right]
$$

**推理过程（生成动作）**：

从纯噪声开始，迭代更新：

$$
x_{i+1} = x_i + v_\theta(x_i, \tau_i) \cdot \Delta\tau
$$

```text
Step 0: x = 随机噪声
Step 1: x = x + v_θ(x, 0.0) × 0.1
Step 2: x = x + v_θ(x, 0.1) × 0.1
...
Step 10: x = 最终清晰的动作
```

**只需10步迭代即可生成高质量动作，比Diffusion Model的50-1000步快得多！**

**Flow Matching vs Diffusion**：

| 特性     | Diffusion   | Flow Matching          |
| -------- | ----------- | ---------------------- |
| 路径类型 | 随机（SDE） | 确定性（ODE）          |
| 训练目标 | 预测噪声    | 预测速度场             |
| 推理步数 | 50-1000步   | 5-10步                 |
| 实时性   | 慢          | 快，适合50Hz机器人控制 |

**π0.5选择Flow Matching的原因**：推理快速，满足实时控制需求。

**代码实现**：

**源代码位置**：[`pi0_pytorch.py` - `forward()` 和 `sample_actions()` 方法](../Repo/openpi/src/openpi/models_pytorch/pi0_pytorch.py)

训练时的核心代码（计算速度场loss）：

```python
# 训练：学习速度场
# 1. 采样时间步（使用Beta分布，强调低噪声）
time = self.sample_time(batch_size, device)  # Beta(1.5, 1.0)
time_expanded = time[:, None, None]

# 2. 构造插值（部分去噪的动作）
x_t = time_expanded * noise + (1 - time_expanded) * actions  # 插值
u_t = noise - actions  # 理想速度场（从噪声到真实动作）

# 3. 网络前向传播
# ... 网络前向传播 ...
v_t = self.action_out_proj(suffix_out)  # 预测的速度场

# 4. 计算MSE loss
return F.mse_loss(u_t, v_t, reduction="none")  # MSE loss
```

**时间步采样（Beta分布）**：

**源代码位置**：[`pi0_pytorch.py` - `sample_time()` 和 `sample_beta()` 函数（L45-L49, L181-L184）](../Repo/openpi/src/openpi/models_pytorch/pi0_pytorch.py#L45-L49)

π0/π0.5使用Beta(1.5, 1.0)分布采样时间步：

```python
def sample_beta(alpha, beta, bsize, device):
    """从Beta分布采样"""
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))

def sample_time(self, bsize, device):
    """采样训练用的时间步，Beta(1.5, 1.0)偏向低噪声"""
    time_beta = sample_beta(1.5, 1.0, bsize, device)  # [0, 1]
    time = time_beta * 0.999 + 0.001  # 缩放到[0.001, 0.999]
    return time.to(dtype=torch.float32, device=device)
```

**为什么使用Beta(1.5, 1.0)？**

- Beta(1.5, 1.0)分布在靠近1的地方概率更高（低噪声区域）
- 机器人控制更关注精细调整（低噪声），而非粗略规划（高噪声）
- 详细讲解请参考：[Flow Matching Notes - Beta分布采样](../Flow-Matching/Flow%20Matching%20Notes.md#7-beta分布采样sample_beta-和-sample_time-方法)

推理时的核心代码（ODE求解）：

```python
# 推理：从噪声逐步生成动作
dt = -1.0 / num_steps  # 时间步长（负数表示从1到0）
x_t = noise  # 从纯噪声开始
time = 1.0
while time >= -dt / 2:
    v_t = self.denoise_step(state, ..., x_t, time)  # 预测速度场
    x_t = x_t + dt * v_t  # 欧拉法更新
    time += dt
return x_t  # 最终生成的动作
```

关于Diffusion Model和Flow Matching的更多细节，请参考：

**[笔记 - Denoising Diffusion Probabilistic Models Tutorial（扩散模型）](../Flow-Matching/papers/DDPM-Tutorial.pdf)**

**[笔记 - Flow Matching Explained - From Noise to Robot Actions（流匹配）](../Flow-Matching/papers/Flow-Matching-Explained.pdf)**

**[笔记 - Flow Matching Notes](../Flow-Matching/Flow%20Matching%20Notes.md)**

### Action Expert设计

**什么是Action Expert？**

一个专门的小型transformer（300M参数），负责Flow Matching的动作生成。

**架构设计**：

```text
输入tokens → 主Transformer (2B参数)
              ↓
      ┌──────┴──────┐
      ↓              ↓
   文本tokens    动作tokens
      ↓              ↓
  主模型处理    Action Expert处理
      ↓              ↓
   文本输出      连续动作输出
```

**Action Expert的输入**：

1. 主transformer的内部表示（语义理解）
2. 当前的部分去噪动作 $x_\tau$
3. 时间步 $\tau$

**Action Expert的输出**：

- 速度场 $v_\theta(x_\tau, \tau)$

**为什么需要单独的Action Expert？**

1. **专业化**：主模型理解语义，Action Expert专注精确动作生成
2. **效率**：Action Expert较小（300M vs 2B），Flow Matching需要迭代10次更经济
3. **模块化**：Pre-training不需要，Post-training时才加入

**类比Mixture of Experts (MoE)**：不同类型的tokens由不同的"专家"模块处理。

**代码实现**：

**源代码位置**：[`gemma_pytorch.py` - `PaliGemmaWithExpertModel` 类](../Repo/openpi/src/openpi/models_pytorch/gemma_pytorch.py)

Action Expert的前向传播（区分主模型和专家模型）：

```python
# 根据token类型选择不同的Transformer处理
def forward(self, inputs_embeds, adarms_cond, ...):
    prefix_embs, suffix_embs = inputs_embeds  # 主模型 vs Action Expert
    prefix_cond, suffix_cond = adarms_cond    # 对应的条件
    
    # 主模型处理prefix（图像、文本等）
    prefix_out = self.paligemma.forward(prefix_embs, ...)
    
    # Action Expert处理suffix（动作tokens）
    suffix_out = self.gemma_expert.forward(
        suffix_embs, 
        adarms_cond=suffix_cond,  # 时间步条件注入
        ...
    )
    return (prefix_out, suffix_out)
```

详细实现请参考：[Flow Matching Notes - Action Expert完整实现](../Flow-Matching/Flow%20Matching%20Notes.md#5-π05关键创新gemmarmnorm-类)

### Pre-training vs Post-training对比

| 阶段              | 动作表示          | 模型组件            | 训练特点   | 目的                       |
| ----------------- | ----------------- | ------------------- | ---------- | -------------------------- |
| **Pre-training**  | FAST离散tokens    | VLM + FAST Decoder  | 快速、稳定 | 学习异构数据，建立基础能力 |
| **Post-training** | Flow Matching连续 | VLM + Action Expert | 精确、实时 | 专门优化目标任务的精确控制 |

**设计哲学**：结合两者优势，Pre-training用离散表示快速学习，Post-training用连续表示实现精确控制。

## IV. THE π0.5 MODEL AND TRAINING RECIPE（模型架构与训练方法）

### π0.5架构概览与训练流程

![alt text](<Pi0.5 Paper Notes.assets/image-1.png>)

**核心设计思想：**

π0.5采用**两阶段训练 + 层次化推理**的架构，通过在不同阶段使用不同的动作表示方法，实现了训练效率和推理精度的平衡：

1. **Pre-training**：从标准VLM适应到多样化机器人任务，使用离散tokens（FAST），简单、可扩展、训练高效
2. **Post-training**：专门化到移动操作，添加Action Expert + Flow Matching，实现精细动作 + 实时控制
3. **推理时**：层次化推理 - 先产生高级子任务（"做什么"），再预测低级动作（"怎么做"）

#### Stage 1: Pre-training（左半部分）

**目标**：快速学习异构数据，建立基础能力

**Pre-training流程图：**

![alt text](<Pi0.5 Paper Notes.assets/image-2.png>)

**数据来源**（左下角）：

- **Multimodal web & robot data**：网络图像+文本、机器人演示数据
- **Task-specific prompts**：各种任务提示

**数据表示形式**（左上角四种任务类型）：

1. **Language subtasks**："put the plate in the sink"
   - 来源：机器人数据中的人工标注子任务
   - 训练高级语义理解能力

2. **Discretized actions**：`-17, 12, 34, 142, -72, -135`
   - 来源：机器人动作数据，经过FAST编码
   - 将50步×7维的连续动作压缩成8个离散tokens
   - 训练基础动作生成能力

3. **Open vocabulary captions**："a dog catches a frisbee"
   - 来源：网络数据
   - 训练视觉理解和描述能力

4. **Bounding boxes**：`3, 35, 145, 223`
   - 来源：网络或机器人数据
   - 训练目标检测和定位能力

**模型**：Pre-trained VLM（SigLIP 400M + Gemma 2.6B = 2B参数）

**训练方式**：标准的自回归transformer，next-token prediction（包括FAST编码的动作tokens）

**优势**：

- 训练快速（离散tokens，和文本统一处理）
- 可以同时学习多种任务
- 建立强大的视觉-语言理解能力

---

#### Stage 2: Post-training & Inference（右半部分）

**目标**：精细化动作生成，支持实时控制

**Post-training流程图：**

![alt text](<Pi0.5 Paper Notes.assets/image-3.png>)

**数据调整**：

- 使用最任务相关的数据
- 加入**Verbal Instructions**（人类口头指令）
- 去掉实验室数据（CE），专注真实环境

**模型变化**：

- VLM → Pre-trained VLA（在Stage 1基础上继续训练）
- **新增Action Expert**（300M参数的小transformer）

**动作表示转换**：

- 从FAST离散tokens → Flow Matching连续动作
- 输出：`-1.7, 1.25, 3.14, 1.42`（连续值）

**优势**：

- 精度高（连续值，无量化误差）
- 推理快（10步迭代 vs autoregressive解码）
- 适合50Hz实时控制

---

#### 推理时的层次化架构（右半部分的两次调用）

**关键设计**：同一个模型，两次前向传播

**Inference流程图：**

![alt text](<Pi0.5 Paper Notes.assets/image-4.png>)

**第一层推理（高级推理）**：

- **输入**：
  - 观察图像（3个摄像头视角）
  - High-level prompt："clean the bedroom"
  
- **处理**：Pre-trained VLA（主transformer）

- **输出**：Subtask prediction："pick up the pillow"

**第二层推理（低级推理）**：

- **输入**：
  - 观察图像（同样的3个视角）
  - Low-level command："pick up the pillow"（来自第一次的输出）
  
- **处理**：Pre-trained VLA + Action Expert

- **输出**：Continuous actions（通过Flow Matching生成）

**折线的含义**：

- 第一层推理的输出 → 作为第二层推理的输入
- 实现"思考"（做什么）→"执行"（怎么做）的分离

**推理阶段的关键特性：**

1. **不使用FAST**：FAST只在训练时保留（防止遗忘），推理时完全不用
2. **只使用FM**：精确（连续值）+ 实时（并行迭代整个动作序列，迭代次数为FM去噪的次数）
3. **层次化推理**：先思考（预测子任务），再执行（生成动作）
4. **Flow Matching去噪迭代**：从纯噪声逐步细化到清晰动作
5. **整个动作矩阵并行**：整个动作矩阵在每次迭代中同时处理，后一个动作不依赖前一个动作

---

#### Action Expert与VLA的关系

**架构设计**：

```text
π0.5 = Pre-trained VLA (主transformer, 2B)
       +
       Action Expert (专家模块, 300M)
```

**信息流动**（通过Attention Mask控制）：

![alt text](<Pi0.5 Paper Notes.assets/image-5.png>)

**Attention规则详解**：

| Token类型      | 可以attend到                        | 不能attend到     |
| -------------- | ----------------------------------- | ---------------- |
| 图像tokens     | 所有VLA tokens（双向）              | Action Expert    |
| 文本tokens     | 所有VLA tokens（双向）              | Action Expert    |
| 本体感知tokens | 所有VLA tokens（双向）              | Action Expert    |
| FAST tokens    | 所有VLA tokens（双向）              | Action Expert    |
| Action Expert  | 图像、文本、本体感知、其他AE tokens | FAST tokens, VLA |

**为什么这样设计？**

1. **VLA内部双向attention**：图像、文本、本体感知、FAST tokens互相可见，充分交互
2. **Action Expert单向读取**：可以看VLA的语义信息（图像、文本、本体感知），但**不能看FAST tokens**
3. **避免信息泄露**：如果Action Expert能看到FAST的离散表示，可能会"作弊"，不学习真正的FM
4. **VLA不依赖Action Expert**：保持VLA的通用性，可以单独用于文本生成等任务

**为什么分离Action Expert？**

1. **效率**：FM需要迭代10次，用300M小模型比2B快
2. **专业化**：动作生成和语义理解是不同任务
3. **模块化**：Pre-training不需要，Post-training才加入

---

#### 整体架构设计总结

**两阶段训练**：

- Pre-training：离散tokens，快速学习异构数据
- Post-training：Flow Matching，精确实时控制

**两阶段推理**：

- 高级推理：预测子任务（做什么）
- 低级推理：生成动作（怎么做）

**两种动作表示**：

- FAST（训练阶段）：8个离散tokens，训练快
- Flow Matching（推理阶段）：连续向量，精度高

---

### A. The π0.5 Architecture（模型架构）

#### 模型的双重输出能力

**核心分布：**

$$\pi_\theta(a_{t:t+H}, \hat{\ell} | o_t, \ell)$$

**符号说明：**

| 符号                | 含义           | 示例                       |
| ------------------- | -------------- | -------------------------- |
| $\theta$            | 模型参数       | 神经网络权重               |
| $o_t$               | 时刻t的观察    | 图像 + 本体感知            |
| $I_1^t, ..., I_n^t$ | n个相机的图像  | 通常3个视角                |
| $q_t$               | 本体感知状态   | 关节角度、夹爪、底盘速度   |
| $\ell$              | 输入的任务提示 | "clean the bedroom"        |
| $\hat{\ell}$        | 输出的文本     | "pick up the pillow"或问答 |
| $a_{t:t+H}$         | 动作块         | 50步×7维 = 350个数值       |

**公式整体含义：**

对于参数为$\theta$的模型$\pi$，给定当前观察$o_t$和总任务$\ell$作为输入，模型输出动作序列$a_{t:t+H}$和子任务文本$\hat{\ell}$的联合概率分布。

**分布分解（层次化推理的数学基础）：**

$$\pi_\theta(a_{t:t+H}, \hat{\ell} | o_t, \ell) = \pi_\theta(a_{t:t+H} | o_t, \hat{\ell}) \cdot \pi_\theta(\hat{\ell} | o_t, \ell)$$

**公式整体含义：**

给定观察$o_t$和总任务$\ell$，输出动作和子任务的联合分布，可以分解为两步：

1. **第一步（高级推理）**：同一个模型$\pi_\theta$根据观察$o_t$和总任务$\ell$，输出子任务$\hat{\ell}$
2. **第二步（低级推理）**：同一个模型$\pi_\theta$根据观察$o_t$和子任务$\hat{\ell}$（注意不再是$\ell$），输出动作$a_{t:t+H}$

**关键洞察**：动作分布不直接依赖总任务$\ell$，只依赖子任务$\hat{\ell}$！这实现了"思考-执行"的层次化分离。

**两个独立的推理过程：**

1. **高级推理**：$\pi_\theta(\hat{\ell} | o_t, \ell)$
   - 输入：观察 + 总任务
   - 输出：子任务
   - 例如：给定"clean bedroom"，预测"pick up pillow"

2. **低级推理**：$\pi_\theta(a_{t:t+H} | o_t, \hat{\ell})$
   - 输入：观察 + 子任务（注意不是$\ell$！）
   - 输出：动作序列
   - 例如：给定"pick up pillow"，生成具体关节角度

#### Transformer的输入输出结构

**函数定义：**

$$y_{1:N'} = f(x_{1:N}, A(x_{1:N}), \rho(x_{1:N}))$$

其中：

- $x_{1:N}$：N个输入tokens（多模态）
- $A(x_{1:N}) \in [0, 1]^{N \times N}$：Attention矩阵（控制Token可见性）
- $\rho(x_{1:N})$：Token类型指示器
- $y_{1:N'}$：N'个输出tokens

**Token类型：**

每个$x_i$可以是：

| 类型          | 符号                                         | 维度   | 说明           |
| ------------- | -------------------------------------------- | ------ | -------------- |
| 文本token     | $x_i^w \in \mathbb{N}$                       | 离散ID | 词汇表中的索引 |
| 图像patch     | $x_i^I \in \mathbb{R}^{p \times p \times 3}$ | 连续值 | 通常16×16×3    |
| 动作token(FM) | $x_i^a \in \mathbb{R}^d$                     | 连续值 | d维动作向量    |

#### Token处理的两个阶段

##### 阶段1：编码器（Encoder）- 统一格式

| 输入类型    | 原始格式          | 编码器                  | 输出      |
| ----------- | ----------------- | ----------------------- | --------- |
| 图像patches | 16×16×3像素       | Vision Encoder (SigLIP) | 768维向量 |
| 文本tokens  | token ID (整数)   | Embedding Matrix        | 768维向量 |
| 本体感知    | [θ₁, θ₂, ..., θ₇] | 离散化→Embedding        | 768维向量 |
| FAST tokens | token ID (整数)   | Embedding Matrix        | 768维向量 |
| 动作(FM)    | [7, 50]矩阵       | Linear Projection       | 768维向量 |

**目标**：将所有不同格式统一到相同的向量空间（768维）

##### 阶段2：Transformer层 - 专家分工

根据$\rho(x_i)$指示的token类型，使用不同的专家权重：

```text
主Transformer (W_main, 2B参数):
  - 处理：图像、文本、本体感知、FAST tokens
  - 负责：语义理解、文本生成

Action Expert (W_action, 300M参数):
  - 处理：动作tokens(FM)
  - 负责：Flow Matching动作生成
  - 可以看到主Transformer的输出（获取语义信息）
```

**类比Mixture of Experts**：不同token由不同"专家"处理，但共享某些信息。

#### 输出结构

$$y_{1:N'} = [y_{1:M}^\ell, y_{1:H}^a]$$

**两部分输出：**

1. **文本token logits** $y_{1:M}^\ell$：
   - M个logits向量
   - 用于采样$\hat{\ell}$（子任务或答案）
   - 通过softmax + 采样生成文本

2. **动作output tokens** $y_{1:H}^a$：
   - H个tokens（H=50，对应50步动作）
   - 由Action Expert产生
   - 通过linear projection投影到动作空间（7维）

**注意**：$M + H \leq N$

不是所有输入tokens都有对应的loss。例如：

- 图像tokens：只是context，没有loss
- 本体tokens：只是context，没有loss
- 只有预测的tokens才计算loss

#### 本体感知的特殊处理

**为什么离散化？**

```python
# 原始本体感知（连续值）
proprioception = {
    'joint_angles': [0.523, 1.047, ...],  # 弧度
    'gripper': 0.8,  # 0-1
}

# 离散化成文本
discretized = "joint1:30deg joint2:60deg gripper:80%"

# 转成token IDs，像文本一样处理
tokens = tokenize(discretized)
embeddings = embedding_matrix(tokens)
```

**优势**：

- 统一处理：和文本tokens相同
- 简单实现：不需要专门的encoder
- 足够精度：本体感知不需要极高精度

**代码实现**：

**源代码位置**：[`tokenizer.py` - `PaligemmaTokenizer.tokenize()` 方法（L22-L48）](../Repo/openpi/src/openpi/models/tokenizer.py#L22-L48)

本体感知离散化的核心代码：

```python
def tokenize(self, prompt, state=None):
    """将prompt和state转换为tokens"""
    if state is not None:
        # π0.5格式：state离散化为256个bins
        discretized_state = np.digitize(
            state, 
            bins=np.linspace(-1, 1, 256 + 1)[:-1]  # 创建256个bins
        ) - 1  # 结果范围：[0, 255]
        
        # 转换为文本字符串
        state_str = " ".join(map(str, discretized_state))
        
        # 拼接到prompt中
        full_prompt = f"Task: {prompt}, State: {state_str};\nAction: "
        tokens = self.tokenizer.encode(full_prompt, add_bos=True)
    else:
        # π0格式：state单独处理（通过state_proj）
        tokens = self.tokenizer.encode(prompt, add_bos=True)
    
    return tokens, mask
```

**离散化细节**：

- 假设state已经归一化到[-1, 1]范围
- 将连续值映射到256个离散bins（0-255）
- 例如：`[0.523, -0.234, 0.891]` → `[164, 98, 242]` → `"164 98 242"`
- 最终作为文本tokens输入到VLM中

#### Attention矩阵的作用

$A(x_{1:N}) \in [0, 1]^{N \times N}$控制token之间的可见性：

- $A[i][j] = 1$ (或0)：token i可以attend到token j
- $A[i][j] = -\infty$：token i看不到token j（softmax后变0）

**与自注意力机制的关系：**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + A}{\sqrt{d}}\right) V$$

Attention矩阵$A$作为mask，控制信息流动。

**双向attention vs 因果attention：**

- **标准LLM**：因果attention（只能看到前面的tokens）
- **π0.5**：图像、文本、动作tokens使用双向attention（互相可见）

**代码实现**：

**源代码位置**：[`pi0_pytorch.py` - `make_att_2d_masks()` 函数（L52-L81）](../Repo/openpi/src/openpi/models_pytorch/pi0_pytorch.py#L52-L81)

Attention Mask构建的核心代码：

```python
def make_att_2d_masks(pad_masks, att_masks):
    """构建2D attention mask矩阵，控制token之间的可见性
    
    att_masks示例：
    - [0, 0, 0, 1, 1, 1]: prefix-LM（前3个双向，后3个因果）
    - [1, 0, 0, 0, 1, 0, ...]: 区分不同block的attention
    """
    # 计算累积mask，相同累积值的tokens可以互相看到
    cumsum = torch.cumsum(att_masks, dim=1)  # [B, N]
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]  # [B, N, N]
    
    # 结合padding mask（忽略padding的tokens）
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks  # [B, N, N]
```

在π0.5中的使用（控制FM和FAST不互相可见）：

```python
# 在embed_suffix()中设置attention masks
att_masks += [1] + ([0] * (action_horizon - 1))  # 第1个action token为1，其余为0
# 结果：所有action tokens有相同的cumsum值，可以双向attend
```

---

### B. Combining Discrete & Continuous Action Representations（动作表示结合）

#### 问题：两种动作表示的权衡

**FAST（离散tokens）的特点：**

| 优势                             | 劣势                          |
| -------------------------------- | ----------------------------- |
| 训练快速（8个tokens，简单）      | 推理慢（自回归，8次顺序调用） |
| 和文本统一处理                   | 有量化误差（离散化损失精度）  |
| Next-token prediction，稳定      | 不适合实时控制（40ms太慢）    |
| 序列短（50步→8tokens，压缩44倍） | 每个token必须等前面的         |

**Flow Matching（连续向量）的特点：**

| 优势                         | 劣势                         |
| ---------------------------- | ---------------------------- |
| 推理快（10步迭代，15ms）     | 训练慢（需要采样τ和ω）       |
| 精度高（连续值，无量化误差） | 训练复杂（MSE loss，速度场） |
| 适合实时控制（50Hz）         | 序列长（50个时间步tokens）   |
| 50步并行处理                 | 需要更多计算资源             |

**类比理解：**

- **FAST = JPG逐列加载**：必须从左到右一列列出现，等前面的列
- **FM = 渐进式JPEG**：整张图先模糊出现，再逐渐清晰，所有像素同时存在

#### π0.5的解决方案：训练时结合，推理时选择

**核心策略：**

```text
Pre-training (α=0):
  只用FAST → 快速建立基础能力
  
Post-training (α>0):
  FAST + FM 同时训练 → 两者优势结合
  关键：FM不能看FAST（Attention Mask）
  
Inference:
  只用FM → 精确实时控制
```

**Attention Mask的关键作用：**

通过attention矩阵确保FM动作tokens和FAST动作tokens不互相attend：

```text
FM动作tokens:
  可以看：图像、文本、本体感知
  不能看：FAST tokens
  
原因避免信息泄露：
  如果FM能看到FAST的离散编码，可能直接解码FAST，不学真正的Flow Matching，失去FM的精确性和实时性
```

#### 训练目标：混合Loss函数

**数学定义：**

$$\mathbb{E}_{D,\tau,\omega}\left[H(x_{1:M}, f_\theta^\ell(o_t, \ell)) + \alpha\left\|\omega - a_{t:t+H} - f_\theta^a(a_{t:t+H}^{\tau,\omega}, o_t, \ell)\right\|^2\right]$$

**分解理解：**

##### Loss 1：文本预测（包括FAST）

$$L_{\text{text}} = H(x_{1:M}, f_\theta^\ell(o_t, \ell))$$

- $H$：交叉熵（Cross Entropy）
- $x_{1:M}$：真实的M个文本tokens（包括FAST tokens）
- $f_\theta^\ell$：主Transformer的输出。包含子任务预测、FAST动作预测、问答等。

**作用说明：**

这是标准的Next-Token Prediction损失，用于训练主Transformer的所有文本生成能力。具体包括：

1. **子任务预测**："pick up the pillow" → 高级语义理解
2. **FAST动作生成**：8个离散tokens → 基础动作生成能力（防止遗忘）
3. **问答和描述**："What is this?" → "A plate" → 通用语言能力

在Pre-training阶段，这是唯一的loss（α=0）。在Post-training阶段，这个loss继续存在，确保模型不会因为专注FM而遗忘文本和FAST能力。

##### Loss 2：Flow Matching

$$L_{\text{FM}} = \left\|\omega - a_{t:t+H} - f_\theta^a(a_{t:t+H}^{\tau,\omega}, o_t, \ell)\right\|^2$$

- $\|\cdot\|^2$：L2范数（均方误差，MSE）
- $a_{t:t+H}$：真实动作（ground truth）
- $\omega$：采样的噪声，$\omega \sim \mathcal{N}(0, I)$
- $a_{t:t+H}^{\tau,\omega} = \tau \cdot a_{t:t+H} + (1-\tau) \cdot \omega$：插值（部分去噪的动作）
- $\omega - a_{t:t+H}$：理想速度场（从噪声指向真实动作的方向）
- $f_\theta^a$：Action Expert的输出（预测的速度场）

**作用说明：**

这是Flow Matching的速度场回归损失，用于训练Action Expert生成精确的连续动作。训练过程：

1. **随机采样时间步**：$\tau \sim \text{Uniform}(0, 1)$，例如τ=0.3
2. **随机采样噪声**：$\omega \sim \mathcal{N}(0, I)$，纯高斯噪声
3. **构造插值**：$x_\tau = 0.3 \times a + 0.7 \times \omega$（30%真实动作 + 70%噪声）
4. **计算理想速度场**：$v^* = a - \omega$（从噪声到真实动作的方向）
5. **Action Expert预测**：$v_\theta = f_\theta^a(x_\tau, \tau, \text{语义特征})$
6. **最小化误差**：$\|v_\theta - v^*\|^2$

通过在不同τ时刻的训练，Action Expert学会从任意"部分去噪"的状态预测正确的前进方向，最终实现从纯噪声生成清晰动作的能力。

只在Post-training阶段使用（α>0），专门优化精确动作生成。

**总Loss：**

$$L_{\text{total}} = L_{\text{text}} + \alpha \cdot L_{\text{FM}}$$

**α的作用（超参数）：**

| α值   | 含义            | 使用场景          |
| ----- | --------------- | ----------------- |
| α = 0 | 只关心文本/FAST | Pre-training阶段  |
| α < 1 | 主要关心文本    | 早期Post-training |
| α = 1 | 两者同等重要    | 标准Post-training |
| α > 1 | 更重视FM        | 强调精确动作      |

**为什么用MSE？**

1. **连续值回归**：速度场是连续向量，MSE是标准选择
2. **理论最优**：在高斯假设下，MSE对应最大似然估计
3. **梯度友好**：MSE的梯度简单，训练稳定

**代码实现**：

**源代码位置**：[`pi0_pytorch.py` - `forward()` 方法（L316-L373）](../Repo/openpi/src/openpi/models_pytorch/pi0_pytorch.py#L316-L373)

Combined Loss的核心实现：

```python
def forward(self, observation, actions, noise=None, time=None):
    # 1. 构造插值（部分去噪的动作）
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions  # 理想速度场
    
    # 2. 编码输入（图像、文本、动作）
    prefix_embs = self.embed_prefix(images, ...)  # VLM处理
    suffix_embs = self.embed_suffix(state, x_t, time)  # Action处理
    
    # 3. 前向传播（主模型 + Action Expert）
    (_, suffix_out), _ = self.paligemma_with_expert.forward(...)
    
    # 4. 预测速度场
    v_t = self.action_out_proj(suffix_out)
    
    # 5. 计算FM loss（文本loss在外部计算）
    return F.mse_loss(u_t, v_t, reduction="none")
```

详细实现请参考：[Flow Matching Notes - CFM训练核心](../Flow-Matching/Flow%20Matching%20Notes.md#1-cfm训练核心forward-方法)

#### FAST vs FM：推理方式的本质区别

**动作矩阵结构（推理的最终目标）：**

| 动作参数  | 步骤1 | 步骤2 | 步骤3 | ... | 步骤50 |
| --------- | ----- | ----- | ----- | --- | ------ |
| 关节1角度 | 0.52  | 0.54  | 0.56  | ... | 0.78   |
| 关节2角度 | 1.05  | 1.08  | 1.12  | ... | 1.45   |
| 关节3角度 | -0.31 | -0.29 | -0.26 | ... | 0.02   |
| 关节4角度 | 2.14  | 2.16  | 2.18  | ... | 2.35   |
| 关节5角度 | 0.89  | 0.91  | 0.93  | ... | 1.12   |
| 关节6角度 | -1.24 | -1.22 | -1.19 | ... | -0.95  |
| 夹爪开合  | 0.0   | 0.0   | 0.1   | ... | 0.8    |

矩阵维度：**[n=7个动作参数, H=50个时间步]**

---

**FAST推理方式（逐列生成，自回归）：**

**生成过程：**

```text
初始状态：整个矩阵都是未知
[?, ?, ?, ..., ?]
[?, ?, ?, ..., ?]
...

第1次调用：生成第1列（步骤1）
[✓, ?, ?, ..., ?]  ← 推理出来，最终值
[✓, ?, ?, ..., ?]
...

第2次调用：基于第1列生成第2列（步骤2）
[✓, ✓, ?, ..., ?]  ← 推理出来，最终值
[✓, ✓, ?, ..., ?]
...

第3次调用：基于第1-2列生成第3列（步骤3）
[✓, ✓, ✓, ..., ?]
...

...依次类推...

第50次调用：基于第1-49列生成第50列
[✓, ✓, ✓, ..., ✓]  ← 完成！
```

**关键特性：**

1. **推理次数**：50次transformer调用（每列一次）
2. **依赖关系**：**自回归** - 每列依赖前面所有列，必须顺序执行
3. **输出性质**：**每列推理一次就是最终结果**，不需要迭代
4. **模型使用**：2B主Transformer（模型较大）
5. **类比**：像JPG图片从左到右一列列加载，必须等前面的列

---

**FM推理方式（整体细化，并行）：**

**生成过程：**

```text
初始状态：整个矩阵填充随机噪声
[噪声, 噪声, 噪声, ..., 噪声]
[噪声, 噪声, 噪声, ..., 噪声]
...（所有50列同时存在，但都是噪声）

第1次迭代（τ=0）：整个矩阵一起去噪10%
[模糊, 模糊, 模糊, ..., 模糊]  ← 所有列同时处理
[模糊, 模糊, 模糊, ..., 模糊]  ← 还不是最终值
...（所有50列都存在，都模糊）

第2次迭代（τ=0.1）：整个矩阵再去噪10%
[稍清晰, 稍清晰, 稍清晰, ..., 稍清晰]
...（所有50列都在变清晰）

第5次迭代（τ=0.4）：整个矩阵半清晰
[半清, 半清, 半清, ..., 半清]
...

第10次迭代（τ=0.9）：整个矩阵完全清晰
[清晰, 清晰, 清晰, ..., 清晰]  ← 最终结果！
[清晰, 清晰, 清晰, ..., 清晰]
...（所有50列都清晰了）
```

**关键特性：**

1. **推理次数**：10次Action Expert调用（整个矩阵迭代10次）
2. **依赖关系**：**无列间依赖** - 50列在同一次前向传播中并行处理
3. **输出性质**：**每次都不是最终结果**，需要迭代10次才完全清晰
4. **模型使用**：300M Action Expert（模型较小）
5. **类比**：像渐进式JPEG，整张图先模糊出现，再逐渐清晰

---

**"并行"的精确含义：**

不是说10次迭代同时进行，而是：

- **每次迭代时**：H=50个时间步（50列）**同时输入**Action Expert
- **一次forward pass**：处理整个[7, 50]动作矩阵
- **不需要等待**：不是"推理第2列要等第1列"，而是"50列一起变清晰"

---

**效率对比：**

| 维度     | FAST                | FM                    |
| -------- | ------------------- | --------------------- |
| 调用次数 | 50次                | 10次                  |
| 模型大小 | 2B（主Transformer） | 300M（Action Expert） |
| 并行性   | 串行（必须逐列）    | 并行（50列一起）      |
| 输出性质 | 一次到位            | 需要迭代              |

**为什么FM更快？**

1. **模型更小**：300M vs 2B，单次前向传播更快
2. **调用更少**：10次 vs 50次
3. **列并行**：50列同时处理，不需要等待

#### FAST+FM结合机制总结

**训练策略：**

- Pre-training：只用FAST（α=0），快速学习
- Post-training：FAST+FM同时训练（α>0），结合优势
- Attention Mask：FM不能看FAST，避免作弊

**推理策略：**

- 文本生成：自回归（预测子任务）
- 动作生成：只用FM（10步迭代，实时控制）
- FAST退休：训练时保留只为防止遗忘

**Loss函数：**

- $L = L_{\text{text}} + \alpha \cdot L_{\text{FM}}$
- 主Transformer受两个loss影响
- Action Expert只受FM loss影响

---

### C. Pre-training（预训练阶段）

#### 训练配置

- **训练步数**：280k gradient steps
- **动作表示**：仅使用FAST（离散token）
- **Loss权重**：α = 0（不使用Flow Matching）
- **目标**：在多样化数据上学习通用的视觉-语言-动作能力

#### 数据配方概览

**关键统计：**

- **总数据中移动操作数据占比**：仅2.4%
- **非移动操作数据占比**：97.6%（其他机器人+网络数据等）
- **数据源数量**：6种异构数据源
- **环境覆盖**：约100个不同的家庭环境（仅MM数据）

**数据源总览：**

| 数据类型 | 全称               | Pre-training | Post-training |
| -------- | ------------------ | ------------ | ------------- |
| **MM**   | Mobile Manipulator | ✅            | ✅（筛选）     |
| **ME**   | Multi-Environment  | ✅            | ✅（筛选）     |
| **CE**   | Cross-Embodiment   | ✅            | ❌             |
| **HL**   | High-Level subtask | ✅            | ✅（部分）     |
| **WD**   | Web Data           | ✅            | ✅             |
| **VI**   | Verbal Instruction | ❌            | ✅             |

![alt text](<Pi0.5 Paper Notes.assets/image-6.png>)

#### 各数据源详细说明

##### 1. MM（Mobile Manipulator）- 移动操作机器人数据

**数据量：**

- **约400小时**机器人操作数据
- **约100个**不同的家庭环境

**数据内容：**

- **任务类型**：家务任务（清洁、整理、收纳等）
- **机器人平台**：Section IV-E描述的双臂移动操作机器人
- **环境类型**：真实家庭（厨房、卧室等）

**重要性：**

- 与评估任务最直接相关
- 提供移动操作的核心能力（导航+操作）
- 但仅占总训练数据的**2.4%**！

##### 2. ME（Multi-Environment）- 多环境非移动机器人数据

**机器人配置：**

- **类型**：单臂或双臂固定机械臂
- **安装方式**：固定在表面或安装平台上
- **优势**：比移动机器人更轻便，易于运输和部署

**数据特点：**

- **环境多样性**：能在更广泛的家庭环境中收集数据
- **Embodiment差异**：与移动机器人不同（无移动底座）
- **作用**：增强环境泛化能力，尤其是操作技能的迁移

##### 3. CE（Cross-Embodiment）- 跨平台实验室数据

**任务范围：**

- 桌面环境中的多种任务：
  - 清理餐桌（bussing table）
  - 叠衣服（folding shirts）
  - 放置餐具（placing utensils）
  - 研磨咖啡豆（grinding coffee beans）
  - 等等

**机器人类型：**

- 单臂和双臂操作机器人
- 静态底座和移动底座

**数据来源：**

- **自采集实验室数据**
- **OXE开源数据集**（Open X-Embodiment）
  - OXE是π0使用数据集的扩展版本
  - 包含大量跨机器人平台的操作数据

**任务相关性：**

- **高度相关**：例如放餐具（与评估任务类似）
- **不相关**：例如研磨咖啡豆（但提供通用操作模式）

##### 4. HL（High-Level subtask）- 高层子任务数据

**数据构建方式：**

- **标注方式**：**手动标注**MM、ME、CE中所有涉及多子任务的数据
- **标注内容**：
  - 子任务的语义描述（文本）
  - 相关边界框（bounding boxes）

**示例：**

- 总任务："clean the bedroom"
- 子任务标注：
  - "adjust the blanket" + 边界框
  - "pick up pillow" + 边界框
  - ...

**训练目标：**

1. **先预测边界框**（物体定位）
2. **再预测子任务标签**（文本）
3. **最后预测动作**（条件在子任务标签上）

**类比：**

- 类似语言模型的**chain-of-thought prompting**
- 让模型学会"先想后做"

**效果：**

- 模型同时具备：
  - **高层策略**：输出子任务（ℓ̂）
  - **低层策略**：执行动作（a_{t:t+H}）

##### 5. WD（Web Data）- 网络视觉-语言数据

**具体数据集：**

| 任务类型 | 数据集                                              |
| -------- | --------------------------------------------------- |
| 图像标注 | CapsFusion, COCO                                    |
| 视觉问答 | Cambrian-7M, PixMo, VQAv2                           |
| 物体定位 | 标准数据集 + **额外的室内场景和家居物品边界框数据** |

**作用：**

- 提供丰富的**视觉-语言语义知识**
- 帮助模型理解场景中的物体（"这是什么？"）
- 增强对新物体的泛化能力

**示例任务：**

- "What kind of pie is on this plate?" → "Chocolate"
- "Detect and label all objects in the scene." → `<loc0112><loc0234>...`

##### 6. VI（Verbal Instruction）- 口头指令数据

**重要提示：**

- **VI仅在Post-training阶段使用**
- **Pre-training不包含VI**

#### 动作数据的技术处理

##### 控制模式标识

- **方法**：在文本提示中添加特殊token
- **格式**：`<control mode> joint/end effector <control mode>`
- **目的**：区分关节控制和末端执行器控制

##### 动作归一化

- **归一化范围**：[-1, 1]
- **归一化方法**：使用每个动作维度的**1%和99%分位数**
  - 不用最小值/最大值，避免极端值影响
  - 例如：如果某维度99%的值在[-2, 3]，则用这个范围归一化
- **目的**：防止极端值影响训练稳定性

##### 动作维度对齐

- **问题**：不同机器人的动作空间维度不同
  - 移动机器人：18-19维（双臂+夹爪+底座+躯干）
  - 单臂固定机器人：可能只有7-8维（一个臂+夹爪）
- **解决方案**：设置统一的固定维度（取最大）
- **低维机器人处理**：用**零填充（zero-pad）**动作向量
  - 例如：7维机器人 → 填充到19维，后12维全为0
- **目的**：支持不同embodiment的机器人共同训练

#### Pre-training数据策略总结

**异构数据共训练的威力：**

- **多样性**：
  - 6种数据源
  - 多种机器人平台（移动、固定、单臂、双臂）
  - 多种环境（约100个家庭+实验室）
  - 多种任务（家务、桌面操作、视觉问答等）

- **互补性**：
  - **MM（2.4%）**：目标任务的直接经验
  - **ME**：更广泛环境中的操作技能
  - **CE+OXE**：跨机器人的通用操作模式
  - **HL**：任务分解和长期规划能力
  - **WD（97.6%的一部分）**：丰富的视觉-语言语义知识

- **统一性**：
  - **FAST编码**：将所有动作编码为离散token
  - **Sequence Modeling**：统一为token序列预测问题
  - **单一Transformer**：一个模型处理所有模态

**关键洞察：**

> 97.6%的训练数据不是目标任务（移动操作），但通过知识迁移，模型仍能在全新家庭环境中执行复杂的长时域任务！

---

### D. Post-training（后训练阶段）

#### Post-training配置

- **训练步数**：80k gradient steps（远少于Pre-training的280k）
- **动作表示**：FAST + Flow Matching同时训练
- **Loss权重**：α = 10.0（FM权重是文本的10倍，强调连续动作质量）
- **模块变化**：新增Action Expert（从随机初始化开始）
- **目标**：专门化到家庭移动操作 + 学习连续动作生成

#### 数据筛选策略

**保留的数据：**

| 数据类型           | Pre-training | Post-training | 筛选标准               |
| ------------------ | ------------ | ------------- | ---------------------- |
| MM（移动操作）     | ✅            | ✅             | 只保留成功+短片段      |
| ME（多环境）       | ✅            | ✅             | 只保留成功+短片段      |
| HL（高层子任务）   | ✅            | ✅             | 只保留多环境相关的切片 |
| WD（网络数据）     | ✅            | ✅             | 防止视觉/语义退化      |
| **VI（口头指令）** | ❌            | ✅ **新增**    | 强化层次化推理         |

**移除的数据：**

- **CE（实验室跨平台）**：与真实家庭场景差异大，不利于专门化

**为什么筛选"成功+短片段"？**

- **成功**：后训练是精调，要学习正确行为模式
- **短片段**：长轨迹可能包含低效行为，短轨迹更高效

#### 口头指令数据（VI）的特殊性

**不是传统的遥操作数据，而是"语言分解示范"：**

1. 专家用户发出高层指令："pick up the plate"
2. 已训练的低层策略执行动作（机器人自己抓）
3. 记录配对：`(观察, "pick up the plate", 动作序列)`

**作用：**

- 教会模型如何将复杂任务分解为合理的子任务序列
- 例如："清理桌子" → "pick plate" → "put in sink" → "pick cup" → ...
- 这是**层次化推理**的关键训练信号

---

### E. Robot system details（机器人系统细节）

#### 硬件配置

**实验平台：**两种移动操作机器人，配置相同

| 部件         | 规格        | 说明                          |
| ------------ | ----------- | ----------------------------- |
| **双臂**     | 2 × 6 DoF   | 每个手臂6个关节               |
| **夹爪**     | 平行爪      | 类似"拇指+食指"，抓取日常物品 |
| **腕部相机** | 2 × 单目RGB | 每个手臂腕部一个，近距离观察  |
| **前后相机** | 2 × 单目RGB | 安装在手臂之间，全局视野      |
| **移动底座** | 全向轮式    | 可前后左右任意方向移动        |
| **躯干升降** | 1-2 DoF     | 调整高度，适应不同操作高度    |

#### 状态和动作空间

##### 总维度：18-19维

| 组件         | 维度 | 具体内容                   |
| ------------ | ---- | -------------------------- |
| **左臂**     | 6    | 6个关节的目标角度          |
| **右臂**     | 6    | 6个关节的目标角度          |
| **左夹爪**   | 1    | 开合程度                   |
| **右夹爪**   | 1    | 开合程度                   |
| **底座**     | 3    | x速度 + y速度 + 旋转速度   |
| **躯干升降** | 1-2  | 1D：上/下；2D：上/下+前/后 |

**相机使用策略：**

- **高层推理**：4个相机（前+后+左腕+右腕）→ 全局理解
- **低层推理**：3个相机（前+左腕+右腕）→ 聚焦操作

#### 控制系统：极简端到端

**架构：**

```text
π0.5模型（50Hz）
    ↓ 输出: [50步] × [18-19维] 动作矩阵
    ↓
PD控制器（简单跟踪）
    ↓
机器人硬件执行
```

**关键特点：**

- **50Hz控制频率**：每秒50次指令
- **Action Chunking**：一次生成50步未来动作
- **无轨迹规划**：不提前计算最优路径
- **无碰撞检测**：完全靠模型学习避障
- **端到端控制**：所有操作和导航控制都由神经网络学习

**方法简单但有效：**

- 传统方法：SLAM建图 + 运动规划 + 碰撞检测
- π0.5：在大量数据中隐式学习了这些能力

---

## V. EXPERIMENTAL EVALUATION（实验评估）

### 评估设计理念

π0.5的实验评估与传统VLA有根本性的不同。传统做法通常在训练集环境或相似环境中测试模型，而π0.5的所有实验都在训练中从未见过的全新环境中进行。这种设计选择反映了作者对真实世界泛化能力的重视。

评估采用两层体系：首先在Mock环境（模拟家庭）中进行定量对比，这些环境可控且可重复，适合进行精确的性能测量和对比实验；其次在3个真实家庭中进行最终评估，这代表了最真实的挑战，也是模型实际部署能力的直接证明。

实验围绕五个核心问题展开：

1. π0.5能否在全新家庭中泛化到复杂多阶段任务？
2. 泛化能力如何随训练数据中的环境数量扩展？
3. Co-training中各个数据源如何贡献最终性能？
4. π0.5与π0 VLA相比表现如何？
5. 高层推理有多重要？

---

### A. Can π0.5 generalize to real homes?（真实家庭泛化能力）

实验在训练集中从未出现的真实家庭中进行，执行厨房和卧室清洁任务。模型展现出强大的自主分解能力，能够将高层指令（如"put the items in the drawer"）分解为完整的子任务序列。实验结果显示π0.5能够稳定完成各种多阶段任务，处理完全新颖的物体、布局和配置，展现出真正的in-the-wild泛化能力。

这一结果在多个维度上超越了先前的VLA模型：先前工作通常在接近训练环境的场景中测试，而π0.5面对全新家庭；先前工作主要处理单步短时操作，而π0.5可以执行长时域多阶段序列。这是首次展示端到端学习的机器人系统能在全新家庭中执行复杂的长时域操作任务。

---

### B. How does generalization scale with the number of scenes?（泛化能力的环境扩展规律）

实验通过变化训练环境数量来研究泛化能力的扩展规律。所有模型在Pre-training阶段使用相同配置，仅在Post-training阶段改变移动操作数据的环境数量。评估包括整体任务性能和语言遵循能力两类测试。

#### 核心发现

**对数线性扩展：** 性能随环境数量呈对数线性增长（$$\text{性能} \propto \log(\text{环境数量})$$），早期增加环境带来显著提升，后期提升逐渐变缓。这启示未来研究应优先追求环境多样性而非单一环境的数据量。

**Co-training不可替代：** 关键发现是即使直接在测试环境训练，如果没有完整的Co-training配方，性能仍然很差。这证明问题的关键不在于数据量，而在于数据多样性和知识迁移。跨机器人数据、Web数据、高层标注提供的知识无法被更多的移动操作数据替代。

**类外泛化：** 语言遵循实验显示，环境多样性不仅提升In-distribution物体的识别（快速），也提升Out-of-distribution物体的识别（较慢但持续）。环境多样性强迫模型学习通用操作策略，而非记忆特定物体，最终使模型能够处理训练中完全没有的物体类别。

---

### C. How important is each part of our co-training recipe?（Co-training配方各部分的重要性）

消融实验通过逐个移除数据源来验证每种数据源的贡献。实验设计了no WD、no ME、no CE、no ME & CE四种变体，在整体任务性能和语言遵循能力两个维度进行评估。

#### 实验结论

**ME和CE不可或缺：** 移除任何一个都会显著降低整体任务性能，同时移除两者情况更糟，展现出明显的累加效应。跨机器人迁移是核心能力来源：ME提供多样化环境中的操作技能，CE提供实验室中的多样化任务经验。

**WD的双重作用：** 在整体任务上，移除WD的影响不显著（这些任务主要测试操作技能）；但在语言遵循和OOD物体识别上，WD的影响巨大。Web数据覆盖了广泛的物理对象，使模型能够理解并遵循涉及未见物体类别的语言命令，支持zero-shot识别能力。

**数据源的互补性：** 不同数据源在不同层面发挥作用：ME和CE提供操作技能迁移，HL提供任务分解能力，WD提供语义泛化能力。它们缺一不可，共同构成模型的完整能力，实现真正的in-the-wild泛化。

---

### D. How does π0.5 compare to other VLAs?（与其他VLA的对比）

实验将π0.5与两个基线进行对比：原始π0（始终使用Flow Matching）和改进版π0-FAST+Flow（采用与π0.5相同的混合训练方式，但只用机器人动作数据，不含HL和WD）。这个对比设计使得π0-FAST+Flow已经尽可能接近π0.5，唯一的区别就是是否使用co-training数据。

为确保公平比较，所有模型接收相同的跨平台机器人训练集，并训练相当数量的步数。实验结果显示π0.5显著优于两个基线。更重要的是，即使让π0训练更长时间，π0.5仍然表现更好，这证明优势来源于co-training和混合训练方式，而非训练时长。这一结果也确认了使用FAST token训练在计算效率上优于纯基于扩散的训练。

---

### E. How important is high-level inference?（高层推理的重要性）

实验评估了高层推理的重要性，设计了七种变体进行对比。所有变体都使用完整的π0.5低层推理过程，区别在于高层策略的不同：完整π0.5、no WD、no VI、implicit HL（训练时有HL数据但推理时不用）、no HL（训练和推理都没有）、GPT-4作为高层策略、以及人类专家作为oracle上界。

#### 实验结果

实验揭示了几个重要发现。完整π0.5表现最好，甚至超越了人类专家这一"oracle"基线，说明模型学到的高层策略比人类的即时判断还要有效。令人意外的是，第二好的模型是implicit HL：虽然推理时不做显式的高层推理，但因为训练时包含HL数据，模型已经隐式地学会了任务分解。这强烈表明co-training配方的重要性——HL数据的价值不仅在于推理时的显式分解，更在于训练时教会模型如何思考任务。

no HL变体（训练和推理都没有HL）表现显著更差，证明HL数据至关重要。no VI变体也明显下降，虽然VI数据仅占高层示例的11%，但其影响巨大，因为它提供了人类专家如何分解任务的高质量示范。no WD变体同样表现很差，说明Web数据不仅帮助语义理解，也改善高层推理能力。

最差的是GPT-4变体，这表明通用大模型无法直接用于机器人控制，必须在机器人数据上训练才能获得有效的高层策略。性能排序为：π0.5（完整） > 人类专家 > implicit HL > no WD > no VI > no HL > GPT-4。

---

## VI. DISCUSSION AND FUTURE WORK（讨论与未来工作）

### 核心贡献总结

π0.5是一个基于π0构建的co-training模型，通过整合多种数据源实现了对新环境的强大泛化能力。模型能够在训练数据中从未见过的家庭中控制移动操作机器人执行复杂的多阶段任务，如清洁厨房和卧室、整理床铺、挂毛巾等灵巧行为。

关键创新在于数据配方：虽然只使用了约400小时的移动操作数据，但通过co-training整合了来自其他机器人的大量数据（多样化环境中的非移动操作机器人、实验室条件下的跨平台数据）、Web数据以及高层预测数据。这种co-training配方促进了有效的知识迁移，证明了仅用中等规模的目标任务数据集就能实现高度泛化的控制能力。

### 局限性与挑战

π0.5并非完美，仍然存在三类主要挑战。物理层面上，某些环境呈现持续的困难，如不熟悉的抽屉把手或物理上难以打开的柜子。感知层面上，部分可观察性带来挑战，例如机械臂遮挡应该擦拭的溢出物。决策层面上，高层子任务推理有时容易分心，例如在放置物品时重复开关抽屉。应对这些挑战需要更好的co-training策略、迁移学习方法以及更大规模的数据集。

技术限制方面，模型目前只能处理相对简单的提示，复杂的用户偏好和指令需要通过更丰富多样的标注数据来支持。模型使用的上下文窗口相对有限，整合更丰富的上下文和记忆机制可以显著提升在高度部分可观察环境中的能力，如跨房间导航或记住物体存储位置等任务。

### 未来方向

π0.5探索了一种特定的异构数据源组合，但数据源的选择可以更广泛地探索。口头指令（VI）展现出强大的潜力，作为一种新的监督模态，未来工作可以探索更多人类为机器人提供额外上下文知识的方式，如在线演示、错误纠正、偏好反馈等。作者希望π0.5能够为新一代VLA奠定基础，推动这些模型在多样化的真实世界环境中展现出真正的广泛泛化能力。
