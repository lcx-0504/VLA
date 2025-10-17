# VLA Learning Notes

## 1. Pi0 & Pi0.5

### 1.1 Blog-Pi05: A VLA with Open-World Generalization

[Blog Link (Suggest for reading)](https://www.physicalintelligence.company/blog/pi05)

[Local PDF (Only for backup)](../Paper/Blog-Pi05:%20A%20VLA%20with%20Open-World%20Generalization.pdf)

#### 研究的背景问题是什么？

机器人最大的问题不在于敏捷或者灵活性，而在于泛化。低层次的泛化是识别外观不同的物体，高层次的泛化是理解任务背后的语义和意图。这要求机器人有不仅要感知环境，还要有物理、视觉和语音上的泛化。

当前机器人数据多样性的不足导致了泛化不足的问题，导致现在大多数机器人必须在工厂这类有严格控制的场所应用，而无法在日常生活中大面积运用。

当前大多数VLA（包括pi0）都是在与训练密切相关的特定环境评估的，但是最新的pi0.5表现出了更好的泛化能力。他的目标并不是新技能或者是高度灵活性，而是在于泛化（例如在没训练过的新的家庭环境中进行清理）。

#### 我们如何解决这个问题？

##### 核心原理

**pi0.5的核心原理是在异构数据上进行联合训练。**

**异构数据 (Heterogeneous Data)**: 不同类型、不同来源的数据。不是都来自mobile robot，不是都是action数据，而是包括图片、文字、动作等各种形式。从而可以学到：

1. 物理执行技能（怎么抓、怎么移动、怎么放）
2. 理解语义上下文（“清理厨房”的任务需要收拾什么，往哪里收拾）
3. 推断任务高层结构（”铺床”要拿枕头、整理床单、放回枕头等步骤规划）
4. 从其他机器人迁移（其他机器人的操作数据也有用，跨实体迁移）

**联合训练 (Co-Training)**: 同时在这些不同数据上训练同一个模型。VLA可以联合训练的重要原因是他是从VLM派生的。二者都是sequence modeling，把不同模态都当成token序列处理。

| 模型类型 | 输入类型                  |
| -------- | ------------------------- |
| VLM      | 图像 + 文字               |
| VLA      | 图像 + 文字 + 动作 + 其他 |

##### 具体的训练任务

1. 通用多模态任务（视觉理解和语义知识）
   - Image captioning (图像字幕): 图片 → "a dog playing frisbee"
   - Visual QA (视觉问答): 图片 + "How many desks?" → "12"
   - Object detection (对象检测): 图片 → bounding boxes

2. 机器人导向任务
   - **带动作的演示数据**：输入是图像+语言指令（如"put the cup in the sink"），输出是动作序列[a1, a2, a3, ...]
   - **高层次机器人示例**（重要！）：输入是观察到的场景（如unmade bed的图片），输出是语义层面的标签（如"pick up the pillow"）。注意这里输出的不是底层动作，而是语义指令
   - **语言指导演示**（Verbal Instruction）：人类像教练一样一步步告诉机器人做什么（例如："close the microwave" → 机器人执行 → "now pick up the mitten"），教会模型如何分解复杂任务和理解自然语言指令

##### 分层推理机制

| 推理层级   | 输入                | 输出                         | 类比                 | 从哪学习                    |
| ---------- | ------------------- | ---------------------------- | -------------------- | --------------------------- |
| High-Level | 当前场景 + 总任务   | 下一个语义步骤               | Chain-of-thought推理 | Web数据、语言指令、高层标注 |
| Low-Level  | 语义步骤 + 当前观察 | 电机命令（关节角度、速度等） | 具体动作执行         | 机器人动作数据              |

**例子**：High-level推理 "clean the kitchen" → "pick up the plate"，Low-level推理 "pick up the plate" → [θ1=0.5, θ2=-0.3, θ3=1.2, ...]

**分层的好处**：高层和低层可以从不同数据源学习，高层受益于语义知识（web数据），低层受益于物理技能（机器人数据），并且可以跨机器人平台迁移。

#### 实验设计与结果

##### 消融实验设计

训练VLA需要"正确的数据混合配方"，就像教人新工作需要合适的课程安排（conceptual + practical）。实验设计了5个版本，每个版本去掉一部分数据，验证各种数据的作用。

**数据来源**：

| 数据类型                 | 缩写 | 具体内容                           | 数据量    | 教会模型什么                 |
| ------------------------ | ---- | ---------------------------------- | --------- | ---------------------------- |
| **Web Data**             | WD   | 图像字幕、视觉问答、物体检测       | 大量      | 视觉语义理解、物体识别、常识 |
| **Multiple Environment** | ME   | 静态机器人在多个不同家庭的数据     | 中量      | 环境多样性、不同场景的物体   |
| **Cross Embodiment**     | CE   | π0原始数据集（不同类型的机器人）   | 中量      | 物理技能的跨平台迁移         |
| **Mobile Manipulation**  | -    | 目标平台的直接数据（mobile robot） | 约400小时 | 特定任务的直接经验           |

**实验模型版本**：

| 版本              | 去掉的数据         | 保留的数据            | 验证什么               |
| ----------------- | ------------------ | --------------------- | ---------------------- |
| **π0.5 (完整版)** | 无                 | WD + ME + CE + Mobile | 完整性能基准           |
| **no WD**         | Web数据            | ME + CE + Mobile      | Web数据是否必要？      |
| **no ME**         | 多环境静态机器人   | WD + CE + Mobile      | 环境多样性是否重要？   |
| **no CE**         | 跨实体数据(π0)     | WD + ME + Mobile      | 跨机器人迁移是否有效？ |
| **no ME or CE**   | 所有其他机器人数据 | WD + Mobile (仅400h)  | 只用直接数据够不够？   |

##### 评估方法

实验设计了两种评估条件：

- **Full Cleaning Tasks**（完整清理任务）：把盘子放进水槽、清理卧室地板等复杂长时的真实场景任务
- **OOD Evaluation**（分布外评估）：把指定物体放进抽屉，难点在于这些物体可能是训练时从未见过的新类别

对于这两种评估，研究者使用了两个指标来衡量性能。**Success Rate**（成功率）衡量子任务的平均完成百分比，例如需要移动5个物体成功移动4个就是80%。**Language Following Rate**（语言遵循率）则衡量机器人行为与用户指令的相符程度。两个指标的区别在于：机器人可能做了某些事情但做错了方向，此时成功率低但可能部分遵循了指令。

##### 实验结果分析

![alt text](<VLA Learning Notes.assets/image.png>)

| 场景                | 主要发现                 | 说明                           |
| ------------------- | ------------------------ | ------------------------------ |
| **In-Distribution** | 去掉ME或CE后性能下降明显 | 其他机器人数据对基础性能很重要 |
| **OOD (新物体)**    | 去掉WD性能下降最严重     | Web数据帮助识别新物体类别      |
| **所有场景**        | ME和CE都重要             | 物理技能迁移在所有情况下都有用 |

**关键发现**：**Web数据 (WD)** 对OOD场景（新物体）特别重要，原因是教会模型物体的语义（"什么是玩具"、"什么是盘子"），没有WD连新物体都认不出来。**其他机器人数据 (ME + CE)** 对所有场景都重要，提供物理技能和环境多样性，即使embodiment不同，技能也能迁移。

##### 环境数量扩展性研究

![alt text](<VLA Learning Notes.assets/image-1.png>)

为了量化π0.5的泛化能力，研究者进行了一个扩展性研究：横轴是训练数据中不同环境的数量（从0到100+），纵轴是在新场景的语言遵循率（OOD Follow Rate）。黄色曲线展示π0.5在全新场景的表现，而绿色横线是一个"作弊"的Oracle baseline——直接在测试环境收集数据训练的模型，代表理论上限。

曲线呈现出明显的两个阶段：

- **快速上升阶段**（0→40个环境）：性能从14%暴涨到76%，每增加环境都带来大量新信息，学习最基本的环境多样性，边际效益最高
- **趋向稳定阶段**（40→100个环境）：从76%增长到87%，增速变慢但仍在提升，主要学习edge cases，逐渐接近理论上限

**关键发现**：约100个训练环境后，π0.5的性能接近甚至略超Oracle baseline。这说明通过在100个其他家庭训练，模型在全新家庭的表现能达到"直接在测试家庭训练"的水平。Oracle只有85%而非100%，反映了任务本身的客观难度、硬件限制等因素。π0.5能达到87%，充分证明了co-training策略的有效性。

**实际意义**：

- **数据需求可行**（relatively accessible）：不需要1000或10000个环境，100个家庭×4小时≈400小时mobile data就足够，对研究团队/公司是可实现的目标
- **真正的泛化**：模型学会了泛化规律而非记忆环境，与简单过拟合有本质区别
- **商业化路径清晰**：核心数据只需~100环境的mobile robot数据，配合web数据和其他机器人数据，通过co-training实现强泛化

##### 核心结论

这个实验证明了数据的分工明确（WD→语义理解和识别新物体，ME→环境多样性，CE→物理技能迁移），绝大部分（97.6%）数据来自非目标平台但通过co-training得到有效利用。**核心结论**：数据多样性 > 数据相关性，co-training + 适量直接数据 = 强泛化能力。

#### 模型训练与推理

##### 模型架构

![alt text](<VLA Learning Notes.assets/image-2.png>)

π0.5基于π0 VLA，通过co-training能够输出多种类型的标签（包括动作和文本），因此可以用**同一个模型**控制机器人的高层和低层。整个系统包含Pre-trained VLA和Action Expert（300M参数），虽然Action Expert是独立模块，但它是π0.5的内置组件，与VLA联合训练，通过内部表示直接传递信息，而非两个独立的模型系统。

##### 推理流程

首先是**输入**（左下）：多视角图像（3张卧室场景）+ 高层提示（"clean the bedroom"），经过**Pre-trained VLA编码**（蓝色token序列）处理视觉和语言输入。

**Step 1 - 高层推理**：**高层分支**（上方灰色区域）通过离散自回归token解码生成子任务预测。例如，给定任务"clean the bedroom"和当前观察图像，模型会生成语义层面的子任务预测，如"pick up the pillow"。这个过程本质上是让模型"告诉自己"下一步应该做什么，类似chain-of-thought推理。

**Step 2 - 低层推理**：**低层分支**（右侧黄色区域）将高层命令"pick up the pillow"与VLA内部表示一起输入Action Expert（300M参数），通过**Flow Matching**技术和绿色方块的中间表示，最终输出连续动作[-1.7, 1.3, 3.1, 1.4]。这些连续动作包括关节角度、速度等底层控制信号，形成50-step（1秒）的action chunk。

图中**关键的虚线箭头**表示高层决策如何传递给低层，以及Action Expert如何从内部表示生成连续动作。

##### 技术实现细节

| 路径                         | 用途     | 技术               | 输出                                         |
| ---------------------------- | -------- | ------------------ | -------------------------------------------- |
| **Discrete Auto-regressive** | 高层推理 | Token-by-token生成 | 文本形式的语义动作（如"pick up the pillow"） |
| **Flow Matching**            | 低层推理 | 连续序列生成       | 50步的连续动作（1秒@50Hz控制频率）           |

**Action Chunk机制**：采用50-step action chunking而非单步预测，原因包括控制频率合理（50Hz是常见的机器人控制频率）、减少模型调用（提高实时性）、动作更流畅（避免抖动）、可提前规划（考虑1秒内的短期未来）。

##### 关键创新点

这个方法借鉴了Hi Robot系统的分层思想，但关键区别在于π0.5使用**同一个模型**进行高层决策和低层控制。这带来的好处包括：

- **信息无损传递**：不只通过文本，还通过内部表示（hidden states）传递丰富信息
- **避免模型间磨合**：高层和低层共享对视觉、物体、物理的理解，不会出现"理解不一致"  
- **端到端优化**：可以根据最终任务成功率联合优化整个pipeline，高低层学会互相配合
- **统一的"世界观"**：同一个模型学到的概念是一致的，不会出现"高层说pick up，低层不知道是什么"的问题

---

##### 概念补充：回归、离散Token生成、扩散模型、流匹配

为了理解π0.5的技术选择（为什么Pre-training用FAST、Post-training用Flow Matching），这里补充四种主要动作生成方法的原理和技术细节。

###### 1. 回归 (Regression)

**核心思想**：学习映射函数$f_\theta: X \to Y$，从观察（图像、状态）直接预测动作。神经网络通过多层非线性变换将输入映射到连续值输出。
  
**训练目标**：给定数据集$D = \{(x_i, y_i)\}$（观察-动作对），最小化预测值与真实值的差距：

$$
\min_\theta \sum_i L(f_\theta(x_i), y_i)
$$

常用均方误差（MSE）损失：$L(\hat{y}, y) = ||\hat{y} - y||^2$，表示预测动作$\hat{y}$与真实动作$y$的欧氏距离平方。通过梯度下降优化参数：$\theta_{t+1} = \theta_t - \eta \nabla_\theta L$。

**网络结构**：输入观察（如图像+机器人状态） → 多层全连接/卷积层 → 输出动作向量。最后一层直接输出连续值，无需激活函数（如sigmoid/softmax）。

**优势**：简单高效，推理只需一次前向传播（O(1)复杂度），适合实时控制。

**致命缺陷——多峰分布问题**：回归学习的是条件期望$E[y|x]$，只能输出单一点估计。当同一观察对应多种合理动作时（如从左/右/上方抓杯子），回归输出所有可能性的平均值。

**问题示例**：训练数据中"抓杯子"有50%从左抓、50%从右抓。回归会输出平均值（从中间抓），导致机器人手臂撞到杯子！这是因为MSE的最优解是条件期望：$\arg\min_{\hat{y}} E[||y - \hat{y}||^2] = E[y|x]$，数学上必然输出平均值。

###### 2. 离散Token生成 (Autoregressive)

**核心思想**：将连续动作离散化为有限的token集合，用类似GPT的自回归模型逐个token生成。解决多峰分布问题：不同token序列可以表示不同的动作模式。

**自回归分解**：将动作序列$y = [y_1, ..., y_n]$分解为条件概率的乘积：

$$
P(y) = P(y_1) \cdot P(y_2|y_1) \cdot P(y_3|y_1,y_2) \cdot \ldots \cdot P(y_n|y_1,...,y_{n-1})
$$

每个token的预测都依赖于前面所有token，这样可以捕捉序列内部的依赖关系。

**训练目标**：最大化真实token序列的对数似然，等价于最小化交叉熵损失：

$$
L = -\sum_i \log P(y_i | y_{1:i-1})
$$

**推理过程**：逐token采样生成。每步模型输出词汇表上的概率分布$P(\cdot|context) = \text{softmax}(logits)$，从中采样下一个token。可用temperature参数$T$控制多样性：$T$越高越随机，$T$越低越确定。

**离散化方法**：

- **简单分桶**：将连续值域`[-1, 1]`均匀划分为K个区间（如256档），`0.512` → token `131`
- **Vector Quantization (VQ)**：学习codebook $\{v_1, ..., v_K\}$，编码时映射到最近code向量：$\arg\min_i ||a - v_i||$
- **FAST**（π0预训练用）：利用时间相关性压缩，将50步×7维=350个连续值压缩为仅8个离散token，大幅减少序列长度

**优势**：能表示多峰分布（不同token序列=不同动作），训练稳定（与VLM架构统一）。

   **局限性**：

- 离散化损失精度：`0.512` → `0.51`的量化误差在精细操作中累积
- 推理慢：自回归需要串行生成（FAST需8次模型调用），难以并行加速
- 不适合高频控制：对需要50Hz实时响应的机器人，8次串行调用太慢

###### 3. 扩散模型 (Diffusion Models)

[Bilibiili -【大白话01】一文理清 Diffusion Model 扩散模型 | 原理图解+公式推导](https://www.bilibili.com/video/BV1xih7ecEMb?vd_source=de5f41a81376d29995d1f5309edd3f52)

[GoodNotes 笔记 - Denoising Diffusion Probabilistic Models Tutorial（扩散模型）](https://share.goodnotes.com/s/9DPIOpPEblnTRYqkrWZxiG)

**核心思想**：通过学习逆转数据的加噪过程来生成样本。将原始数据逐步添加噪声，获得含有每一步添加了噪声的版本。然后训练模型学习逆转这个加噪过程，从而生成原始数据。直接预测上一步的图像是很困难的，因此我们训练模型预测每一步添加的噪声，从而可以一步步去噪得到原始数据。

**前向过程（加噪）**：给真实动作$x_0$逐步加高斯噪声，经过T步（如1000步）变成纯噪声$x_T$。关键性质是可以一步到位采样任意时刻的噪声版本：

$$
x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon,\quad \epsilon \sim \mathcal{N}(0,I)
$$

其中$\bar{\alpha}_t$控制噪声程度（$t$越大噪声越多）。

**训练目标**：训练神经网络$\epsilon_\theta(x_t, t)$预测每一步加入的噪声，最小化预测误差：

$$
L = E_{t,x_0,\epsilon} [||\epsilon - \epsilon_\theta(x_t, t)||^2]
$$

训练时：随机选时间步$t$，从数据集采样动作$x_0$，生成加噪版本$x_t$，让模型预测噪$\epsilon$。

**反向过程（去噪/推理）**：从纯噪声$x_T \sim \mathcal{N}(0,I)$开始，每步用模型预测噪声并除，迭代T步得到最终动作。DDPM的去噪公式：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z
$$

其中$z$是少量随机性（除最后一步）。需要T=50-1000次迭代，每次调用一次模型。

**优势**：能建模复杂多峰分布，生成质量高。**劣势**：推理慢（需要几百上千次模型调用），不适合要实时响应的机器人控制。

**详细内容请见（含有PPT和详细笔记）：[GoodNotes 笔记 - Denoising Diffusion Probabilistic Models Tutorial（扩散模型）](https://share.goodnotes.com/s/9DPIOpPEblnTRYqkrWZxiG)**

###### 4. 流匹配 (Flow Matching)

**核心思想**：学习时间依赖的向量场$v_\theta(x, t)$，描述如何从简单分布（标准高斯噪声）流动到目标分布（真实动作）。类比GPS导航：学习从任意起点到终点的"最优导航路径"。

**定义确定性路径**：从噪声$x_0 \sim \mathcal{N}(0,I)$到真实动作$x_1$之间建立线性插值路径：

$$
x_t = (1-t) \cdot x_0 + t \cdot x_1, \quad t \in [0,1]
$$

这条路径的"速度场"（切向量）为$v_t(x) = dx/dt = x_1 - x_0$，表示在路径上每个点应该往哪个方向移动。

**训练目标**：训练神经网络$v_\theta(x_t, t)$学习预测速度场，最小化与真实速度的差距：

$$
L = E_{t,x_0,x_1} [||v_\theta(x_t, t) - (x_1 - x_0)||^2]
$$

训练时：随机采样时间$t \in [0,1]$、噪声$x_0$和真实数据$x_1$，计算插值点$x_t$，让模型预测应该往哪个方向走。

**推理过程**：使用欧拉法数值积分求解ODE（常微分方程）：

$$
x_{i+1} = x_i + v_\theta(x_i, t_i) \cdot \Delta t
$$

从噪声$x_0$开始，每步询问模型"当前应该往哪走"，迭代K=5-10步即可到达目标动作。相比Diffusion基于SDE（随机微分方程），Flow Matching的路径是确定性的，更稳定可控。

**核心优势**：

- **推理效率高**：5-10步 vs Diffusion的50-1000步，减少90-99%计算
- **训练稳定**：目标速度场$(x_1 - x_0)$有解析形式，无需设计复杂的noise schedule
- **确定性路径**：基于ODE而非SDE，轨迹可预测可控

**π0.5的应用**：Action Expert用Flow Matching生成50步×7维的连续动作chunk（1秒@50Hz控制频率）。推理时约10步迭代（~200ms），满足实时控制需求。这是π0.5选择Flow Matching而非Diffusion的关键原因。

| 方法          | 训练目标      | 推理复杂度      | 表达能力 | 精度             |
| ------------- | ------------- | --------------- | -------- | ---------------- |
| 回归          | MSE           | O(1)            | 单峰     | 高               |
| 离散Token     | Cross-entropy | O(n) 自回归     | 多峰     | 中（离散化损失） |
| Diffusion     | 预测噪声      | O(T), T=50-1000 | 多峰     | 高               |
| Flow Matching | 预测速度场    | O(K), K=5-10    | 多峰     | 高               |

π0.5采用混合策略：Pre-training阶段使用离散Token（FAST）以提高训练效率和稳定性，Post-training阶段引入Flow Matching以实现高精度、低延迟的实时控制。这样既利用了离散token训练快、易与VLM统一的优势，又通过Flow Matching获得连续动作的高精度和实时性。

---

#### 实际应用和未来改进

π0.5的关键测试是让机器人在从未见过的新家庭环境中执行复杂的长时清洁任务（如收拾盘子、铺床、清理卧室）。这比之前的VLA演示更困难，因为过去的演示通常在与训练数据相同或相似的环境中进行。实验结果显示π0.5能够：处理环境变化和人类干扰、接受不同粒度的语言指令（从高层的"put the dishes in the sink"到具体的"pick up the round brush"）、完成需要语义理解和子任务分解的复杂行为。

虽然π0.5仍有局限（在高层语义推断和电机命令方面会犯错），但这证明了通过co-training让机器人从多种知识源学习的方法是有效的。未来的改进方向包括：利用verbal feedback、从自主经验学习、在不熟悉情况下主动寻求帮助，以及改进知识迁移和增加数据源多样性。

### 1.2 Paper-Pi05: a Vision-Language-Action Model with Open-World Generalization

[Paper arxiv](https://arxiv.org/abs/2504.16054)

[Local PDF](../Paper/Paper-Pi05:%20a%20Vision-Language-Action%20Model%20with%20Open-World%20Generalization.pdf)

### 1.3 Code-openpi

[Github Repo](https://github.com/Physical-Intelligence/openpi)

[Local Repo](../Repo/openpi)

## 2. VLA Survey

### 2.1 A Survey on Vision-Language-Action Models: An Action Tokenization Perspective

[Paper arxiv](https://arxiv.org/abs/2507.01925)

[Local PDF](../Paper/A%20Survey%20on%20Vision-Language-Action%20Models:%20An%20Action%20Tokenization%20Perspective.pdf)

### 2.2 Pure Vision Language Action (VLA) Models: A Comprehensive Survey

[Paper arxiv](https://arxiv.org/abs/2509.19012)

[Local PDF](../Paper/Pure%20Vision%20Language%20Action%20(VLA)%20Models-%20A%20Comprehensive%20Survey.pdf)

## 3. MuJoco

[Github Repo](https://github.com/unitreerobotics/unitree_mujoco)

[Local Repo](../Repo/unitree_mujoco)
