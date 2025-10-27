# Weekly Progress

## Week 1（2025.10.8 - 2025.10.17）

### Week 1 完成内容

#### 1. 理论学习

- **π0.5 Blog 学习**
  - 阅读 [Blog-Pi05: A VLA with Open-World Generalization]((https://www.physicalintelligence.company/blog/pi05))
  - 学习了 π0.5 的研究背景、解决方案、实验设计与结果，并整理了笔记（见 [VLA Learning Notes - 1.1 Blog-Pi05: A VLA with Open-World Generalization](VLA%20Learning%20Notes.md#11-blog-pi05-a-vla-with-open-world-generalization) ）

- **Diffusion Model 知识学习**
  - π0.5 的 Flow Matching 技术与扩散模型密切相关，因此补充学习了扩散模型的知识
  - 学习 DDPM（Denoising Diffusion Probabilistic Models）的原理，理解了前向加噪过程和反向去噪过程，笔记见 [笔记 - Denoising Diffusion Probabilistic Models Tutorial（扩散模型）](../Paper/Denoising%20Diffusion%20Probabilistic%20Models%20Tutorial（扩散模型）.pdf)

#### 2. 实践环境搭建

- **MuJoCo 仿真环境部署**
  - 在 macOS 上成功部署 [unitree_mujoco](../Repo/unitree_mujoco/) 仿真环境
  - 解决了官方代码在 macOS 上的兼容性问题，修复了兼容性 fallback 分支中的一个小 bug。具体见 [macOS修改说明](../Repo/unitree_mujoco/macOS修改说明.md)
  - 成功运行机器人站起来/坐下的基本动作

[环境部署和动作演示（3倍速）](./Weekly%20Progress.assets/video.mp4)

<!-- markdownlint-disable-next-line MD034 -->
https://github.com/user-attachments/assets/6aac130f-94b9-4149-872e-d3e4f08c1e40

<!-- markdownlint-disable-next-line MD033 -->
<video src="./Weekly Progress.assets/video.mp4" controls></video>

### Week 2 计划

#### 1. 理论深入

- **Flow Matching 学习**
  - 深入理解 Flow Matching 的具体原理，理解为什么 π0.5 在 Post-training 阶段选择 Flow Matching

- **π0.5 论文精读**
  - 精读 [π0.5 完整论文](../Paper/Paper-Pi05:%20a%20Vision-Language-Action%20Model%20with%20Open-World%20Generalization.pdf)
  - 理解实验设计和消融实验结果
  - 学习模型架构细节和训练策略

#### 2. 实践探索

- **VLA Models 驱动机器人**
  - 研究 [openpi 代码库](../Repo/openpi/) 的使用方法，尝试将 VLA 模型与 MuJoCo 仿真环境结合
  - 探索如何用 π0 或其他 VLA 模型控制 unitree 机器人

### Week 1 总结

本周主要建立了 VLA 领域的理论基础和实践环境。通过学习 π0.5 Blog 和 Diffusion Model，对视觉-语言-动作模型的核心技术有了系统性理解。同时成功搭建了机器人仿真环境，为后续的实践探索打下了基础。下周将深入学习 Flow Matching 和 π0.5 论文，并开始尝试用 VLA 模型驱动机器人。

---

## Week 2（2025.10.18 - 2025.10.27）

### Week 2 完成内容

#### 1. π0.5 论文精读

- **论文核心内容学习**
  - 精读 [π0.5 完整论文](../Paper/Paper-Pi05:%20a%20Vision-Language-Action%20Model%20with%20Open-World%20Generalization.pdf) 的前半部分（Abstract, Introduction, Related Work, Preliminaries）
  - 深入理解 π0.5 的核心创新点：异构数据联合训练（co-training on heterogeneous data）实现开放世界泛化
  - 学习了 VLA 的发展脉络和 π0.5 相对于其他模型（OpenVLA, Octo, RT-2等）的技术优势
  - 详细笔记见 [VLA Learning Notes - Paper-Pi05: a Vision-Language-Action Model with Open-World Generalization](VLA%20Learning%20Notes.md#12-paper-pi05-a-vision-language-action-model-with-open-world-generalization)

- **VLA 关键技术深度学习（Preliminaries 部分）**
  - VLA 训练目标：最大化对数似然 $\max_\theta E[\log \pi_\theta(a_{t:t+H} | o_t, \ell)]$
  - 观察与动作表示：多模态观察的编码方式、离散token与连续action的表示
  - **FAST 动作编码**：通过VQ-VAE实现动作序列的压缩表示，类似"关键帧+神经网络拟合"
  - **Action Expert 架构**：Pre-training使用FAST离散token，Post-training使用Flow Matching生成连续动作
  - 理解了Pre-training（广泛任务泛化）和Post-training（精细动作控制）的不同目标和设计

#### 2. Flow Matching 深度学习

为了深入理解 π0.5 的 Action Expert 中使用的 Flow Matching 技术，系统学习了 Flow Matching 的理论和应用。

Flow Matching详细笔记见：[笔记 - Flow Matching Explained - From Noise to Robot Actions（流匹配）](../Paper/Flow%20Matching%20Explained%20-%20From%20Noise%20to%20Robot%20Actions（流匹配）.pdf)

- **理论基础**
  - 学习了连续归一化流（CNF）和常微分方程（ODE）的基础概念
  - 理解速度场（velocity field/vector field）的核心作用：在每个位置定义"往哪走、多快走"
  - 掌握概率路径（probability path）的构造：从噪声分布p₀到数据分布p₁的平滑演化

- **条件流匹配（CFM）**
  - **核心问题**：直接学习整体速度场不可解（源分布到目标分布有无穷多种演化方式）
  - **解决方案**：增加条件——给定具体终点Z和线性插值方法，使速度场变得可计算
  - **训练方式**：随机采样(时刻τ, 起点x₀, 终点x₁)，只学习单点的速度方向，避免模拟完整轨迹（相比CNF更高效）
  - **类比理解**：类似插值动画，只有规定了起点、终点和插值方式，每一帧的状态才能确定，才能训练

- **机器人控制应用**
  - 在机器人控制中，一个"点"是n×H维矩阵（如7关节×50时间步=350维）
  - 通过Flow Matching从随机噪声矩阵逐步去噪生成完整的动作序列
  - 代码实现：学习了FlowMatchingNetwork的架构（观察编码器、时间编码器、动作编码器、解码器）
  - 训练损失：最小化预测速度和理想速度之间的均方误差（MSE Loss）
  - 推理生成：通过欧拉法迭代求解ODE，steps=10-50次即可生成高质量动作序列

- **与 DDPM 对比理解**
  - **DDPM**：学习逆向还原"数据→噪声"的扩散过程，需要1000步，路径复杂
  - **Flow Matching**：直接学习"噪声→数据"的直线路径，10-50步即可，目标导向更明确
  - **核心区别**：DDPM是"过程驱动"（逆向工程），FM是"目标驱动"（GPS导航）
  - **优势**：更快、更平滑、更适合实时机器人控制

#### 3. 关键理解与总结

- **VLA 的本质**：将视觉-语言-动作统一为sequence modeling，通过Transformer架构实现端到端学习
- **π0.5 的创新**：
  - 异构数据联合训练（视觉任务、高层语义任务、低层动作任务）
  - 分阶段设计（Pre-training用离散FAST、Post-training用连续Flow Matching）
  - 实现开放世界泛化（在未见过的环境中执行复杂任务）
- **Flow Matching 的价值**：
  - 生成平滑连续的动作序列（ODE连续性+训练数据平滑性）
  - 支持多模态动作分布（随机初始化产生不同合理方案）
  - 高效训练和快速生成（相比DDPM步数少10-100倍）
  - 长期规划能力（一次生成整个动作序列，不是逐步预测）

### Week 3 计划

#### 1. 完成 π0.5 论文精读

- **IV. THE π0.5 MODEL AND TRAINING RECIPE**：π0.5的完整模型架构和训练流程
- **V. EXPERIMENTAL EVALUATION**：实验设计、评估指标、与baseline对比、消融实验分析
- **VI. DISCUSSION AND FUTURE WORK**：总结讨论、局限性和未来工作

#### 2. 调整学习节奏

- 将周报提交时间调整为每周五 **（Week 3提交时间2025.10.31）**
- 适当加快阅读速度，在保证理解质量的前提下提高效率

### Week 2 总结

本周深入学习了 π0.5 论文的理论基础部分，重点学习了 Flow Matching 这一关键技术。通过系统学习理论基础、条件流匹配、机器人应用和代码实现，对 VLA 模型的动作生成机制有了深刻理解。通过与 DDPM 的对比，认识到 Flow Matching 在机器人控制中的独特优势（快速、平滑、目标导向）。

由于在 Flow Matching 的理解上花费了较多时间（涉及ODE、概率路径、CFM等较深的理论概念），本周进度不及预期。下周将加快学习节奏，确保完成论文剩余部分的精读。
