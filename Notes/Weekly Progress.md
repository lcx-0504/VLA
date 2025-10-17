# Weekly Progress

## Week 1（2024.10.8 - 2024.10.17）

### 本周完成内容

#### 1. 理论学习

- **π0.5 Blog 学习**
  - 阅读 [Blog-Pi05: A VLA with Open-World Generalization]((https://www.physicalintelligence.company/blog/pi05))
  - 学习了 π0.5 的研究背景、解决方案、实验设计与结果，并整理了笔记（见 [VLA Learning Notes - 1.1 Blog-Pi05: A VLA with Open-World Generalization](VLA%20Learning%20Notes.md#11-blog-pi05-a-vla-with-open-world-generalization) ）

- **Diffusion Model 知识学习**
  - π0.5 的 Flow Matching 技术与扩散模型密切相关，因此补充学习了扩散模型的知识
  - 学习 DDPM（Denoising Diffusion Probabilistic Models）的原理，理解了前向加噪过程和反向去噪过程，笔记见 [GoodNotes 笔记 - Denoising Diffusion Probabilistic Models Tutorial（扩散模型）](https://share.goodnotes.com/s/9DPIOpPEblnTRYqkrWZxiG)

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

### 下周计划

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

### 本周总结

本周主要建立了 VLA 领域的理论基础和实践环境。通过学习 π0.5 Blog 和 Diffusion Model，对视觉-语言-动作模型的核心技术有了系统性理解。同时成功搭建了机器人仿真环境，为后续的实践探索打下了基础。下周将深入学习 Flow Matching 和 π0.5 论文，并开始尝试用 VLA 模型驱动机器人。
