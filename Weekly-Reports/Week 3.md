# Week 3（2025.10.28 - 2025.11.01）

## 本周完成内容

### 1. 仓库结构重构

- 将原有的Notes/和Paper/目录重新组织为主题式结构
- 新结构：Flow-Matching/、Pi0-Pi0.5/、VLA-Survey/、Weekly-Reports/
- 便于后续知识管理和查找

### 2. 整理Flow Matching的理论，及其在π0和0.5上的应用

[Flow Matching Notes.md](../Flow-Matching/Flow%20Matching%20Notes.md)

整理了Flow Matching的通用理论和在机器人中的应用，包括：

- Flow Matching基础、条件流匹配（CFM）、与DDPM的对比
- CFM的训练过程（forward方法）、推理流程（sample_actions、denoise_step）
- 时间步处理（Beta分布采样、正弦位置编码）
- Adaptive RMSNorm的实现
- π0/π0.5中Action序列的表示和处理方式

### 3. π0.5论文完成精读

[Pi0.5 Paper Notes.md](../Pi0-Pi0.5/Pi0.5%20Paper%20Notes.md)

完成了论文全文（I-VI章）的精读，学习了：

- 观察预处理、FAST编码（FSQ tokenizer）
- Flow Matching的训练和推理流程
- 时间步采样（Beta分布）
- Action Expert架构（PaliGemma + Expert结构）
- 本体感知离散化处理
- Attention Mask构建（控制token间可见性）
- Combined Loss（FAST + FM混合损失）

### 4. π0论文阅读与π0/π0.5技术对比

[Pi0-Pi0.5 Comparison.md](../Pi0-Pi0.5/Pi0-Pi0.5%20Comparison.md)

阅读了π0 Paper和Blog，理解了π0的基本架构和训练策略，对比了π0与π0.5的核心技术差异：

- Action Expert中引入Adaptive RMSNorm进行时间步注入
- State Token处理方式的变化（从projection到离散化）
- 训练策略的演进（从两阶段训练到联合训练）

### 5. 代码分析

- 大致浏览了openpi仓库中的关键实现文件
- 将部分关键代码实现整合到Flow Matching Notes、Pi0.5 Paper Notes和Comparison笔记的对应章节中

## 下周可以进一步拓展的方向

### 1. Flow Matching深入学习

目前对Flow Matching的理解主要集中在原理层面和代码阅读，下一步计划：

- 补充更底层的数学基础（如最优传输理论、常微分方程数值求解等）
- 尝试从零实现一个简化版的Flow Matching（理解训练和采样的完整流程）
- 运行openpi中的相关代码，实际观察训练过程

### 2. VLA领域High-Level了解

- 阅读VLA Survey论文，了解领域整体情况
- 了解其他主流VLA模型的核心思路
- 理解VLA领域的主要研究方向和挑战

### 3. 其他工作

- 根据与张老师的交流结果，安排后续工作重点。

## 本周总结

本周根据张老师的反馈，重点完成了Flow Matching的学习和代码层面的理解。通过整理教程、精读论文、分析代码，对π0和π0.5的核心技术有了比较系统的认识。

在理论方面，理解了Flow Matching从ODE、速度场、条件流匹配的数学原理，以及与DDPM的区别。在代码方面，分析了openpi中的关键实现，包括训练的forward过程、推理的采样过程、时间步的处理等。

通过对比π0和π0.5，了解了模型演进过程中的设计考虑，如Adaptive RMSNorm的引入、State Token处理方式的变化等。

本周在VLA学习上投入了较多时间，本科课程和作业方面有一些积压。下周会在做好课程和科研平衡的同时，继续提高学习效率，推进VLA相关内容的学习。

## 想与张老师交流的问题

### **1. 关于本科毕业设计**

由于本科学院在大四安排了较多的课程和作业，如果毕设能够与张老师这边的科研方向结合，一方面可以将毕设工作和当前的VLA学习更好地整合，提高时间利用效率；另一方面也能让我对VLA领域有更系统、更深入的理解。目前本科毕设的开题答辩预计安排在11月12日左右，想请问张老师是否方便在选题或研究方向上给予一些指导？

### 2. 关于硕士毕业要求

在上周张老师分享的周报中，看到昊林同学提到了关于硕士毕业的具体要求。我也希望能够提前了解一下相关的指标要求，例如论文发表的数量和质量要求（如目标会议/期刊级别）、时间节点规划等，以便更好地安排学习和科研进度。

### 3. 关于周报反馈

张老师先前分享的同学们的范例都是飞书文档，我使用GitHub会不会给张老师带来不便？之后是否需要像本次一样提供飞书版本的周报？
