# 1D-GAN-Data-Augmentation-MATLAB
A MATLAB implementation of 1-D Generative Adversarial Networks (1D-GAN) for tabular/time-series data generation, distribution validation, and downstream SVM classification augmentation.
# 1-D Generative Adversarial Networks (1D-GAN) for Data Augmentation in MATLAB

本项目提供了一套基于 MATLAB 的一维生成对抗网络（1D-GAN）数据合成与样本增强完整流程。针对高维特征或时序切片数据（如 1024 维信号特征），利用生成器（Generator）与判别器（Discriminator）的对抗博弈学习多类真实样本的高维统计分布，生成高质量合成数据以解决样本匮乏、类别不平衡等问题，并通过 SVM 分类器对增强效果进行量化验证。

---

## 📌 核心功能与技术亮点

1. **一维对抗生成架构**：
   - **生成器 (Generator)**：基于两层前馈全连接网络（128-64 神经元），输入随机高斯噪声向量，输出 1024 维合成特征向量。
   - **判别器 (Discriminator)**：基于全连接分类网络（64-32 神经元），输出真伪置信度，结合均方误差/二元交叉熵损失进行反向传播更新。
2. **端到端标准化与数据还原**：
   - 内置 Z-score 特征标准化与反标准化模块，保证网络梯度更新稳定并在输出端严格还原物理量纲。
3. **多维度统计一致性可视化**：
   - 面积图对比（原始 vs 合成时序分布）。
   - 双特征散点图（gscatter 观察类别簇聚合与重叠边界）。
   - 概率密度曲线（PDF）与直方图（Histogram）分布重叠度校验。
   - 箱线图（Boxplot）与正态概率图（Probplot）离群点及分位数验证。
4. **下游任务有效性验证（SVM）**：
   - 提供使用合成数据扩充训练集后训练 ECOC 多分类 SVM 模型（`fitcecoc`），并在独立原始测试集上评估分类泛化精度。

---

## 📊 数据集结构与参数配置

以多工况特征数据（1000 样本 × 1024 特征，共 4 分类，每类 250 个样本）为例：

| 配置项 | 设定值 | 说明 |
| :--- | :---: | :--- |
| **样本总数 / 特征维度** | $1000 \times 1024$ | 单样本包含 1024 个点/特征 |
| **类别总数** | 4 类 | 标签分布：1 (1~250), 2 (251~500), 3 (501~750), 4 (751~1000) |
| **训练轮数 (Epochs)** | 100 | 迭代轮次 |
| **批大小 (Batch Size)** | 64 | 单批次输入样本数 |
| **学习率 (Learning Rate)** | 0.001 / 0.01 | 优化步长 |
| **训练/测试集切分** | 70% / 30% | 用于验证扩增后的分类泛化性能 |

---

## 📁 推荐仓库目录结构

```plaintext
├── data/
│   └── data_total.mat          # 示例数据（包含 Rawdata 矩阵）
├── src/
│   ├── GAN_Synthetic_Data.m    # 基于神经网络工具箱的 1D-GAN 主程序
│   ├── GAN_basic_math.m        # 基于函数句柄的轻量数学映射 GAN 脚本
│   └── plot_distributions.m    # 9宫格统计分布绘图子程序
├── figures/
│   ├── original_vs_synthetic.png # 散点与直方图对比展示
│   └── boxplot_comparison.png    # 箱线图分布对比
├── README.md
└── LICENSE
