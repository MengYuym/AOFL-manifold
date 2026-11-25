# AOFL-manifold

AOFL 本地研究代码，包含 manifold / 子空间实验和 decoder 特征重构实验。

## Decoder 重构

- `Model/FeatureReconstruction.py`：三个模态各自使用独立的两层 MLP，将细化前的共享特征 g 与特有特征 h 拼接，重构对应的输入特征 x。
- 默认结构：Linear(512, 256) → ReLU → Linear(256, 256)。重构目标 x 断开梯度。
- `Train/RunDecoderComparison.py`：通过 install() 给训练器接入 decoder，屏蔽 padding，加入重构 MSE；baseline 代表原 AOFL 加 decoder，gapd 代表 GAPD 加 decoder。
- `Train/RunAOFLDecoderPaired.py`：多种子配对实验入口。

## 环境和数据

依赖见 requirements.txt。本地原环境为 Python 3.9。数据特征、模型权重、虚拟环境和训练日志不随源码上传。
请自行准备 Data/MELD 和 Data/IEMOCAP 下的特征文件，具体文件名参考 Dataset 中的数据加载代码。

## 运行注意

从项目根目录运行。原实验脚本保留了一些本机绝对输出路径；在其他电脑运行前需修改这些路径。
RunAOFLDecoderPaired.py 等脚本还依赖外部 experiment_plan.json 及源码哈希，不能在缺少原实验计划时直接运行。

原 AOFL 加 decoder 的实验入口为：

```powershell
python Train/RunDecoderComparison.py --variant baseline
```

运行前请调整脚本中的 output 路径并确保其父目录存在。该入口固定 seed=7、30 个 epoch、重构权重 0.1；已有输出应先确认，避免覆盖实验结果。

本仓库是当前本地研究代码的快照，保留历史实验和分析脚本；上传操作不代表已经完成跨机器复现验证。
