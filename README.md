# 玄瞳 MetaEye 无监督可见光与红外检索基线

本目录实现项目使用的 ADCA + TransReID 基线。训练读取 RegDB 图像并生成聚类伪标签，导出模型权重、双向检索指标和图库特征。入口为 `main.py`，默认配置是 `configs/regdb_laptop.json`。

RAHSM 公式复现版已接入相同框架，配置为 `configs/rahsm_regdb_laptop.json`。它包含共享标签关联、CHP、KHN、PTS 及角色感知损失。实现选择和验证范围见 [RAHSM 复现说明](docs/RAHSM复现说明.md)。

验证命令为 `.\.venv\Scripts\python.exe scripts\verify_rahsm.py`，仅使用合成数据检查公式、梯度、阶段切换和保存恢复。正式性能实验在实现核对完成后统一组织。

公开仓库包含源码、配置、验证脚本、验证摘要和上游许可证。数据集、模型权重、虚拟环境、训练产物及本机原始验证日志保留在本地，不纳入 Git；首次克隆后按文末步骤安装依赖并获取数据与权重。本项目是基于论文和上游源码构建的研究实现，RAHSM 的论文精度尚未复现。

2026 年 11 月 16 日的本机验证已完成数据完整性检查、CUDA 前向与反向传播、四项核心检查，以及真实 RegDB 上两轮各 5 次迭代的短程训练与双向评测。短程运行分别覆盖单模态阶段与 CMA 阶段，结果保存在 `runs/regdb_pipeline_check`，峰值训练显存约 1.59 GiB。短程指标用于记录流程验证，正式 50 轮、10 组划分的精度结论需要相应实验。

## 当前环境

- Python 3.12 独立虚拟环境位于 `.venv`。
- PyTorch 2.7.1 + CUDA 12.8，torchvision 0.22.1。
- 使用本机 RTX 5060 Laptop GPU，约 8 GB 显存。
- 模型使用官方 TransReID 的 ViT-Base 全局特征提取器和官方推荐的 ImageNet 权重。
- RegDB 原始包位于 `data/downloads/RegDB.zip`，解压数据位于 `data/RegDB`。

## 开始运行

在 PowerShell 中执行以下命令。已有环境可直接运行。

```powershell
cd .\MetaEye-Baseline

# 检查 Python、CUDA、显卡、权重和数据是否就绪
.\.venv\Scripts\python.exe main.py doctor

# 检查 10 组划分、可见光与红外身份对应及全部图片能否打开
.\.venv\Scripts\python.exe main.py validate-data --verify-images

# 检查评测公式、聚类距离与采样器
.\.venv\Scripts\python.exe scripts\verify_core.py

# 短程验证。独立保存，包含单模态训练和 CMA 两个阶段
.\.venv\Scripts\python.exe main.py train --epochs 2 --warmup-epochs 1 --iterations 5 --output runs\my_pipeline_check
.\.venv\Scripts\python.exe main.py evaluate --checkpoint runs\my_pipeline_check\checkpoint_last.pt

# 正式起跑配置。第 1 组划分、种子 1，50 轮，每轮 400 次更新
.\.venv\Scripts\python.exe main.py train

# 中断后继续同一次训练，配置必须与保存的配置相同
.\.venv\Scripts\python.exe main.py train --resume runs\regdb_trial1_seed1\checkpoint_last.pt

# 评测固定的最后一轮权重，同时导出两种模态图库特征
.\.venv\Scripts\python.exe main.py evaluate --checkpoint runs\regdb_trial1_seed1\checkpoint_last.pt
```

正式配置的每模态批量为 8 张图，即每批 4 个伪身份、每身份 2 张图。可见光另有一份通道增强视图，一次前向共 24 张图。输入大小为 288×144，启用 AMP 与梯度检查点。显存不足时可将 `batch_size` 改为 4，并把 `feature_batch_size` 改为 8，实验中记录调整。

对另一组划分训练时同时指定独立输出目录，例如 `main.py train --trial 2 --output runs/regdb_trial2_seed1`。完整 RegDB 结果应分别训练和评测 1—10 组划分后求平均，重复种子实验另行组织。单次短程验证只用于确认流程能运行。

## 代码阅读顺序

| 文件 | 作用 |
|---|---|
| `configs/regdb_laptop.json` | 数据位置、轮次、批量、聚类、温度与记忆更新参数 |
| `main.py` | 命令行入口 |
| `metaeye/data.py` | 原始划分校验、图像增强、伪身份均衡采样 |
| `metaeye/model.py` | 加载官方 TransReID、ImageNet 初始化、BNNeck 和特征归一化 |
| `metaeye/clustering.py` | k-reciprocal Jaccard、DBSCAN 和跨模态记忆聚合 CMA |
| `metaeye/rahsm.py` | RAHSM 共享标签、CHP、KHN、PTS、RACL 损失与模态均值更新 |
| `metaeye/training.py` | 两阶段训练、对比损失、动量更新、日志和断点 |
| `metaeye/evaluation.py` | 特征提取、RegDB 双向 Rank-1/mAP/mINP、图库特征导出 |
| `third_party` | 固定版本的官方源码与原许可证，版本记录见 `sources.json` |

## 一轮训练做什么

1. 从可见光与红外训练图像分别提取特征。
2. 用 k-reciprocal Jaccard 距离和 DBSCAN 分别生成伪标签，排除离群点。
3. 从簇均值初始化两套记忆库。前 30 轮分别学习，第 31 轮起启用 ADCA 的 CMA 聚合。
4. 采样伪身份均衡的批次，计算两种模态的记忆对比损失，反向传播并更新模型与记忆库。
5. 保存本轮模型、优化器、调度器、AMP 状态、伪标签和日志。

真实身份标签用于划分完整性检查和测试评测。训练数据集返回聚类伪标签，损失、采样和 CMA 使用这些伪标签。RegDB 的训练和测试标签各自从 0 开始重新编号，划分泄漏检查以原始身份文件夹核对。

## 输出文件

- `checkpoint_last.pt` 保存模型、优化器等状态，采用临时文件写完后替换。
- `config.json` 保存实际使用的配置。
- `training.jsonl` 记录损失、簇数、离群点、CMA 匹配数、耗时和峰值显存。
- `pseudo_labels_latest.npz` 与 `cma_pairs_latest.json` 保存最近一轮伪标签和 CMA 关联。
- `metrics.json` 由评测命令生成，记录双向 Rank-1、mAP、mINP，数值为 0—1。
- `gallery_features.pt` 由评测命令生成，包含可见光和红外特征、图片路径、测试标签和模型关联信息。

### 论文材料自动归档

后续训练和评测自动保存以下资料。每次恢复、重试及评测采用独立归档路径。

- `provenance/run_*/` 保存实际配置、Python/依赖/显卡信息、源码 ZIP 与校验值、ImageNet 权重校验值、原始划分和训练样本顺序。
- `epochs/epoch_轮次_*/clustering.npz` 保存该轮更新前的可见光与红外特征、伪标签和原始簇中心，可结合该次运行的 `training_samples.json` 对应图片。
- 同目录的 `cma_pairs.json`、`steps.jsonl`、`summary.json` 保存关联记录、逐步损失与学习率、AMP 跳步、耗时和显存。`status=completed` 标记该轮已保存断点，未完成的尝试留作排查。
- 单模态阶段末和总训练末的 `checkpoint.pt` 额外归档。根目录的 `checkpoint_last.pt` 继续用于恢复训练。
- `evaluations/evaluation_*/` 保存本次固定权重及其 SHA-256、配置、双向指标、图库特征，以及逐查询 AP、INP、正确候选排名、Top-20 顺序/分数和 CMC 曲线。

根目录的 `latest` 文件是快速读取入口，逐轮记录和各次评测保存在上述归档目录。RAHSM 跨模态阶段额外保存共享标签、身份与子簇中心、实际 K 值、λ 轨迹和最后一批难样本选择记录。置信度自适应 λ 属于后续改进任务。

每轮完整浮点训练特征约 12 MB，关键节点权重约 0.69 GB。权重和评测特征优先通过本地硬链接留存，后续更新使用文件替换以保留历史版本。已有两轮流程短跑仅能归档目前仍保存的文件，首轮已被覆盖的伪标签和当时缺失的逐步日志无法补回。

同一输出目录已有权重时需显式 `--resume`，新实验使用新目录。评测固定的最后一轮权重，训练过程依据训练数据生成伪标签。

## 实现与实验口径

这是项目的 TransReID 骨干适配版本。官方 ADCA 原实现使用 ResNet，项目把全局特征提取器换为 TransReID，使用共享 ViT、BNNeck、归一化特征，关闭 SIE/JPM 与身份分类头。保留可见光原视图与通道增强视图、独立模态伪标签及记忆库、CMA 实例配对投票与动量记忆更新。

为适配 Windows，以 CPU 精确近邻代替 ADCA 的 GPU FAISS 搜索，保留原 Jaccard 编码公式，并用官方数学函数作数值对齐检查。该实现优先支持 RegDB 规模的数据。CMA 使用原 RegDB 训练代码的投票与聚合方式，确定并列票数时的顺序可能与上游不同。

50 轮、前 30 轮单模态及 SGD 的安排依据 RAHSM 第 2.2 节。笔记本批量小于论文设置。官方 ADCA 的原始脚本采用 Adam 及独立阶段安排，因此本项目结果应标明实际配置。RAHSM 配置在跨模态阶段采用共享标签与 RACL，具体公式及原文未明确的细节见复现说明。

## 数据和源码来源

- ADCA 官方仓库为 https://github.com/yangbincv/ADCA ，固定到 `8d0a8cb3e74ec3383ebe9f528ae229d7c6bbdab9`，MIT 许可。
- TransReID 官方仓库为 https://github.com/damo-cv/TransReID ，固定到 `dec55046fcdfadee14e2c28e2df89305d8f7557a`，MIT 许可。只将旧 `torch._six` 导入改为 `collections.abc`。
- ImageNet 权重采用 TransReID README 推荐的 `jx_vit_base_p16_224-80ecf9dd.pth`，下载时校验 SHA-256 前缀。
- RegDB 下载链接来自 HiCMD 作者仓库 https://github.com/bismex/HiCMD#training-and-testing ，文件来源和校验值见 `data/regdb_source.json`。使用数据及发表成果应遵守原数据库的使用条件并引用 Nguyen 等的论文 https://doi.org/10.3390/s17030605 。
- PyTorch 安装依据 https://pytorch.org/get-started/previous-versions/ 。

重新安装时，先用 Python 3.12 创建 `.venv`，安装 CUDA 版 torch/torchvision，再安装 `requirements.txt`。执行 `scripts/fetch_assets.py sources`、`weights`、`regdb` 可获取所需文件。下载需要代理时，可给单次命令追加 `--proxy http://127.0.0.1:实际端口`。成功后运行完整性检查。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\fetch_assets.py sources
.\.venv\Scripts\python.exe scripts\fetch_assets.py weights
.\.venv\Scripts\python.exe scripts\fetch_assets.py regdb
```
