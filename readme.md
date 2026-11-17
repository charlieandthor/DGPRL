# DGPRL: 动态图与原型表征学习时序异常检测 



---

## 📁 1. 数据集准备

### 数据集简介
在时序异常检测领域，使用标准公开数据集进行评测是行业惯例。本研究所采用的数据集广泛应用于各大顶会和权威文献中：
- **MSL (Mars Science Laboratory)** & **SMAP (Soil Moisture Active Passive)**：美国宇航局（NASA）的火星科学实验室和土壤湿度监测卫星的真实遥测数据集。
- **PSM (Pooled Server Metrics)**：eBay 应用服务器节点的多维服务器指标数据。
- **SMD (Server Machine Dataset)**：大型互联网公司多个应用服务器的大规模基准数据集。
- **SWaT (Secure Water Treatment)**：工业水处理测试床真实物理过程与攻击场景数据集。

### 下载地址
可以从以下任意渠道下载预处理好的数据集，并将其放置于 `./data` 目录下：
- **Google Drive**: [[点击下载]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing)
- **百度网盘**: [[点击下载]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy) *(提取码: `i9iy`)*
- **Hugging Face**: [[Time-Series-Library]](https://huggingface.co/datasets/thuml/Time-Series-Library)

### 目录结构
下载完成后，确保解压并按如下目录结构放置数据：

```text
DGPRL
├── data/    [将下载的数据集放在这里]
│   ├── MSL/
│   │   ├── MSL_distances.csv
│   │   ├── MSL_train.npy
│   │   ├── MSL_test.npy
│   │   └── MSL_test_label.npy
│   ├── PSM/
│   ├── SMAP/
│   ├── SMD/
│   └── SWaT/
├── conf/
├── prepareData.py
├── train.py
└── requirements.txt
```
💡 目录中的 `MSL_distances.csv`（及其他数据集对应的距离/拓扑文件）
是由`prepareData.py`中的`granger_binary_graph`函数自动计算生成。

## ⚙️ 2. 环境安装
```bash
pip install -r requirements.txt
```
## 🚀 3. 代码运行指南
### 第一步：数据准备与样本构建
运行以下脚本完成数据的预处理和图结构构建：

```bash
# 针对各数据集的指定配置准备
python prepareData.py --config conf/MSL_construct_samples.conf
python prepareData.py --config conf/PSM_construct_samples.conf
python prepareData.py --config conf/SMAP_construct_samples.conf
python prepareData.py --config conf/SMD_construct_samples.conf
python prepareData.py --config conf/SWaT_construct_samples.conf
```

### 第二步：模型训练与测试
使用对应的配置文件运行训练脚本：
```bash
# 针对各数据集的指定配置训练
python train.py --config conf/MSL_1dim_100.conf
python train.py --config conf/PSM_1dim_100.conf
python train.py --config conf/SMAP_1dim_100.conf
python train.py --config conf/SMD_1dim_100.conf
python train.py --config conf/SWaT_1dim_100.conf
```