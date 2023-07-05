<p align=center> <img src="./Figure/cover.png" width = 60%/> </p>


<p align="center"><em>Learning Hierarchical Time Series Data Augmentation Invariances via Contrastive Supervision for Human Activity Recognition (KBS 2023)</em></p>

<p align="center"><a href="https://scholar.google.com.sg/citations?user=zOg9ENIAAAAJ&hl=zh-CN&oi=sra">Dongzhou Cheng</a></p>


## Abstract
Human activity recognition (HAR) using wearable sensors is always a research hotspot in ubiquitous computing scenario, in which feature learning has played a crucial role. Recent years have witnessed outstanding success of contrastive learning in image data, which learns invariant representations by adding contrastive loss to the last layer of deep neural networks. However, the advantages of contrastive loss have been rarely leveraged in time series data for activity recognition. A fundamental obstacle to contrastive learning in HAR is that image-based augmentation could not fit well with sensor data, which raises a critical issue: the distortions induced by augmentation might be further enlarged by intermediate layers of a network and thus severely harm semantic structure of original activity instance. In this paper, taking an inspiration from deeply-supervised learning, we propose a novel approach called Contrastive Supervision by considering “where” to contrast, which aims to learn time series augmentation invariances by forcing positive pairs nearby and negative pairs far apart at different depths of neural network. Our approach can be seen as a generalization of contrastive learning in a deeply-supervised setting, where the contrastive loss is used to supervise the intermediate layers instead of only the last layer, allowing us to effectively leverage label information so as to better fuse the multi-level features. Experiments on popular benchmarks demonstrate that our approach can learn better representations and improve classification accuracy without additional inference cost for various HAR tasks in supervised and semi-supervised learning paradigms.

## Getting Started
1. Git clone the repo
```
git clone https://github.com/cheng-haha/CoS.git
```
2. Requirements
```
pip install -r requirements.txt
```
3. Extract the file to get the following directory tree
```
.
├── common.py
├── configs.py
├── dataset
│   ├── HAR_dataset.py
│   └── ucihar
├── Figure
│   ├── frame.png
│   └── inference_time.png
├── loss
│   └── SupCon.py
├── main.py
├── models
│   ├── Baseline_CNN.py
│   ├── complex
│   ├── CoS_CNN.py
│   └── __init__.py
├── README.md
├── requirements.txt
├── save
│   └── ucihar
├── scripts
│   └── run.sh
├── tree.txt
└── utils
    ├── augmentations.py
    ├── logger.py
    ├── metric.py
    ├── setup.py
    └── train.py

10 directories, 19 files
```
Get required dataset from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php), do data pre-processing by sliding window strategy and split the data into different sets. Other datasets are too large, please contact us by e-mail <chengdongzhou666@qq.com> if you need one of these.
## Run
NOTE: Check in the `config.py` file that your path `dataset_path` is correct. Then you can do
```
bash scripts/run.sh
```
## Citation
```
TODO
```
