## 基于ESPCN的图像超分辨率重建

### Installation
```
pip3 install -r requirements.txt
```

### Usage
准备训练数据
```
python3 prepare_data.py
```
训练模型
```
python3 train.py
```
重建
```
python3 rebuild.py
```
评估重建效果
```
python3 psnr.py
```

