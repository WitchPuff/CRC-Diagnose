# 基于ResNet和ViT的医学影像识别

## 环境

数据集：CRC-VAL-HE-7K

前端框架：vue.js

后端框架：flask

## Quick Start

### 训练

```shell
# 训练ResNet并导入已经训练好的权重
python main.py train --model ResNet34 --pretrained True
# 训练ResNet并导入已经训练好的权重
python main.py train --model ViT --pretrained True

# epoch、batch_size、learning_rate等训练参数已经默认设置为40/32/1e-2，可以在命令行修改训练参数
python main.py train --model ResNet34 --epoch 30 --batch_size 16 --learning_rate 0.0001 --pretrained True

# data路径，dataset，log路径，num_classes已经根据训练的数据集默认设置，如不更换数据集无需修改
```

### 预测

```shell
# 输入待预测图片的路径及进行模型选择
python main.py predict --img data/example.tif --model ResNet34
python main.py predict --img data/example.tif --model ViT
```

### Web Demo

#### dashboard

![image-20231010175227043](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202310101752310.png)

#### diagnose
