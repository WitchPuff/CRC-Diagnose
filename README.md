# CRC-Diagnose: 基于ResNet和ViT的医学影像识别
该项目使用PyTorch开发，基于自主实现的ResNet和ViT模型及NCT-CRC-HE-100K数据集，实现了直结肠癌医学图像识别，配备了基于Flask和Vue的用户友好型Web演示。
## Quick Start

### Python环境

```shell
pip install -r requirements.txt
```

### 前端Vue

#### 1、安装 Node.js 和 npm：

首先，确保你的系统上已经安装了 Node.js 和 npm。你可以从 [Node.js 官网](https://nodejs.org/) 下载并安装它们。安装完成后，你可以通过以下命令验证是否成功安装：

```shell
node -v
npm -v
```

#### 2. 安装 Vue CLI：

Vue CLI 是一个用于创建和管理 Vue.js 项目的官方命令行工具。安装 Vue CLI 可以让你更轻松地启动和管理 Vue.js 项目。运行以下命令来安装 Vue CLI：

```shell
npm install -g @vue/cli
```

安装完成后，你可以使用以下命令验证安装：

```shell
vue --version
```

#### 3、运行Vue

```shell
cd gui/frontend
npm run serve
```

### 后端Flask

#### 运行Flask服务器

```shell
python gui.backend.app
```

### 准备数据

在此处查看[数据集](https://zenodo.org/record/1214456)。

下载[CRC-VAL-HE-7K.zip](https://zenodo.org/record/1214456/files/CRC-VAL-HE-7K.zip?download=1)至根目录，也可以下载整个数据集[NCT-CRC-HE-100K](https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip?download=1)。

运行`utils/data.py`，将自动处理`CRC-VAL-HE-7K.zip`，预处理后的数据将分为test（`data_0.pt`, `label_0.pt`)、train、valid存放在`data/`中。

```shell
# if __name__ == '__main__':
#    process('CRC-VAL-HE-7K.zip')
python utils\data.py
```

### 准备模型权重

ResNet34权重已经放在`models/ResNet34_CRC.pt`，在此处下载[ViT权重](https://drive.google.com/file/d/1ju3CiaP4WGBEpF4rCN1TqT18L-5nE56X/view?usp=drive_link)，同样放至`models/ViT_CRC.pt`。

## Usage

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
python main.py predict --img data/adi.tif --model ResNet34
python main.py predict --img data/back.tif --model ViT
```

### Web Demo

#### dashboard

显示两个模型的训练性能数据。

![image-20231010175227043](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202310101752310.png)

#### diagnose

能上传图片、选择模型进行预测。

![image-20231010200111965](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202310102001069.png)

![image-20231010200049544](https://raw.githubusercontent.com/WitchPuff/typora_images/main/img/202310102000689.png)

## Citation

前端使用了该[Vue模板](https://github.com/creativetimofficial/vue-black-dashboard)。
