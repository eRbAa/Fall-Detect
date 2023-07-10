## 基于Ultralytics训练的行人跌倒检测模型

### 模型介绍

YOLOv8 是来自 Ultralytics 的最新的基于 YOLO 的对象检测模型系列，提供最先进的性能。

### 模型结构

特征提取部分采用了一种名为CSPDarknet的网络结构，它是一种基于Darknet的改进版本。CSPDarknet采用了Cross Stage Partial Network（CSP）结构，将网络分为两个部分，每个部分都包含多个残差块。这种结构可以有效地减少模型的参数量和计算量，同时提高特征提取的效率。

目标检测部分采用了一种名为YOLOv4-Head的检测头结构。该结构包含了多个卷积层和池化层，用于对特征图进行处理和压缩。然后，通过多个卷积层和全连接层，将特征图转换为目标检测结果。YOLOv8采用了一种基于Anchor-Free的检测方式，即直接预测目标的中心点和宽高比例，而不是预测Anchor框的位置和大小。这种方式可以减少Anchor框的数量，提高检测速度和精度。

### 数据集

 推荐使用一个[行人跌倒数据集](https://download.csdn.net/download/weixin_45921929/87404296?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168871052416800215067805%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=168871052416800215067805&biz_id=1&utm_medium=distribute.pc_search_result.none-task-download-2~all~first_rank_ecpm_v1~rank_v31_ecpm-6-87404296-null-null.142^v88^control_2,239^v2^insert_chatgpt&utm_term=%E8%A1%8C%E4%BA%BA%E8%B7%8C%E5%80%92%E6%95%B0%E6%8D%AE%E9%9B%86%EF%BC%88VOC%E6%A0%BC%E5%BC%8F%EF%BC%89%20&spm=1018.2226.3001.4187.7) ，可用于行人跌倒的检测与识别。 

### 训练及推理

#### 环境配置

训练环境如下：

- Windows11
- cuda:11.1
- pytorch:1.8.1+cu111

准备好环境后，先进入自己带pytorch的虚拟环境，安装 ultralytics 库

` pip install ultralytics` 

#### 训练

Ultralytics 模型的训练程序是train.py ，修改相应路径后，执行该python文件



#### 预训练模型

 在根目录下提供了一个预训练模型以及对应的onnx模型 

 fall_detect.pt  —— `#基于pytorch框架训练出的yolo预训练模型 ` 

 fall_detect.onnx —— `#由fall_detect.pt转换的onnx模型 ` 

#### 测试

yolo模型用如下命令对新数据进行预测 ，source需要指定为自己的图像路径，或者摄像头（0）。 

`yolo task=detect mode=predict model=fall_detect.pt source=data/images device=0`

|      名称      |        默认值        |                      描述                       |
| :------------: | :------------------: | :---------------------------------------------: |
|     source     | ‘ultralytics/assets’ |               图片或视频的源目录                |
|      save      |        False         |                  是否保存结果                   |
|      show      |        False         |                  是否显示结果                   |
|    save_txt    |        False         |             将结果保存为 .txt 文件              |
|   save_conf    |        False         |            保存带有置信度分数的结果             |
|   save_crop    |        Fasle         |             保存裁剪后的图像和结果              |
|      conf      |         0.3          |                   置信度阈值                    |
|  hide_labels   |        False         |                    隐藏标签                     |
|   hide_conf    |        False         |                 隐藏置信度分数                  |
|   vid_stride   |        False         |                  视频帧率步幅                   |
| line_thickness |          3           |               边界框厚度（像素）                |
|   visualize    |        False         |                 可视化模型特征                  |
|    augment     |        False         |             将图像增强应用于预测源              |
|  agnostic_nms  |        False         |                类别不可知的 NMS                 |
|  retina_masks  |        False         |              使用高分辨率分割蒙版               |
|    classes     |         null         | 只显示某几类结果，如class=0, 或者 class=[0,2,3] |



#### 推理

我提供了基于ONNXruntime(ORT)的推理程序，版本依赖： 

onnxruntime>= 1.14.0 

#### ORT

TL.py 是基于ORT的的推理程序

### 性能和准确率数据 

 测试数据使用的是行人跌倒数据集

|   Engine    |    Model Path    | Model Format | Accuracy(%) |
| :---------: | :--------------: | :----------: | :---------: |
| ONNXRuntime | fall_detect.onnx |     onnx     |    83.5     |

### 源码仓库及问题反馈 

https://github.com/eRbAa/Fall-Detect

###  参考 

https://blog.csdn.net/weixin_45921929/article/details/128673338

https://blog.csdn.net/qq_40553788/article/details/130666321