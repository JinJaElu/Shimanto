# Shimanto
KITTI to YOLOv5 数据转换与训练工具本脚本专门为 YOLOv5 架构设计，用于将 KITTI Object Detection 数据集转换为 YOLO 格式。它集成了标签归一化、类别映射、自动数据集划分以及转换结果可视化验证功能。
🌟 核心特性完全兼容 YOLOv5：生成符合官方标准的 images/、labels/ 结构及 kitti.yaml 配置文件。
智能类别合并：Car + Van -> vehicle (ID: 0)Pedestrian + Person_sitting -> pedestrian (ID: 2)
自动过滤 DontCare、Misc、Tram 等噪声类别。
异常防护：自动处理并过滤 KITTI 标注中的无效框（宽或高 $\le 0$ 的标注）。
可视化验证：转换后自动生成 verify_*.jpg 样本图，确保标注没有偏移。
🛠️ 环境准备在开始之前，请确保你的环境中已安装以下库：
Bash
pip install opencv-python tqdm pyyaml
📂 数据准备请按照 KITTI 官方格式组织原始数据：
Plaintextkitti/
└── training/
    ├── image_2/   # 原始 .png 图片
    └── label_2/   # 原始 .txt 标签
🚀 使用方法1. 执行转换脚本在终端中运行以下命令。脚本会自动完成数据拆分（默认训练集 80%，验证集 20%）：Bashpython kitti_to_yolo.py --kitti ../kitti --output ./data/kitti_yolo
2. 参数说明
--kitti: 原始 KITTI 数据集的根目录路径。
--output: 转换后 YOLO 数据的存储位置（建议直接设在 YOLOv5 项目的 data 目录下）。
🔥 YOLOv5 训练指南转换完成后，你可以直接在 YOLOv5 仓库中启动训练：
1. 验证配置文件脚本生成的 kitti.yaml 内容大致如下：YAMLpath: ./data/kitti_yolo
train: images/train
val: images/val
nc: 4
names: ['vehicle', 'truck', 'pedestrian', 'cyclist']
2. 启动训练由于 KITTI 图像分辨率较高（$1242 \times 375$），建议针对性调整 --img 参数：Bash# 在 yolov5 根目录下执行
python train.py --img 640 --batch 16 --epochs 100 --data ./data/kitti_yolo/kitti.yaml --weights yolov5s.pt
注：如果你希望对远处的行人有更好的检测效果，可以尝试将 --img 设为 1024 或 1280。
