# KITTI to YOLOv5 Converter 🚀

这是一个专为 **YOLOv5** 优化的 KITTI 数据集转换工具。

## 🌟 核心特性
- **自动分集**：按 8:2 自动划分训练/验证集。
- **类别合并**：`Car/Van` 合并为 `vehicle`，`Pedestrian/Sitting` 合并为 `pedestrian`。
- **可视化验证**：生成 `verify_*.jpg` 确认标注框位置。

## 🚀 快速开始
1. **安装依赖**：
   `pip install opencv-python tqdm pyyaml`

2. **转换数据**：
   `python kitti_to_yolo.py --kitti /你的KITTI路径 --output ./data/kitti_yolo`

3. **开始训练 (YOLOv5)**：
   `python train.py --img 640 --data ./data/kitti_yolo/kitti.yaml --weights yolov5s.pt`

## 📊 类别 ID
- 0: vehicle (Car, Van)
- 1: truck
- 2: pedestrian
- 3: cyclist
