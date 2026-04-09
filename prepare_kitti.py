#!/usr/bin/env python3
"""
验证版 KITTI to YOLO 转换脚本
基于 https://github.com/joelibaceta/kitti-to-yolo 和官方KITTI格式验证
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict
import cv2
import yaml
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 关键配置（与开源案例一致）==========

# KITTI到YOLO的类别映射
# 注意：YOLOv5要求class_id从0开始连续
CLASS_MAP = {
    'Car': 0,
    'Van': 0,           # 合并为vehicle
    'Truck': 1,
    'Pedestrian': 2,
    'Person_sitting': 2, # 合并为pedestrian
    'Cyclist': 3,
    # 明确忽略的类别
    'DontCare': None,
    'Misc': None,
    'Tram': None,       # 可选：设为4如果数据充足
}

YOLO_NAMES = ['vehicle', 'truck', 'pedestrian', 'cyclist']

# KITTI图片标准尺寸（用于验证）
KITTI_WIDTH = 1242
KITTI_HEIGHT = 375


def parse_kitti_label_strict(label_path, img_w, img_h):
    """
    严格遵循KITTI格式规范解析
    KITTI格式: type truncated occluded alpha bbox(4) dimensions(3) location(3) rotation_y
    """
    labels = []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) < 15:
            logger.warning(f"{label_path.name}:{line_num} 字段不足({len(parts)}<15)，跳过")
            continue
        
        class_name = parts[0]
        
        # 类别过滤（关键！）
        if class_name not in CLASS_MAP:
            continue
        if CLASS_MAP[class_name] is None:
            continue
        
        # 解析bbox (left, top, right, bottom)
        try:
            left, top, right, bottom = map(float, parts[4:8])
        except ValueError:
            continue
        
        # 严格边界检查（KITTI坐标系原点在左上角）
        left = max(0.0, min(left, img_w))
        top = max(0.0, min(top, img_h))
        right = max(0.0, min(right, img_w))
        bottom = max(0.0, min(bottom, img_h))
        
        # 跳过无效框
        if right <= left or bottom <= top:
            continue
        
        # 转换为YOLO格式（归一化）
        x_center = ((left + right) / 2.0) / img_w
        y_center = ((top + bottom) / 2.0) / img_h
        width = (right - left) / img_w
        height = (bottom - top) / img_h
        
        # 最终裁剪到[0,1]（浮点精度安全）
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))
        
        # 跳过归一化后无效的框
        if width <= 0 or height <= 0:
            continue
        
        class_id = CLASS_MAP[class_name]
        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return labels


def verify_conversion(sample_img_path, sample_lbl_path):
    """
    可视化验证转换结果
    """
    img = cv2.imread(str(sample_img_path))
    if img is None:
        return False
    
    h, w = img.shape[:2]
    
    with open(sample_lbl_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            return False
        
        cls, xc, yc, bw, bh = map(float, parts)
        
        # 反算像素坐标
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)
        
        # 绘制验证
        color = (0, 255, 0) if int(cls) == 0 else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, str(int(cls)), (x1, y1-5), 0, 0.6, color, 2)
    
    # 保存验证图
    vis_path = sample_img_path.parent / f"verify_{sample_img_path.stem}.jpg"
    cv2.imwrite(str(vis_path), img)
    logger.info(f"验证图已保存: {vis_path}")
    return True


def prepare_dataset(kitti_path, output_path, split_ratio=(0.8, 0.2)):
    """
    主流程
    """
    kitti_path = Path(kitti_path).resolve()
    output_path = Path(output_path).resolve()
    
    # 验证KITTI结构
    img_dir = kitti_path / 'training' / 'image_2'
    lbl_dir = kitti_path / 'training' / 'label_2'
    
    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"KITTI目录结构错误: 需要 {img_dir} 和 {lbl_dir}")
    
    # 创建输出结构
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 获取匹配的文件对
    img_files = sorted(img_dir.glob('*.png'))
    valid_pairs = []
    
    for img_path in img_files:
        lbl_path = lbl_dir / (img_path.stem + '.txt')
        if lbl_path.exists():
            valid_pairs.append((img_path, lbl_path))
    
    logger.info(f"找到 {len(valid_pairs)} 个有效样本对")
    
    # 划分数据集
    random.seed(42)
    random.shuffle(valid_pairs)
    
    n = len(valid_pairs)
    n_train = int(n * split_ratio[0])
    
    splits = {
        'train': valid_pairs[:n_train],
        'val': valid_pairs[n_train:]
    }
    
    # 处理每个split
    for split_name, pairs in splits.items():
        logger.info(f"\n处理 [{split_name}] ({len(pairs)} 样本)...")
        
        for img_path, lbl_path in tqdm(pairs):
            # 读取图片获取实际尺寸
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # 复制图片
            dst_img = output_path / 'images' / split_name / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # 转换标签
            yolo_labels = parse_kitti_label_strict(lbl_path, w, h)
            
            # 保存标签（空文件也保存）
            dst_lbl = output_path / 'labels' / split_name / (img_path.stem + '.txt')
            with open(dst_lbl, 'w') as f:
                f.write('\n'.join(yolo_labels))
    
    # 生成YAML（使用绝对路径，与JLIN77案例一致）
    yaml_content = {
        'path': str(output_path),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/val',  # KITTI通常没有标准测试集标签
        'nc': len(YOLO_NAMES),
        'names': YOLO_NAMES
    }
    
    yaml_path = output_path / 'kitti.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"\nYAML配置: {yaml_path}")
    
    # 抽样验证
    logger.info("\n执行抽样验证...")
    sample_split = 'train'
    sample_imgs = list((output_path / 'images' / sample_split).glob('*.png'))[:3]
    
    for img_path in sample_imgs:
        lbl_path = output_path / 'labels' / sample_split / (img_path.stem + '.txt')
        if lbl_path.exists():
            verify_conversion(img_path, lbl_path)
    
    logger.info("\n✅ 完成！请检查生成的 verify_*.jpg 文件确认转换正确")
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti', default='../kitti', help='KITTI路径')
    parser.add_argument('--output', default='./data/kitti_yolo', help='输出路径')
    args = parser.parse_args()
    
    prepare_dataset(args.kitti, args.output)
