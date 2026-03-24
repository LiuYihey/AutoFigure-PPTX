"""
增强检测模块：补充SAM3遗漏的视觉元素

策略：
1. 颜色聚类检测：找出图中的独立色块区域
2. 边缘检测：识别具有明显边界的对象
3. 连通组件分析：找出独立的视觉区域
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple


def detect_color_regions(
    image: Image.Image,
    min_area: int = 500,
    max_area: int = 50000,
) -> List[Dict]:
    """
    基于颜色聚类检测独立区域
    
    Args:
        image: PIL图像
        min_area: 最小区域面积
        max_area: 最大区域面积
        
    Returns:
        检测到的区域列表 [{"x1": int, "y1": int, "x2": int, "y2": int, "area": int}]
    """
    # 转换为OpenCV格式
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 转换到LAB色彩空间（更适合颜色聚类）
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    
    # 使用自适应阈值分割
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 多尺度边缘检测
    edges = cv2.Canny(gray, 50, 150)
    
    # 形态学操作：闭运算连接边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 过滤太小或太大的区域
        if area < min_area or area > max_area:
            continue
            
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 过滤长宽比异常的区域（可能是噪声）
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.1 or aspect_ratio > 10:
            continue
        
        regions.append({
            "x1": int(x),
            "y1": int(y),
            "x2": int(x + w),
            "y2": int(y + h),
            "area": int(area),
            "score": 0.6,  # 默认置信度
        })
    
    return regions


def detect_isolated_objects(
    image: Image.Image,
    background_color_threshold: int = 30,
    min_area: int = 500,
) -> List[Dict]:
    """
    检测与背景颜色差异明显的独立对象
    
    Args:
        image: PIL图像
        background_color_threshold: 背景颜色差异阈值
        min_area: 最小区域面积
        
    Returns:
        检测到的对象列表
    """
    img_array = np.array(image)
    
    # 估计背景颜色（使用图像四角的平均颜色）
    h, w = img_array.shape[:2]
    corners = [
        img_array[0:10, 0:10],
        img_array[0:10, w-10:w],
        img_array[h-10:h, 0:10],
        img_array[h-10:h, w-10:w],
    ]
    bg_color = np.mean([np.mean(corner, axis=(0, 1)) for corner in corners], axis=0)
    
    # 计算每个像素与背景的颜色距离
    color_diff = np.sqrt(np.sum((img_array - bg_color) ** 2, axis=2))
    
    # 二值化
    mask = (color_diff > background_color_threshold).astype(np.uint8) * 255
    
    # 形态学操作去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 查找连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    objects = []
    for i in range(1, num_labels):  # 跳过背景（label 0）
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area < min_area:
            continue
        
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        objects.append({
            "x1": int(x),
            "y1": int(y),
            "x2": int(x + w),
            "y2": int(y + h),
            "area": int(area),
            "score": 0.65,
        })
    
    return objects


def merge_with_sam_results(
    sam_boxes: List[Dict],
    supplementary_boxes: List[Dict],
    iou_threshold: float = 0.3,
) -> List[Dict]:
    """
    将补充检测结果与SAM结果合并，去除重复
    
    Args:
        sam_boxes: SAM3检测的boxes
        supplementary_boxes: 补充检测的boxes
        iou_threshold: IoU阈值，超过此值认为是重复
        
    Returns:
        合并后的boxes列表
    """
    def calculate_iou(box1: Dict, box2: Dict) -> float:
        x1_inter = max(box1["x1"], box2["x1"])
        y1_inter = max(box1["y1"], box2["y1"])
        x2_inter = min(box1["x2"], box2["x2"])
        y2_inter = min(box1["y2"], box2["y2"])
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
        area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
        
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    # 从SAM boxes开始
    merged = sam_boxes.copy()
    
    # 检查每个补充box是否与现有box重复
    for supp_box in supplementary_boxes:
        is_duplicate = False
        
        for existing_box in merged:
            iou = calculate_iou(supp_box, existing_box)
            if iou > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            # 添加标记表示这是补充检测的
            supp_box["source"] = "supplementary"
            merged.append(supp_box)
    
    return merged


def enhance_sam_detection(
    image: Image.Image,
    sam_boxes: List[Dict],
    enable_color_detection: bool = True,
    enable_object_detection: bool = True,
) -> List[Dict]:
    """
    增强SAM检测结果，补充遗漏的视觉元素
    
    Args:
        image: 原始图像
        sam_boxes: SAM3检测的boxes
        enable_color_detection: 是否启用颜色区域检测
        enable_object_detection: 是否启用独立对象检测
        
    Returns:
        增强后的boxes列表
    """
    supplementary_boxes = []
    
    if enable_color_detection:
        print("  执行颜色区域检测...")
        color_regions = detect_color_regions(image)
        print(f"    检测到 {len(color_regions)} 个颜色区域")
        supplementary_boxes.extend(color_regions)
    
    if enable_object_detection:
        print("  执行独立对象检测...")
        isolated_objects = detect_isolated_objects(image)
        print(f"    检测到 {len(isolated_objects)} 个独立对象")
        supplementary_boxes.extend(isolated_objects)
    
    # 合并结果
    print(f"  合并检测结果...")
    enhanced_boxes = merge_with_sam_results(sam_boxes, supplementary_boxes)
    
    added_count = len(enhanced_boxes) - len(sam_boxes)
    print(f"  增强完成: {len(sam_boxes)} -> {len(enhanced_boxes)} (新增 {added_count} 个)")
    
    return enhanced_boxes
