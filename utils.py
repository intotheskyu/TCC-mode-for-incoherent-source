import numpy as np
import cv2
import os
from scipy.ndimage import gaussian_filter

def save_results(obj, save_dir, save_name):

    # 将数据保存为npy以及PNG格式
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 保存NPY格式无损数据用于后续处理
    np.save(os.path.join(save_dir, f"{save_name}.npy"), obj)

    print(f"原始数据 (.npy) 已保存至 {save_dir}")
    # 保存PNG格式数据用于展示分析
    print(f"max_obj = {np.max(abs(obj))}")
    obj_uint16 = (np.clip(abs(obj), 0, 1) * 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(save_dir, f"{save_name}.png"), obj_uint16)

def generate_source(
        size_source=512,
        delta_source=50e-6,
        radius=7.6e-3,
        quadrant_mode=0, #[形状参数，描述四分之一圆所在象限]
        angle=0.0,
        sigma=0.0,
        intensity=1.0
):
    """
        生成参数化的面光源/象限光源图案

        Args:
            size_source (int): 图像像素尺寸 (N x N)
            delta_source (float): 单像素物理尺寸 (meters)
            radius (float): 光源的物理半径 (meters)
            quadrant_mode (int): 象限控制开关
                0: 完整圆形光源
                1: 第一象限 (右上, x>=0, y>=0)
                2: 第二象限 (左上, x<=0, y>=0)
                3: 第三象限 (左下, x<=0, y<=0)
                4: 第四象限 (右下, x>=0, y<=0)
            sigma (float): 高斯滤波的标准差，单位为米 (meters)
            intensity (float): 光源强度值

        Returns:
            np.ndarray: 生成的光源二维矩阵
        """
    # 1. 生成物理坐标网格 (Centered)
    y_idx, x_idx = np.ogrid[-size_source // 2:size_source // 2, -size_source // 2:size_source // 2]
    # 转换为物理坐标
    y_phy = y_idx * delta_source
    x_phy = x_idx * delta_source
    # 坐标旋转
    theta = np.deg2rad(angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = x_phy * cos_t + y_phy * sin_t
    y_rot = -x_phy * sin_t + y_phy * cos_t
    # 2. 生成基础圆形掩膜
    # mask_base: (x^2 + y^2 <= R^2)
    mask_base = (x_phy ** 2 + y_phy ** 2 <= radius ** 2)

    # 3. 处理象限截断 (Quadrant Truncation)
    mask_quad = np.zeros_like(mask_base, dtype=bool)

    if quadrant_mode == 0:
        # 全圆模式：所有区域都有效
        mask_quad[:] = True
    elif quadrant_mode == 1:
        # 第一象限: X>=0, Y>=0 (注意图像坐标系y通常向下，但物理坐标系我们通常假设y向上)
        # 这里按照常见的笛卡尔坐标系逻辑:
        mask_quad = (x_rot >= 0) & (y_rot >= 0)
    elif quadrant_mode == 2:
        # 第二象限: X<=0, Y>=0
        mask_quad = (x_rot <= 0) & (y_rot >= 0)
    elif quadrant_mode == 3:
        # 第三象限: X<=0, Y<=0
        mask_quad = (x_rot <= 0) & (y_rot <= 0)
    elif quadrant_mode == 4:
        # 第四象限: X>=0, Y<=0
        mask_quad = (x_rot >= 0) & (y_rot <= 0)
    else:
        raise ValueError("quadrant_mode must be 0, 1, 2, 3, or 4")

    # 4. 组合掩膜并赋值
    final_mask = mask_base & mask_quad

    source_data = np.zeros((size_source, size_source), dtype=float)
    source_data[final_mask] = intensity

    # 5. 物理参数滤波 (Physical Gaussian Blur)
    # 将物理尺寸 (meters) 转换为 像素尺寸 (pixels)
    if sigma > 0:
        sigma_pixel = sigma / delta_source
        source_data = gaussian_filter(source_data, sigma=sigma_pixel)

    return source_data

def generate_pattern(size=1024, num_rects=50, min_w=10, max_w=100):
    """生成类集成电路的随机矩形图案"""
    # 初始化全黑背景
    obj = np.zeros((size, size), dtype=float)

    # 随机种子，保证每次生成一样的图，方便复现
    np.random.seed(42)

    for _ in range(num_rects):
        # 随机生成矩形参数
        w = np.random.randint(min_w, max_w)
        h = np.random.randint(min_w, max_w)
        x = np.random.randint(0, size - w)
        y = np.random.randint(0, size - h)
        val = np.random.uniform(0.5, 1.0)  # 随机透射率

        # 叠加矩形 (模拟多层电路结构)
        obj[y:y + h, x:x + w] = np.maximum(obj[y:y + h, x:x + w], val)

    # 添加一些细线条模拟导线
    for _ in range(30):
        if np.random.rand() > 0.5:  # 横线
            w = np.random.randint(50, 300)
            h = np.random.randint(2, 6)
            x = np.random.randint(0, size - w)
            y = np.random.randint(0, size - h)
        else:  # 竖线
            h = np.random.randint(50, 300)
            w = np.random.randint(2, 6)
            x = np.random.randint(0, size - w)
            y = np.random.randint(0, size - h)
        obj[y:y + h, x:x + w] = 1.0

    return obj