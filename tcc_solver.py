import os
import math

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.ndimage import zoom, map_coordinates, gaussian_filter
from scipy.misc import ascent


def calculate_tcc_modes_with_source(
        source_data,
        delta_source,
        wavelength=0.532e-9,
        NA_obj=0.382,
        z_led=17.2e-3,
        target_grid_size=1024,
        calc_grid_size=128,
        dxd = 0.217e-6,
        n_modes_out=5
):
    # 参数设置
    # 1. 系统参数定义
    # 定义成像光瞳
    k_cutoff = NA_obj / wavelength
    # 定义 K 平面
    k_max = 1/(2*dxd)
    k_axis = np.linspace(-k_max, k_max, calc_grid_size)
    KX, KY = np.meshgrid(k_axis, k_axis)
    K_R = np.sqrt(KX ** 2 + KY ** 2)
    pupil_small = (K_R <= k_cutoff).astype(float)
    # 定义 NA 平面
    NA_grid = K_R * wavelength
    valid_mask = NA_grid < 1.0
    # 定义 光源空间平面
    R_grid_source = np.zeros_like(NA_grid)
    with np.errstate(invalid='ignore', divide='ignore'):
        tan_theta = NA_grid[valid_mask] / np.sqrt(1 - NA_grid[valid_mask] ** 2)
        R_grid_source[valid_mask] = z_led * tan_theta
    with np.errstate(invalid='ignore', divide='ignore'):
        cos_phi = np.divide(KX, K_R, out=np.zeros_like(KX), where=K_R != 0)
        sin_phi = np.divide(KY, K_R, out=np.zeros_like(KY), where=K_R != 0)
        X_grid_source = R_grid_source * cos_phi
        Y_grid_source = R_grid_source * sin_phi
    # 将光源距离平面转化为source_data索引
    src_h, src_w = source_data.shape
    center_y, center_x = src_h // 2, src_w // 2
    # index = physical_pos / pixel_size + center_index
    map_x = X_grid_source / delta_source + center_x
    map_y = Y_grid_source / delta_source + center_y
    # 将索引映射到k平面
    k_source_intensity = map_coordinates(
        source_data,
        [map_y.ravel(), map_x.ravel()],
        order=1,
        mode='constant',
        cval=0.0
    ).reshape(calc_grid_size, calc_grid_size)
    # 防止有效光瞳之外的点被错误映射
    k_source_intensity[~valid_mask] = 0.0

    source_energy = np.sum(k_source_intensity)
    if source_energy > 0:
        k_source_intensity = k_source_intensity/(source_energy + 1e-10)  # 归一化


    # 估计最大照明NA
    visible_region = k_source_intensity > 1e-4
    if np.any(visible_region):
        max_kr_source = np.max(K_R[visible_region])
        print(f"估计最大照明 NA ≈ {max_kr_source * wavelength:.4f}")

    # 2. 构建矩阵 A
    source_indices = np.where(k_source_intensity > 1e-6)
    n_source_points = len(source_indices[0])
    print(f"有效光源点数: {n_source_points}")
    n_pixels = calc_grid_size * calc_grid_size
    A = np.zeros((n_pixels, n_source_points), dtype=complex)

    dk = k_axis[1] - k_axis[0]
    center = calc_grid_size // 2

    for i in range(n_source_points):
        idx_y, idx_x = source_indices[0][i], source_indices[1][i]
        shift_y = idx_y - center
        shift_x = idx_x - center
        weight = np.sqrt(k_source_intensity[idx_y, idx_x])

        # 使用几何计算位移 (比 roll 更准)
        kx_shifted = KX + shift_x * dk
        ky_shifted = KY + shift_y * dk
        kr_shifted = np.sqrt(kx_shifted ** 2 + ky_shifted ** 2)
        pupil_shifted = (kr_shifted <= k_cutoff).astype(float)

        A[:, i] = pupil_shifted.flatten() * weight

    # 3. SVD 分解
    U, S_vals, Vh = svd(A, full_matrices=False)

    eigenvalues = S_vals ** 2
    #eigenvalues_norm = eigenvalues / np.sum(eigenvalues)
    eigenvalues_norm = eigenvalues
    # 4. 插值放缩
    zoom_factor = target_grid_size / calc_grid_size
    final_modes = []
    pupil_large = zoom(pupil_small, zoom_factor, order=0)

    # 放缩光源
    source_k = zoom(k_source_intensity, zoom_factor, order=0)
    source_k /= np.max(source_k)  # 归一化显示
    # 放缩光瞳模态
    for m in range(min(n_modes_out, U.shape[1])):
        mode_small = U[:, m].reshape((calc_grid_size, calc_grid_size))
        mode_large_real = zoom(np.real(mode_small), zoom_factor, order=3)
        mode_large_imag = zoom(np.imag(mode_small), zoom_factor, order=3)
        mode_large = mode_large_real + 1j * mode_large_imag
        final_modes.append(mode_large)

    return eigenvalues_norm, np.array(final_modes), pupil_large, source_k

# 根据TCC分解生成拓展光源照明成像结果
def simulate_imaging_TCCmodes(modes, values, obj):
    """
    基于 SOCS (Sum of Coherent Systems) 原理仿真成像
    I = Sum( lambda_i * |IFFT( P_i * O_spec )|^2 )
    """
    # 参数初始化
    H, W = obj.shape
    image_intensity = np.zeros((H, W), dtype=float)
    # 物体转换到k域
    obj_fft = fftshift(fft2(ifftshift(obj)))
    # 逐个模态相干成像并累加
    for m in range(len(modes)):
        # 相干成像
        pupil_mode = modes[m]
        weight = values[m]
        coherent_spectrum = obj_fft * pupil_mode
        coherent_field = fftshift(ifft2(ifftshift(coherent_spectrum)))
        # 强度非相干叠加
        intensity_m = np.abs(coherent_field) ** 2
        image_intensity += weight * intensity_m

    return image_intensity

# 根据逐点积分生成拓展光源照明成像结果
def simulate_imaging_traditional(
        source_data,  # 原始光源图像
        delta_source,  # 光源像素大小
        obj,  # 物体复振幅
        wavelength,
        NA_obj,
        z_led,
        dxd,  # 像面像素尺寸
        calc_grid_size,  # TCC使用的低分辨网格尺寸 (如 128)
        target_grid_size  # 最终成像尺寸 (如 1024)
):
    print(f"\n--- 开始传统方法仿真 (K域网格遍历版) ---")

    # ==========================================
    # 第一步：完全复刻 TCC 的光源映射过程
    # (确保输入数据完全一致)
    # ==========================================

    # 1. 定义 K 平面 (低分辨率，用于确定光源分布)
    k_max = 1 / (2 * dxd)
    k_axis_low = np.linspace(-k_max, k_max, calc_grid_size)
    KX_low, KY_low = np.meshgrid(k_axis_low, k_axis_low)
    K_R_low = np.sqrt(KX_low ** 2 + KY_low ** 2)

    # 定义 NA 平面
    NA_grid = K_R_low * wavelength
    valid_mask = NA_grid < 1.0

    # 定义 光源空间平面并映射
    R_grid_source = np.zeros_like(NA_grid)
    with np.errstate(invalid='ignore', divide='ignore'):
        tan_theta = NA_grid[valid_mask] / np.sqrt(1 - NA_grid[valid_mask] ** 2)
        R_grid_source[valid_mask] = z_led * tan_theta
    with np.errstate(invalid='ignore', divide='ignore'):
        cos_phi = np.divide(KX_low, K_R_low, out=np.zeros_like(KX_low), where=K_R_low != 0)
        sin_phi = np.divide(KY_low, K_R_low, out=np.zeros_like(KY_low), where=K_R_low != 0)
        X_grid_source = R_grid_source * cos_phi
        Y_grid_source = R_grid_source * sin_phi

    src_h, src_w = source_data.shape
    center_y, center_x = src_h // 2, src_w // 2
    map_x = X_grid_source / delta_source + center_x
    map_y = Y_grid_source / delta_source + center_y

    # 得到低分辨率的 K 域光源强度
    k_source_intensity = map_coordinates(
        source_data,
        [map_y.ravel(), map_x.ravel()],
        order=1,
        mode='constant',
        cval=0.0
    ).reshape(calc_grid_size, calc_grid_size)

    k_source_intensity[~valid_mask] = 0.0

    # 归一化 (保持与 TCC 一致)
    source_energy = np.sum(k_source_intensity)
    if source_energy > 0:
        k_source_intensity /= (source_energy + 1e-10)

    # ==========================================
    # 第二步：准备高分辨率的成像网格
    # ==========================================
    H, W = target_grid_size, target_grid_size
    # 注意：这里必须保证 K 范围与低分辨的一致，只是点更密
    k_axis_high = np.linspace(-k_max, k_max, H)
    KX_high, KY_high = np.meshgrid(k_axis_high, k_axis_high)

    k_cutoff = NA_obj / wavelength
    obj_fft = fftshift(fft2(ifftshift(obj)))
    total_intensity = np.zeros((H, W), dtype=float)

    # ==========================================
    # 第三步：遍历 K 域网格中的有效点进行积分
    # ==========================================
    # 找到所有有亮度的 K 域坐标索引
    source_indices = np.where(k_source_intensity > 1e-6)
    n_source_points = len(source_indices[0])

    print(f"传统方法：将在 K 空间遍历 {n_source_points} 个有效光源点...")

    dk = k_axis_low[1] - k_axis_low[0]  # 低分辨网格的步长
    center_idx = calc_grid_size // 2

    for i in range(n_source_points):
        # 1. 获取低分辨网格上的索引
        idx_y, idx_x = source_indices[0][i], source_indices[1][i]
        weight = k_source_intensity[idx_y, idx_x]

        # 2. 计算位移量 (Shift)
        # 注意：这里计算的是该点相对于中心的物理 K 位移
        # 逻辑必须与 TCC 中 "shift_x * dk" 保持一致
        shift_x_idx = idx_x - center_idx
        shift_y_idx = idx_y - center_idx

        shift_kx_val = shift_x_idx * dk
        shift_ky_val = shift_y_idx * dk

        # 3. 在高分辨网格上生成移动后的光瞳
        # TCC 逻辑: pupil = ( (KX + shift)^2 ... ) <= cutoff
        # 我们在高分辨网格上复现这个逻辑
        kx_shifted_high = KX_high + shift_kx_val
        ky_shifted_high = KY_high + shift_ky_val
        kr_shifted_high_sq = kx_shifted_high ** 2 + ky_shifted_high ** 2

        pupil_shifted_high = (kr_shifted_high_sq <= k_cutoff ** 2).astype(float)

        # 4. 相干成像 (IFFT)
        # E = IFFT( O(k) * P_shifted(k) )
        coherent_spectrum = obj_fft * pupil_shifted_high
        coherent_field = fftshift(ifft2(ifftshift(coherent_spectrum)))

        # 5. 强度叠加
        # I += |E|^2 * intensity
        total_intensity += (np.abs(coherent_field) ** 2) * weight

        if i % 10 == 0:
            print(f"\rProgress: {i}/{n_source_points}", end="")

    print("\n积分方法计算完成。")
    return total_intensity

# 质心拟合法计算成像结果
def simulate_imaging_ideal(
        source_data,  # 原始光源图像
        delta_source,  # 光源像素大小
        obj,  # 物体复振幅
        wavelength,
        NA_obj,
        z_led,
        dxd,  # 像面像素尺寸
        target_grid_size  # 最终成像尺寸 (如 1024)
):
    print(f"\n--- 开始质心近似方法仿真 (单点相干) ---")

    # 1. 计算光源总能量
    total_energy = np.sum(source_data)
    if total_energy == 0:
        return np.zeros((target_grid_size, target_grid_size))

    # 2. 计算光源质心 (Center of Mass)
    # 网格坐标 (indices)
    grid_y, grid_x = np.indices(source_data.shape)

    # 加权平均求质心索引
    center_y_idx = np.sum(grid_y * source_data) / total_energy
    center_x_idx = np.sum(grid_x * source_data) / total_energy

    print(f"光源质心索引: (y={center_y_idx:.2f}, x={center_x_idx:.2f})")

    # 转换为物理坐标 (相对于光源中心)
    src_h, src_w = source_data.shape
    y_phy = (center_y_idx - src_h // 2) * delta_source
    x_phy = (center_x_idx - src_w // 2) * delta_source

    # 3. 计算对应的 K 域位移
    # sin(theta) = r / sqrt(r^2 + z^2)
    dist = np.sqrt(x_phy ** 2 + y_phy ** 2 + z_led ** 2)
    sin_theta_x = x_phy / dist
    sin_theta_y = y_phy / dist

    shift_kx = sin_theta_x / wavelength
    shift_ky = sin_theta_y / wavelength

    print(f"等效 K 域位移: kx={shift_kx:.2e}, ky={shift_ky:.2e}")

    # 4. 准备高分辨率网格
    H, W = target_grid_size, target_grid_size
    k_max = 1 / (2 * dxd)
    k_axis = np.linspace(-k_max, k_max, H)
    KX, KY = np.meshgrid(k_axis, k_axis)

    # 5. 生成光瞳 (应用位移)
    k_cutoff = NA_obj / wavelength
    # 注意符号：光瞳中心要移动到 shift_kx 位置
    KX_shifted = KX - shift_kx
    KY_shifted = KY - shift_ky

    pupil = (KX_shifted ** 2 + KY_shifted ** 2 <= k_cutoff ** 2).astype(float)

    # 6. 单次相干成像
    obj_fft = fftshift(fft2(ifftshift(obj)))
    coherent_spectrum = obj_fft * pupil
    coherent_field = fftshift(ifft2(ifftshift(coherent_spectrum)))

    # 7. 转换为强度并应用总能量
    # I = |E|^2 * Total_Energy
    img_centroid = (np.abs(coherent_field) ** 2) * total_energy

    return img_centroid