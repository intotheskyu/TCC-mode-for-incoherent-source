import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, fftshift, ifftshift
import math
from scipy.ndimage import zoom, gaussian_filter
from scipy.misc import ascent
from Optlab import (
    calculate_tcc_modes_with_source,
    simulate_imaging_TCCmodes,
    generate_pattern,
    generate_source,
    save_results
)

# 0.系统参数初始化
print('--- 0.参数初始化 ---')
# 硬件参数
wavelength = 532e-9
NA_obj = 0.382
dxd = (326.6e-6 / 3000) * 2
z_led = 17.2e-3
# 仿真参数
calc_grid_size = 128
n_modes_out = 8
size_obj = 1024
run_validation = False
print('--- 1.仿真建模 ---')
# 1. 仿真建模
# 物体建模
# amp_obj = generate_ic_pattern(size=size_obj, num_rects=300, min_w=10, max_w=100)
amp_obj = ascent()
scale = size_obj / amp_obj.shape[0]
amp_obj = zoom(amp_obj, scale)
amp_obj = (amp_obj - amp_obj.min()) / (amp_obj.max() - amp_obj.min())
Phi_obj = amp_obj * np.pi
obj = amp_obj * np.exp(1j * Phi_obj)

# 光源建模
size_source = 512
delta_source = 50e-6
source_data = generate_source(
    size_source=size_source,
    delta_source=delta_source,
    radius=7.6e-3,
    quadrant_mode=3,        # 指定第3象限
    angle=45,
    sigma=250e-6,         # 输入物理平滑尺寸
    intensity=1.0
)

# 2. 生成TCC光瞳模式
print('--- 2.生成光瞳模式 ---')
vals, modes, p_phy, source_k = calculate_tcc_modes_with_source(
    source_data=source_data,
    delta_source=delta_source,
    wavelength=wavelength,
    NA_obj=NA_obj,
    z_led=z_led,
    target_grid_size=size_obj,  # 输出尺寸
    calc_grid_size=calc_grid_size,  # 仿真尺寸
    dxd=dxd,
    n_modes_out=n_modes_out
)

# 3. 拓展光源成像
print("--- 3.拓展光源成像 ---")
img = simulate_imaging_TCCmodes(
    modes=modes,
    values=vals,
    obj=obj)
# 4. 结果展示
print('--- 4.结果展示 ---')
# 面光源分布(x , k)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(source_data, cmap='inferno')
plt.title("Source Distribution in Space field")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(source_k, cmap='inferno')
plt.title("Source Distribution in K field")
plt.axis('off')
plt.tight_layout()
plt.savefig(
    'images/result_sources_modes.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)
plt.show()

# 绘制模态图
N_mode = modes.shape[0]
cols = 4
rows = math.ceil(N_mode / cols)
plt.figure(figsize=(3 * cols, 3 * (rows + 1)))
# 各模态能量曲线
plt.subplot(rows + 1, 1, 1)
plt.semilogy(vals, '.-')
# plt.plot(vals, '.-')
plt.axvline(x=N_mode, color='r', linestyle='--', label=f'Cutoff={N_mode}')
plt.title("Eigenvalue Decay")
plt.legend()
plt.grid(True)
# 各模态分布
for i in range(N_mode):
    plt.subplot(rows + 1, cols, cols + i + 1)
    p_mode = np.abs(modes[i])
    plt.imshow(p_mode, cmap='inferno')
    plt.title(f"Mode {i}\n(Val = {vals[i]:.4f})")
    plt.axis('off')
plt.tight_layout()
plt.savefig(
    'images/result_TCC_modes.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)
plt.show()

# 拓展光源成像结果
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.imshow(amp_obj ** 2, cmap='gray')
plt.title("Intensity of object( I = Amp^2 )")
plt.axis('off')
plt.subplot(2, 2, 2)
obj_fft = fftshift(fft2(ifftshift(obj)))
plt.imshow(np.log10(abs(obj_fft) + 1e-4), cmap='magma')
plt.title("K field of object")
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(img, cmap='gray')
plt.title("Intensity of image")
plt.axis('off')
plt.subplot(2, 2, 4)
img_fft = fftshift(fft2(ifftshift(np.sqrt(img))))
plt.imshow(np.log10(abs(img_fft) + 1e-4), cmap='magma')
plt.title("K field of image")
plt.axis('off')
plt.tight_layout()
plt.savefig(
    'images/result_obj_and_img.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.1,
    transparent=False
)
plt.show()

# 数据保存
print('--- 5.数据保存 ---')
save_dir = './images/result'
save_results(obj, save_dir=save_dir, save_name='obj')
save_results(img, save_dir=save_dir, save_name='img')
