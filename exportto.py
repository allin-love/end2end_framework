import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from camera.camera_zernike_axial import BaseCamera
from util.refractive_index import refractive_index_glass_bk7 # 假设基底是 BK7

# ================= 配置区域 =================
# 1. 模型路径 (请修改为你训练好的 ckpt 文件路径)
ckpt_path = "training_logs/Learned_flatscope/version_2/checkpoints/epoch=47-validation/image_loss=0.0000743.ckpt" 
# 依照 ckpt 名称创建独立输出目录，避免不同模型互相覆盖
ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]

# 2. 物理参数 (必须与训练时完全一致!)
params = {
    'image_size': 384,
    'sensor_diameter': 5.5e-3,
    'lens_diameter': 3.45e-3,
    'camera_pixel_pitch': 1.0e-6,  # 务必与修改后的训练代码一致
    'd1': 65e-3,
    'd2': 13.59e-3,
    'num_polynomials': 100,
    'require_grad': False
}
wavelength = 532e-9
material_func = refractive_index_glass_bk7 # 使用 BK7 玻璃折射率
# ===========================================

def export():
    # 1. 加载模型
    print(f"正在加载模型: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    zernike_coeffs = checkpoint['state_dict']['camera.optim_param']
    
    # 2. 重建相位
    camera = BaseCamera(**params)
    camera.optim_param.data = zernike_coeffs
    
    # 获取连续相位 (弧度)
    # phase_bias 返回 [1, 1, H, W], 需要 squeeze
    continuous_phase = camera.phase_bias().squeeze().detach().numpy()
    
    # 3. 相位包裹 (Wrap) 到 [0, 2pi]
    wrapped_phase = np.mod(continuous_phase, 2 * np.pi)
    
    # 4. 二值化 (Binarization)
    # 策略：大于 pi 的设为 pi (高台阶)，小于 pi 的设为 0 (低台阶)
    # 注意：这会引入量化误差，建议在 ZEMAX 中验证二值化后的效果
    binary_phase = np.where(wrapped_phase > np.pi, np.pi, 0.0)
    
    # 5. 计算物理高度 (Sag)
    # 公式: Phase = k * (n - 1) * h  =>  h = Phase * lambda / (2 * pi * (n-1))
    n_ref = material_func(wavelength)
    delta_n = n_ref - 1.0
    
    # 理论二元台阶高度
    h_step = wavelength / (2 * delta_n)
    print(f"材料折射率 (BK7 @ 532nm): {n_ref:.4f}")
    print(f"二元面台阶理论高度: {h_step*1e6:.4f} um")
    
    # 生成高度图 (单位: mm, 方便 ZEMAX 读取)
    height_map_mm = np.where(binary_phase > 0, h_step, 0.0) * 1000.0
    
    # 6. 导出为 ZEMAX Grid Sag (.dat)
    output_dir = os.path.join("fabrication_output", ckpt_tag)
    os.makedirs(output_dir, exist_ok=True)
    dat_path = os.path.join(output_dir, "binary_doe_zemax.dat")
    
    Nx, Ny = height_map_mm.shape
    dx_mm = params['camera_pixel_pitch'] * 1000.0
    
    with open(dat_path, 'w') as f:
        # 写入 ZEMAX 表头: Nx Ny dx dy unit(0=mm) x_decenter y_decenter
        f.write(f"{Nx} {Ny} {dx_mm} {dx_mm} 0 0 0\n")
        np.savetxt(f, height_map_mm, fmt='%.8e')
        
    print(f"ZEMAX 文件已导出: {dat_path}")
    
    # 7. 导出为 Numpy 格式 (方便 Python 查看)
    np.save(os.path.join(output_dir, "height_map.npy"), height_map_mm)
    
    # 8. 绘图检查
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(wrapped_phase, cmap='twilight')
    plt.title(f"Continuous Phase (Optimized)\nZernike Order={params['num_polynomials']}")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(height_map_mm, cmap='gray')
    plt.title("Binary Height Map (for Fabrication)")
    plt.colorbar(label='Height (mm)')
    plt.savefig(os.path.join(output_dir, "phase_preview.png"))
    plt.show()

if __name__ == "__main__":
    export()