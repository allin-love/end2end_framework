"""
导出同心圆二元相位板 (Binary Phase Plate) 用于加工

与 Zernike 版本的主要区别：
1. 相位值仅为 0 或 π (无需包裹)
2. 高度图由清晰的同心环组成
3. 导出环带半径信息，便于加工厂商制作
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from camera.camera_binary_rings import BinaryRingsCamera
from util.refractive_index import refractive_index_glass_bk7

# ================= 配置区域 =================
# 1. 模型路径 (请修改为你训练好的 ckpt 文件路径)
ckpt_path = "training_logs/Learned_flatscope/version_2/checkpoints/epoch=45-validation/image_loss=0.0000742.ckpt" 

# 2. 物理参数 (必须与训练时完全一致!)
params = {
    'image_size': 384,
    'sensor_diameter': 5.5e-3,
    'lens_diameter': 3.45e-3,
    'camera_pixel_pitch': 1.0e-6,
    'd1': 65e-3,
    'd2': 13.59e-3,
    'num_rings': 100,  # 环带数量 (对应训练时的 num_polynomials)
    'require_grad': False
}
wavelength = 532e-9
material_func = refractive_index_glass_bk7
# ===========================================

# 依照 ckpt 名称创建独立输出目录，避免不同模型互相覆盖
ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]

def export():
    # 1. 加载模型
    print(f"正在加载模型: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    ring_params = checkpoint['state_dict']['camera.optim_param']
    
    # 2. 重建同心圆二元相位
    camera = BinaryRingsCamera(**params)
    camera.optim_param.data = ring_params
    
    # 获取二元相位 (已经是 0 或 π, 无需包裹)
    binary_phase = camera.phase_bias().squeeze().detach().numpy()
    
    # 获取环带半径 (归一化值 [0, 1])
    ring_radii_normalized = camera.get_ring_radii().detach().numpy()
    
    # 转换为物理半径 (mm)
    lens_radius_mm = params['lens_diameter'] * 1000.0 / 2.0
    ring_radii_mm = ring_radii_normalized * lens_radius_mm
    
    # 3. 计算物理高度 (Sag)
    n_ref = material_func(wavelength)
    delta_n = n_ref - 1.0
    h_step = wavelength / (2 * delta_n)  # 理论二元台阶高度
    
    print(f"材料折射率 (BK7 @ 532nm): {n_ref:.4f}")
    print(f"二元面台阶理论高度: {h_step*1e6:.4f} um")
    print(f"环带数量: {params['num_rings']}")
    print(f"最外环半径: {ring_radii_mm[-1]:.4f} mm")
    
    # 生成高度图 (单位: mm)
    height_map_mm = np.where(binary_phase > 0, h_step, 0.0) * 1000.0
    
    # 4. 导出为 ZEMAX Grid Sag (.dat)
    output_dir = os.path.join("fabrication_output", ckpt_tag)
    os.makedirs(output_dir, exist_ok=True)
    dat_path = os.path.join(output_dir, "binary_rings_zemax.dat")
    
    Nx, Ny = height_map_mm.shape
    dx_mm = params['camera_pixel_pitch'] * 1000.0
    
    with open(dat_path, 'w') as f:
        # 写入 ZEMAX 表头: Nx Ny dx dy unit(0=mm) x_decenter y_decenter
        f.write(f"{Nx} {Ny} {dx_mm} {dx_mm} 0 0 0\n")
        np.savetxt(f, height_map_mm, fmt='%.8e')
    
    print(f"ZEMAX 文件已导出: {dat_path}")
    
    # 5. 导出环带半径信息 (用于加工厂商参考)
    radii_txt_path = os.path.join(output_dir, "ring_radii_specification.txt")
    with open(radii_txt_path, 'w') as f:
        f.write("# 同心圆二元相位板加工参数\n")
        f.write(f"# 总环带数量: {params['num_rings']}\n")
        f.write(f"# 透镜半径: {lens_radius_mm:.4f} mm\n")
        f.write(f"# 台阶高度: {h_step*1e6:.4f} um\n")
        f.write(f"# 波长: {wavelength*1e9:.1f} nm\n")
        f.write(f"# 材料: BK7 (n={n_ref:.4f})\n\n")
        f.write("# 环带编号, 归一化半径, 物理半径(mm), 相位值(rad)\n")
        for i, (r_norm, r_mm) in enumerate(zip(ring_radii_normalized, ring_radii_mm)):
            phase_val = (i % 2) * np.pi
            f.write(f"{i+1:3d}, {r_norm:.6f}, {r_mm:.6f}, {phase_val:.4f}\n")
    
    print(f"环带半径参数已导出: {radii_txt_path}")
    
    # 6. 导出为 Numpy 格式
    np.save(os.path.join(output_dir, "height_map.npy"), height_map_mm)
    np.save(os.path.join(output_dir, "ring_radii_mm.npy"), ring_radii_mm)
    
    # 7. 绘图检查
    fig = plt.figure(figsize=(16, 5))
    
    # 7.1 二元相位分布
    plt.subplot(1, 3, 1)
    plt.imshow(binary_phase, cmap='twilight', vmin=0, vmax=2*np.pi)
    plt.title(f"Binary Phase Plate\n({params['num_rings']} Rings)")
    cbar = plt.colorbar(label='Phase (rad)')
    cbar.set_ticks([0, np.pi, 2*np.pi])
    cbar.set_ticklabels(['0', 'π', '2π'])
    
    # 7.2 高度图
    plt.subplot(1, 3, 2)
    plt.imshow(height_map_mm, cmap='gray')
    plt.title("Height Map for Fabrication")
    plt.colorbar(label='Height (mm)')
    
    # 7.3 径向剖面
    plt.subplot(1, 3, 3)
    center = binary_phase.shape[0] // 2
    x_axis = (np.arange(binary_phase.shape[1]) - center) * params['camera_pixel_pitch'] * 1000.0
    plt.plot(x_axis, binary_phase[center, :], linewidth=2)
    plt.axhline(y=np.pi, color='r', linestyle='--', linewidth=1, alpha=0.5, label='π')
    plt.axhline(y=0, color='b', linestyle='--', linewidth=1, alpha=0.5, label='0')
    plt.xlabel('Radial Position (mm)')
    plt.ylabel('Phase (rad)')
    plt.title('Phase Cross-section (Central Row)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    preview_path = os.path.join(output_dir, "phase_preview.png")
    plt.savefig(preview_path, dpi=150)
    print(f"预览图已保存: {preview_path}")
    plt.show()
    
    # 8. 额外绘图：环带半径分布
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 8.1 环带半径随环数变化
    ax1.plot(np.arange(1, len(ring_radii_mm)+1), ring_radii_mm, 'o-', markersize=4)
    ax1.set_xlabel('Ring Index')
    ax1.set_ylabel('Physical Radius (mm)')
    ax1.set_title('Ring Radii Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 8.2 相邻环带间距
    ring_spacing = np.diff(ring_radii_mm, prepend=0)
    ax2.plot(np.arange(1, len(ring_spacing)+1), ring_spacing * 1000.0, 'o-', markersize=4, color='orange')
    ax2.set_xlabel('Ring Index')
    ax2.set_ylabel('Ring Spacing (μm)')
    ax2.set_title('Adjacent Ring Spacing')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    radii_plot_path = os.path.join(output_dir, "ring_radii_analysis.png")
    plt.savefig(radii_plot_path, dpi=150)
    print(f"环带分析图已保存: {radii_plot_path}")
    plt.show()
    
    print("\n导出完成！输出文件包括：")
    print(f"  - ZEMAX Grid Sag: {dat_path}")
    print(f"  - 环带参数表: {radii_txt_path}")
    print(f"  - 预览图: {preview_path}")
    print(f"  - 环带分析图: {radii_plot_path}")

if __name__ == "__main__":
    export()
