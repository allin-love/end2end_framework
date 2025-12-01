"""Verify binary ring phase-plate checkpoints via phase/PSF inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from camera.camera_binary_rings import BinaryRingsCamera


DEFAULT_PARAMS = {
    "image_size": 384,
    "sensor_diameter": 5.5e-3,
    "lens_diameter": 3.45e-3,
    "camera_pixel_pitch": 1.0e-6,
    "d1": 65e-3,
    "d2": 13.59e-3,
    "num_rings": 100,
    "require_grad": False,
}
DEFAULT_WINDOW = 128

def visualize_phase(camera: BinaryRingsCamera, params: dict, out_dir: Path) -> np.ndarray:
    """可视化相位分布"""
    phase = camera.phase_bias().squeeze().detach().numpy()
    ring_radii = camera.get_ring_radii().detach().numpy()

    fig = plt.figure(figsize=(18, 5))

    # 1. 二维相位图
    ax1 = plt.subplot(1, 4, 1)
    im1 = ax1.imshow(phase, cmap='twilight', vmin=0, vmax=2*np.pi)
    ax1.set_title(f'Binary Phase Plate\n({camera.num_rings} rings)', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1, label='Phase (rad)')
    cbar1.set_ticks([0, np.pi, 2*np.pi])
    cbar1.set_ticklabels(['0', 'π', '2π'])

    center = phase.shape[0] // 2
    x_axis = (np.arange(phase.shape[1]) - center) * params['camera_pixel_pitch'] * 1000.0

    # 2. 全范围截面
    ax2 = plt.subplot(1, 4, 2)
    ax2.plot(x_axis, phase[center, :], linewidth=1, color='blue')
    ax2.set_xlabel('Radial Position (mm)', fontsize=9)
    ax2.set_ylabel('Phase (rad)', fontsize=9)
    ax2.set_title('Phase Cross-section (full)', fontsize=11)
    ax2.set_ylim(-0.2, 3.5)
    ax2.grid(True, alpha=0.3)

    # 3. 局部放大截面
    zoom_mask = np.abs(x_axis) < 0.15
    ax3 = plt.subplot(1, 4, 3)
    ax3.step(x_axis[zoom_mask], phase[center, zoom_mask], where='mid', linewidth=1.2)
    ax3.set_xlabel('Radial Position (mm)', fontsize=9)
    ax3.set_ylabel('Phase (rad)', fontsize=9)
    ax3.set_title('Phase Cross-section (zoom)', fontsize=11)
    ax3.set_ylim(-0.2, 3.5)
    ax3.grid(True, alpha=0.3)

    # 4. 环带半径分布
    ax4 = plt.subplot(1, 4, 4)
    lens_radius_mm = params['lens_diameter'] * 1000.0 / 2.0
    ring_radii_mm = ring_radii * lens_radius_mm
    ax4.plot(np.arange(1, len(ring_radii)+1), ring_radii_mm, 'o-', markersize=3, color='orange')
    ax4.set_xlabel('Ring Index', fontsize=9)
    ax4.set_ylabel('Physical Radius (mm)', fontsize=9)
    ax4.set_title('Optimized Ring Radii', fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = out_dir / "validation_phase.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 相位图已保存: {output_path}")
    
    return ring_radii_mm

def analyze_psf(camera: BinaryRingsCamera, params: dict, out_dir: Path):
    """生成并分析 PSF"""
    camera.delta_z = [-0.5e-3, 0, 0.5e-3]
    camera.obj_offsets_y = [0]
    camera.obj_offsets_x = [0]
    
    print("正在生成 PSF...")
    psfs = camera.gen_psf()
    print(f"PSF shape: {psfs.shape}")
    
    # 可视化不同深度的 PSF
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    depth_labels = ['-0.5mm', 'Focus', '+0.5mm']
    
    for i, (ax, label) in enumerate(zip(axes, depth_labels)):
        psf_img = psfs[0, i].detach().numpy()
        im = ax.imshow(psf_img, cmap='hot')
        ax.set_title(f'PSF @ {label}', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 计算并显示 PSF 指标
        psf_max = psf_img.max()
        psf_sum = psf_img.sum()
        # 计算半峰全宽 (简化版)
        center = psf_img.shape[0] // 2
        profile = psf_img[center, :]
        fwhm_pixels = np.sum(profile > psf_max / 2)
        fwhm_um = fwhm_pixels * params['camera_pixel_pitch'] * 1e6
        
        ax.text(0.02, 0.98, f'Max: {psf_max:.2e}\nFWHM: {fwhm_um:.1f} μm', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)
    
    plt.tight_layout()
    output_path = out_dir / "validation_psf.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ PSF 图已保存: {output_path}")
    
    return psfs

def print_statistics(ring_radii_mm):
    """打印环带统计信息"""
    print("\n" + "="*50)
    print("环带统计信息")
    print("="*50)
    print(f"环带数量: {len(ring_radii_mm)}")
    print(f"最小半径: {ring_radii_mm[0]:.4f} mm")
    print(f"最大半径: {ring_radii_mm[-1]:.4f} mm")
    
    # 计算相邻环带间距
    ring_spacing = np.diff(ring_radii_mm, prepend=0)
    print(f"平均环间距: {ring_spacing.mean()*1000:.2f} μm")
    print(f"最小环间距: {ring_spacing.min()*1000:.2f} μm")
    print(f"最大环间距: {ring_spacing.max()*1000:.2f} μm")
    
    # 检查加工可行性（环间距应 > 1 μm）
    min_spacing_um = ring_spacing.min() * 1000
    if min_spacing_um < 1.0:
        print(f"\n⚠️  警告: 最小环间距 {min_spacing_um:.2f} μm < 1 μm，加工可能困难！")
        print("   建议: 减少环带数量或增大透镜直径")
    else:
        print(f"\n✓ 环间距满足加工要求 (>{min_spacing_um:.2f} μm)")
    
    # 列出前 10 个环带
    print("\n前 10 个环带半径 (mm):")
    for i in range(min(10, len(ring_radii_mm))):
        phase_val = (i % 2) * np.pi
        print(f"  Ring {i+1:2d}: {ring_radii_mm[i]:.6f} mm, Phase: {phase_val:.4f} rad")
    
    print("="*50)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate binary-ring checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint file")
    parser.add_argument(
        "--output-dir",
        default=Path("artifacts/validation"),
        type=Path,
        help="Root directory for validation outputs",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="PSF crop window (must match training)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")

    ckpt_tag = ckpt_path.stem
    out_dir = args.output_dir / ckpt_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("同心圆二元相位板训练结果验证")
    print("="*60)
    print(f"\n加载 checkpoint: {ckpt_path}")
    
    checkpoint = torch.load(str(ckpt_path), map_location='cpu')
    ring_params = checkpoint['state_dict']['camera.optim_param']

    camera = BinaryRingsCamera(**DEFAULT_PARAMS, window_cropsize=args.window)
    camera.optim_param.data = ring_params
    camera.eval()
    
    print(f"✓ 模型加载成功")
    print(f"  - 环带数量: {camera.num_rings}")
    print(f"  - 透镜直径: {DEFAULT_PARAMS['lens_diameter']*1000:.2f} mm")
    print(f"  - 相位板分辨率: {camera.valid_resolution} × {camera.valid_resolution}")
    
    # 3. 可视化相位
    print("\n" + "-"*60)
    print("步骤 1: 可视化优化后的相位分布")
    print("-"*60)
    ring_radii_mm = visualize_phase(camera, DEFAULT_PARAMS, out_dir)
    
    # 4. 打印统计信息
    print_statistics(ring_radii_mm)
    
    # 5. 生成和分析 PSF
    print("\n" + "-"*60)
    print("步骤 2: 生成点扩散函数 (PSF)")
    print("-"*60)
    psfs = analyze_psf(camera, DEFAULT_PARAMS, out_dir)
    
    # 6. 总结
    print("\n" + "="*60)
    print("验证完成！")
    print("="*60)
    print("生成文件:")
    print(f"  - {out_dir / 'validation_phase.png'}")
    print(f"  - {out_dir / 'validation_psf.png'}")
    print("\n下一步:")
    print("  1. 如果效果满意，运行 exportto_binary_rings.py 导出加工文件")
    print("  2. 如果需要改进，调整训练参数继续训练")
    print("="*60)

if __name__ == "__main__":
    main()
