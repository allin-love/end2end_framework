# Binary Rings DOE Optimization Report

## 1. 背景概述
- 目标：利用同心圆二元相位板在 \(-0.5\,\text{mm}\) 到 \(+0.5\,\text{mm}\) 范围内保持近似不变的点扩散函数（PSF），实现 1\,mm 级景深延伸（EDoF）。
- 原始实现：`lightning_flatscope.py` 仅依赖归一化后的 L1 重建损失驱动训练；`camera/camera_binary_rings.py` 则固定 2\,µm 采样网格输出 PSF；验证脚本 `validate_binary_rings.py` 用于展示 phase/PSF 结果。

## 2. 早期失败的原因
1. **PSF 优化目标缺失**  
   - Lightning 模块只优化单深度的重建误差（`image_l1_loss`），不同深度的 PSF 既不约束、也不出现在损失里。算法自然会把最清晰的焦点移动到某一个深度，以降低单一 L1 损失。
2. **焦移 (Focus Shift)**  
   - 从验证图可以看到 \(-0.5\,\text{mm}\) 处 FWHM≈2\,µm，而焦面 (0\,mm) 扩散到 16\,µm。这意味着 DOE 等效地增加了正光焦度，把最佳焦平面整体推到 -0.5\,mm 附近。
3. **采样与 aliasing 风险**  
   - 尽管 DOE 在 0.5\,µm 栅格上仿真，但传感器模型直接把该 PSF 卷积到 1\,µm 像素，缺乏自动 binning/裁剪逻辑。对于较密的外环，这容易造成 alias 或能量泄漏，使优化器得到“虚假”高分。

## 3. 新增/修改内容
1. **损失函数增强 (`lightning_flatscope.py`)**
   - `__compute_loss` 现在接受 `psfs` 并叠加三类正则：
     - `psf_consistency_weight`：利用方差 `variance(psf_z - psf_mean)` 惩罚不同深度的形状差异。
     - `psf_worst_weight`：约束最差深度的 L1 偏差（min-max 思路）。
     - `focus_balance_weight`：根据能量质心 `depth_center` 惩罚焦点整体偏移，可配合 `focus_center_target` 强制保持居中。
   - 新增 `__psf_regularizers`，统一计算上述指标并写入日志（`psf/variance`, `psf/worst_l1`, `psf/depth_center`）。
2. **光学采样/裁剪可配置化**
   - `forward` 中加入 `psf_binning_factor`、`reconstruction_crop`，自动对 PSF 进行 binning 和尺寸匹配，避免 0.5\,µm 栅格直接作用于 1\,µm 像素。
   - 新 CLI 参数：`--psf_window`, `--mask_pixel_pitch`，通过 `camera_binary_rings.BinaryRingsCamera` 注入，确保仿真窗口与训练窗口一致。
   - 训练日志新增 `optics/min_ring_spacing_um`, `optics/max_ring_spacing_um`, `optics/mask_pixel_pitch_um`，方便监控加工可行性与采样比。
3. **相机模块扩展 (`camera/camera_binary_rings.py`)**
   - 构造函数允许外部指定 `mask_pixel_pitch`，从而与 Lightning 端的 binning 策略相匹配。
4. **验证脚本 (`validate_binary_rings.py`)**
   - 增加了 zoom cross-section、环半径物理刻度，以便更快地识别 0/π 交替与焦点偏移（该修改已在前一阶段完成，报告中引用其结论）。

## 4. 新版本的物理原理与优化本质
1. **PSF 不变性作为主导目标**  
   - 通过 `variance` 与 `worst_l1` 正则，优化器被迫在每个深度产生形状相似的 PSF，而不是把能量集中在单一深度。这等价于在设计空间里寻找接近 Bessel / Axicon 行为的解，使景深真正拉长。
2. **焦点质心约束**  
   - `focus_balance_weight` 驱动 PSF 的能量中心回到预期深度（通常是 0 或中间层），避免把 DOE 当作“额外透镜”使用。这样能保证端到端系统仍以指定工作距离为中心。
3. **物理/数值采样匹配**  
   - 通过显式的 `mask_pixel_pitch` 与 `psf_binning_factor`，仿真栅格与传感器像素之间建立固定关系，避免 alias，确保外环场型在传播过程中保持正确的空间频率。这一步是实现真实可加工 DOE 的必要条件。
4. **可调的加工约束监控**  
   - 实时记录环带间距（`optics/min_ring_spacing_um` 等）意味着优化过程始终受到加工可行性的反馈，防止出现 <1\,µm 的不可制造结构。

## 5. 使用建议 / 下一步
1. **训练命令示例**  
   ```powershell
   python lightning_trainer.py ^
       --gpus 1 ^
       --batch_sz 2 ^
       --image_size 384 ^
       --num_polynomials 100 ^
       --psf_consistency_weight 0.5 ^
       --psf_worst_weight 0.2 ^
       --focus_balance_weight 0.05 ^
       --psf_binning_factor 2 ^
       --psf_window 288 ^
       --mask_pixel_pitch 2e-6 ^
       --reconstruction_crop 288
   ```
2. **验证流程**  
   - 训练到合适 epoch 后，使用 `validate_binary_rings.py` 检查 `validation_psf.png` 三个深度的 FWHM 是否接近；若仍有偏差，调高相应正则权重。
3. **导出前检查**  
   - 观察 TensorBoard 中的 `psf/variance` 与 `psf/depth_center`，确认它们稳定下降并趋于居中；只有在满足目标后再运行 `exportto_binary_rings.py`。

通过以上改动，新版本的优化本质从“单点锐化”转变为“多深度 PSF 一致化 + 焦点平衡 + 采样物理匹配”，能更可靠地逼近真实可加工的长景深 DOE 设计。