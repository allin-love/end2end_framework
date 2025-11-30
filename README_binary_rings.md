# 同心圆二元相位板 (Binary Phase Plate) 使用指南

## 📌 概述

相比 Zernike 多项式参数化，同心圆二元相位板具有以下优势：
- ✅ **更适合光刻加工**：相位仅为 0 或 π，结构清晰
- ✅ **便于制造**：直接输出环带半径参数
- ✅ **参数物理意义明确**：每个参数对应一个环带的位置

## 🏗️ 架构说明

### 1. 新增文件

- `camera/camera_binary_rings.py` - 同心圆二元相位板相机模型
- `exportto_binary_rings.py` - 专用导出脚本
- `README_binary_rings.md` - 本文档

### 2. 核心修改

**`lightning_flatscope.py`** (第 7、154-166 行)：
```python
# 导入新模块
from camera import camera_binary_rings

# 在 __build_model() 中选择模型
camera_recipe_rings = camera_recipe.copy()
camera_recipe_rings['num_rings'] = camera_recipe_rings.pop('num_polynomials')
self.camera = camera_binary_rings.BinaryRingsCamera(**camera_recipe_rings, require_grad=optimize_optics)
```

## 🚀 使用方法

### Step 1: 训练模型

训练脚本已自动切换到同心圆二元相位板模型。运行：

```powershell
conda activate doe
python lightning_trainer.py
```

**关键参数**：
- `num_polynomials=100` 会被映射为 `num_rings=100` (100 个环带)
- 训练过程会优化环带半径参数
- checkpoint 保存在 `training_logs/Learned_flatscope/`

### Step 2: 导出加工文件

使用专用导出脚本：

```powershell
python exportto_binary_rings.py
```

**配置 `exportto_binary_rings.py` 中的参数**：
```python
ckpt_path = "training_logs/.../epoch=XX.ckpt"  # 修改为你的 ckpt 路径
params['num_rings'] = 100  # 与训练时一致
```

### Step 3: 输出文件

导出后会在 `fabrication_output/<ckpt_name>/` 生成：

1. **`binary_rings_zemax.dat`** - ZEMAX Grid Sag 格式
   - 可直接导入 ZEMAX OpticStudio
   - 表面类型选择 "Grid Sag"

2. **`ring_radii_specification.txt`** - 环带参数表
   ```
   # 环带编号, 归一化半径, 物理半径(mm), 相位值(rad)
     1, 0.100000, 0.172500, 0.0000
     2, 0.195000, 0.336375, 3.1416
     3, 0.285000, 0.491625, 0.0000
     ...
   ```
   发给加工厂商用于制作

3. **`height_map.npy`** - Numpy 格式高度图
4. **`phase_preview.png`** - 相位分布预览
5. **`ring_radii_analysis.png`** - 环带分布分析图

## 🔬 与 Zernike 版本对比

| 特性 | Zernike 多项式 | 同心圆二元相位板 |
|------|---------------|----------------|
| **相位连续性** | 连续相位 (0 ~ 2π) | 二元相位 (0 或 π) |
| **参数类型** | Zernike 系数 (无物理意义) | 环带半径 (物理位置) |
| **加工难度** | 需要多级灰度刻蚀 | 单次二元光刻即可 |
| **加工精度要求** | 高 (需精确控制相位) | 相对较低 (只需控制台阶高度) |
| **导出格式** | 需包裹+二值化 | 直接输出 |
| **适用场景** | 实验室原型 | 批量生产 |

## 📐 物理参数说明

### 环带半径参数化

模型使用 **单调递增约束** 的参数化：
```python
delta_r_logits = [Δr₁, Δr₂, ..., Δrₙ]  # 可优化参数
ring_radii = cumsum(softmax(delta_r_logits))  # 恢复半径
```

### 台阶高度计算

理论二元台阶高度：
```
h = λ / (2 * (n - 1))
```

对于 BK7 玻璃 (n ≈ 1.519 @ 532nm)：
```
h = 532nm / (2 * 0.519) ≈ 512.5 nm
```

## 🎨 可视化检查

运行测试脚本查看相位分布：

```powershell
python camera\camera_binary_rings.py
```

生成图像：
- `binary_rings_phase_test.png` - 相位分布
- `binary_rings_psf_test.png` - PSF 效果

## ⚙️ 调整参数

### 增加环带数量

```python
# lightning_trainer.py 或 训练配置中
params['num_polynomials'] = 200  # 会映射为 num_rings=200
```

更多环带 → 更精细的相位控制，但加工复杂度增加

### 修改材料

```python
# exportto_binary_rings.py
from util.refractive_index import refractive_index_glass_xxx
material_func = refractive_index_glass_xxx
```

## 🔧 故障排查

### Q: 训练时报错 "no module named camera_binary_rings"
**A**: 检查 `lightning_flatscope.py` 第 7 行导入语句是否正确

### Q: 导出的相位不是清晰的 0/π 二值
**A**: 确认使用 `exportto_binary_rings.py` 而非原始 `exportto.py`

### Q: ZEMAX 导入后相位错误
**A**: 
1. 检查 ZEMAX 表面类型是否为 "Grid Sag"
2. 确认材料设置为 BK7
3. 验证波长设置为 532nm

## 📞 加工厂商沟通要点

提供以下文件：
1. `ring_radii_specification.txt` - 环带参数表
2. `phase_preview.png` - 相位分布示意图
3. 以下规格信息：
   - 透镜直径: 3.45 mm
   - 台阶高度: ~512.5 nm
   - 环带数量: 100 (或你的设置)
   - 基底材料: BK7 玻璃
   - 工作波长: 532 nm

## 🔄 切换回 Zernike 模型

如需恢复 Zernike 版本，在 `lightning_flatscope.py` 中：

```python
# 注释掉同心圆模型
# self.camera = camera_binary_rings.BinaryRingsCamera(...)

# 取消注释 Zernike 模型
self.camera = camera_zernike_axial.BaseCamera(**camera_recipe, requires_grad=optimize_optics)
```

---

**创建日期**: 2025-11-30  
**适用版本**: end2end_framework v2.0+
