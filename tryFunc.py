from Diffr_FuncS import RS_diffr
from Diffr_FuncS import Fres_AS_diffr
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 定义物理参数 =====================
lamda = 0.633  # 波长，单位μm
k = 2 * np.pi / lamda  # 波数 (1/μm)
R_max = 5 # 圆孔半径，单位μm

# 采样范围与点数（入射面/观测面采样间隔一致）
x_min, x_max = -20, 20   # x方向范围 (μm)
y_min, y_max = -20, 20   # y方向范围 (μm)
z_min, z_max = 0.1 , 100      # z方向范围 (μm)

Nx = 512  # x方向采样点数
Ny = 512  # y方向采样点数
Nz = 1024  # z方向采样点数

# 生成采样三维格点
x0 = np.linspace(x_min, x_max, Nx)    # 入射面x坐标
y0 = np.linspace(y_min, y_max, Ny)    # 入射面y坐标
dx0 = x0[1] - x0[0]                   # 采样间隔
z = np.linspace(z_min, z_max, Nz)     # 观测面z坐标

# 创建二维网格
X0, Y0 = np.meshgrid(x0, y0)


# ===================== 2. 生成圆孔振幅分布 =====================
def generate_RoundHole(X, Y, lamda, R_max):
    """生成01型圆孔"""
    r = np.sqrt(X**2 + Y**2)           # 二维径向距离
    amp = np.zeros_like(X)              # 初始化透振振幅
  
    # 遮光条件：r < R_max = 5μm
    mask_shade = r < R_max
    amp[mask_shade] = 1.0
    amp = np.where(np.isnan(amp), 1, amp)
    return amp

# 生成圆孔振幅分布 U(x0, y0, 0)
U0 = generate_RoundHole(X0, Y0, lamda, R_max)
U0 = U0.astype(np.complex64)

plt.figure(figsize=(6, 5))
plt.imshow(np.real(U0), cmap='gray', extent=[x_min, x_max, y_min, y_max], origin='lower') 
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.title('Round Hole Initial Amplitude')
plt.show()


# 3. 直接调用函数（自动计算并弹出图窗）
# 演示1：绘制 z-x 截面的演化图
I_xz = Fres_AS_diffr(U0, x0, y0, z, lamda)

# 演示2：如果以后想看 z=50 μm 处的光斑分布，可以这么写：
# I_xy = RS_diffr(U0, x0, y0, 50, lamda)