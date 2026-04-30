import numpy as np
import cupy as cp 
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
x0 = cp.linspace(x_min, x_max, Nx)    # 入射面x坐标
y0 = cp.linspace(y_min, y_max, Ny)    # 入射面y坐标
dx0 = x0[1] - x0[0]                   # 采样间隔
z = cp.linspace(z_min, z_max, Nz)     # 观测面z坐标

# 创建二维网格
X0, Y0 = cp.meshgrid(x0, y0)
# 通过将 x0 按行复制得到。它的每一行都一样，代表了网格中每个点对应的 X 坐标
# 如果把 X0 和 Y0 相同位置的元素组合起来，就得到了平面上所有的坐标点
# 便于进行向量化运算，比使用for循坏快得多


# ===================== 2. 生成圆孔振幅分布 =====================
def generate_RoundHole(X, Y, lamda, R_max):
    """生成01型圆孔
        X, Y: 二维坐标网格数组，把 X0 和 Y0 相同位置的元素组合起来，就得到了平面上所有的坐标点
        lamda: 波长
        f: 波带片焦距
        R_max: 最大半径，超过此半径的区域将被截断（单位μm）"""
    r = cp.sqrt(X**2 + Y**2)           # 二维径向距离（二维数组）
    amp = cp.zeros_like(X)              # 初始化透振振幅,全不透光
  
    # 遮光条件：r < R_max = 5μm
    mask_shade = r < R_max
    amp[mask_shade] = 1.0
    amp = cp.where(cp.isnan(amp), 1, amp)
    # cp.where(condition, x, y)，condition满足时返回x，否则返回y。这里将NaN值替换为1，确保振幅数组中没有NaN值。
    return amp

# 生成圆孔振幅分布 U(x0, y0, 0)
U0 = generate_RoundHole(X0, Y0, lamda, R_max)
U0 = U0.astype(cp.complex64)  # 转换为复数类型，方便后续计算

plt.imshow(cp.asnumpy(cp.real(U0))) # 显示圆孔定义是否正确.注意把gpu数据改成numpy


# ===================== 3. RS衍射积分 =====================
intensity = cp.zeros((Nz, Nx))   # 初始化光强数组，只记录y=0截面

# 生成卷积核的二维坐标网格
X_kernel, Y_kernel = cp.meshgrid(x0, y0)

# 对每个z来说，U0_fft是一样的，所以放在循环外面计算一次就好
U0_fft = cp.fft.fft2(U0)   # 不必挪动，否则衍射结果也会跟着挪动，画图的时候也要挪回来

# 遍历每个z位置计算衍射场
for iz in range(Nz):
    z_current = z[iz]   # 当前z位置
  
    if cp.isclose(z_current, 0):
        # z=0时为圆孔平面，场分布等于初始场
        U = U0
    else:
        # 步骤1：计算二维RS卷积核 g(x, y, z) = ∂/∂z [exp(ikr)/r] / (2π)
        r = cp.sqrt(X_kernel**2 + Y_kernel**2 + z_current**2) 
        # 小心不要永成前面的二维r了。此处r也是二维数组
        r = cp.where(r == 0, 1e-12, r)   # 避免数值奇异
        # 偏导数展开：∂/∂z [exp(ikr)/r] = exp(ikr)*(ikz/r^2 - z/r^3)
        kernel = -(1/(2*cp.pi)) * cp.exp(1j * k * r) * (1j * k * z_current / r**2 - z_current / r**3)
        # 步骤2：二维FFT实现卷积
        # U0_fft 已经在循环外面计算好了
        kernel_fft = cp.fft.fft2(cp.fft.ifftshift(kernel))
        # 必须对 kernel 做 ifftshift 对齐 FFT 坐标原点
        # 把卷积核中心（r=0）对应的值放在数组的左上角，这样才能正确地进行频域乘积。
        U_fft = U0_fft * kernel_fft          # 频域乘积
        U = cp.fft.ifft2(U_fft)              # 二维逆FFT回到空域
        U = U * (dx0**2)                     
        # FFT 只是简单地把矩阵里的数字相乘相加，它并不知道你的像素点之间实际距离是多少。
        # 因此，为了让离散计算的结果在数值大小上等于物理积分的结果，你必须乘上每一个网格点的单位面积
        # 也就是 dx0*dy0，这里因为 dx0=dy0，所以就是 dx0^2。

    # 计算光强，并提取y=0截面
    intensity_full = cp.abs(U)**2 #是给定Z位置的整个X-Y平面的光强分布
    intensity[iz, :] = intensity_full[Ny//2, :]   
    # 取y=0的x-z截面。行是y，列是x，Ny//2这一行是圆孔中心一条线


# ===================== 4. 绘制 x-z 平面光强分布 =====================
# 计算完成后，将需要保存/绘图的数据拉回 CPU
x_cpu = cp.asnumpy(x0)
z_cpu = cp.asnumpy(z)
intensity_cpu = cp.asnumpy(intensity)

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.figure(figsize=(12, 7))

extent = [x_min, x_max, z_min, z_max]

# 直接使用线性光强 intensity_cpu 画图
im = plt.imshow(
    intensity_cpu,    
    extent=extent,      
    aspect='auto',      
    cmap='jet',         
    origin='lower'      
)

plt.colorbar(im, label='Intensity (a.u.)')

# 设置轴标签和标题
plt.xlabel('x (μm)', fontsize=12)
plt.ylabel('z (μm)', fontsize=12)
plt.title('Fresnel Zone Plate - RS Diffraction (X-Z Plane)', fontsize=14)

plt.legend()
plt.tight_layout()

# 保存图片
plt.savefig('/root/Amp_RoundHole_Result.png', dpi=300)
plt.show()