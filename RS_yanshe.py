# 要可迁移，改一改就能用
import numpy as np
import cupy as cp 
import matplotlib.pyplot as plt


# ===================== 1. 定义物理参数 =====================
lamda = 0.532  # 波长，单位μm
f = 25         # 波带片焦距，单位μm
k = 2 * np.pi / lamda  # 波数 (1/μm)

# 采样范围与点数（入射面/观测面采样间隔一致）
x_min, x_max = -80, 80   # x方向范围 (μm)
y_min, y_max = -80, 80   # y方向范围 (μm)
z_min, z_max = 0, 50      # z方向范围 (μm)

Nx = 8192  # x方向采样点数
Ny = 8192  # y方向采样点数
Nz = 2048  # z方向采样点数

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


# ===================== 2. 生成二维菲涅尔波带片振幅分布 =====================
def generate_fresnel_zone_plate_2d(X, Y, lamda, f, R_max=50):
    """生成二维圆对称菲涅尔波带片
        X, Y: 二维坐标网格数组，把 X0 和 Y0 相同位置的元素组合起来，就得到了平面上所有的坐标点
        lamda: 波长
        f: 波带片焦距
        R_max: 最大半径，超过此半径的区域将被截断（单位μm）"""
    r = cp.sqrt(X**2 + Y**2)           # 二维径向距离（二维数组）
    n = r**2 / (lamda * f)             # 每个点对应的半波带序号（n也是一个二维数组）
    n_int = cp.floor(n) + 1                 # 取整(ai写的是向下取整，似乎不对)
    amp = cp.ones_like(X)              # 初始化透振振幅,全透光
  
    # 遮光条件：奇数级波带遮光 + 半径截断
    mask_shade = (n_int % 2 == 1) | (r > R_max)
    amp[mask_shade] = 0.0
    amp = cp.where(cp.isnan(amp), 0, amp)
    # cp.where(condition, x, y)，condition满足时返回x，否则返回y。这里将NaN值替换为0，确保振幅数组中没有NaN值。
    return amp

# 生成二维波带片初始场分布 U(x0, y0, 0)
U0 = generate_fresnel_zone_plate_2d(X0, Y0, lamda, f)
U0 = U0.astype(cp.complex64)  # 转换为复数类型，方便后续计算

plt.imshow(cp.asnumpy(cp.real(U0))) # 显示波带片定义是否正确.注意把gpu数据改成numpy
# 振幅性波带片的透射率就是U0取实部


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
        # z=0时为波带片平面，场分布等于初始场
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
    # 取y=0的x-z截面。行是y，列是x，Ny//2这一行是波片中心一条线


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

# 画设计焦距虚线，用于和实际焦距对比
plt.axhline(y=f, color='white', linestyle='--', alpha=0.5, label=f'Design focal length z={f} μm')

plt.legend()
plt.tight_layout()

# 保存图片
plt.savefig('/root/diffraction_result_linear.png', dpi=300)
plt.show()

# ===================== 5. 焦距信息提取 =====================
# 找到离设计焦距最近的计算平面的索引
z_f_idx = cp.argmin(cp.abs(z - f))
print(f"\n设计焦距: z = {f} μm")
print(f"最近计算位置: z = {float(z[z_f_idx]):.2f} μm")


# ===================== 6. 提取焦平面1D光强并测量 FWHM =====================
# 提取焦平面处的横向光强分布 (使用之前找到的最近 z 平面索引 z_f_idx)
I_focal = intensity_cpu[int(z_f_idx), :]

# 找到最大光强
I_max = np.max(I_focal)
I_half = I_max / 2.0  # 半最大值 (Half Maximum)

# 寻找半高宽的左右边界
# 找到所有光强 >= 一半最大值的像素点索引
half_max_indices = np.where(I_focal >= I_half)[0]
left_idx = half_max_indices[0]   # 左边界索引
right_idx = half_max_indices[-1] # 右边界索引

# 获取对应的真实物理坐标
x_left = x_cpu[left_idx]
x_right = x_cpu[right_idx]
fwhm = x_right - x_left  # 计算物理宽度

print(f"测量得到的焦斑半高宽 (FWHM): {fwhm:.3f} μm")

# ===================== 7. 绘制 1D 焦点剖面图 =====================
plt.figure(figsize=(8, 5))

# 画出 1D 光强曲线
plt.plot(x_cpu, I_focal, color='blue', linewidth=1.5, label='Intensity Profile')

# 画一根红色的水平线段，直接标出 FWHM 的位置
plt.hlines(I_half, x_left, x_right, color='red', linestyle='-', linewidth=2.5, 
           label=f'FWHM = {fwhm:.3f} μm')

# 画两条垂直虚线，对齐边界，方便视觉对齐
plt.axvline(x_left, color='red', linestyle='--', alpha=0.5)
plt.axvline(x_right, color='red', linestyle='--', alpha=0.5)

# 为了看清光斑细节，把 X 轴放大到中心区域 [-5, 5] 微米
plt.xlim(-5, 5) 

plt.xlabel('x (μm)', fontsize=12)
plt.ylabel('Intensity (a.u.)', fontsize=12)
plt.title(f'1D Focal Spot Profile at z = {float(z[z_f_idx]):.2f} μm', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig('/root/focal_spot_1d.png', dpi=300)