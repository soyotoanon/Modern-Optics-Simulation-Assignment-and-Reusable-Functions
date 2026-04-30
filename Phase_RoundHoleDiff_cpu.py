# 计算机科学的精髓在于抽象主干，隐藏细节。可以考虑把rs衍射写成函数，传入复振幅即可
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 定义物理参数 =====================
lamda = 0.633  # 波长，单位μm
k = 2 * np.pi / lamda  # 波数 (1/μm)
R_max = 50 # 圆孔半径，单位μm
f = 50 # 该位相片把平行光变成汇聚的球面波，半径为f，单位μm

# 采样范围与点数（入射面/观测面采样间隔一致）
x_min, x_max = -150, 150   # x方向范围 (μm)
y_min, y_max = -150, 150   # y方向范围 (μm)
z_min, z_max = 30 , 70      # z方向范围 (μm)

Nx = 1024  # x方向采样点数
Ny = 1024  # y方向采样点数
Nz = 1024  # z方向采样点数

# 生成采样三维格点
x0 = np.linspace(x_min, x_max, Nx)    # 入射面x坐标
y0 = np.linspace(y_min, y_max, Ny)    # 入射面y坐标
dx0 = x0[1] - x0[0]                   # 采样间隔
z = np.linspace(z_min, z_max, Nz)     # 观测面z坐标

# 创建二维网格
X0, Y0 = np.meshgrid(x0, y0)


# ===================== 2. 生成圆形位相片的复振幅分布 =====================
def generate_RoundHole(X, Y, lamda, R_max, k, f):
    """生成圆形位相片"""
    r = np.sqrt(X**2 + Y**2)           # 二维径向距离
    amp = np.ones_like(X, dtype=np.complex64) # 初始化透振振幅，可以存储复数
  
    # 遮光条件：r > R_max μm
    mask_shade = r > R_max
    amp[mask_shade] = 0.0
    amp = np.where(np.isnan(amp), 0, amp)
    # 位相条件： r处位相是exp[-ikr^2/2f]
    mask_phase = r < R_max
    amp[mask_phase] = np.exp(-1j*k*(r[mask_phase])**2/(2*f))
    # 传进去二维数据的时候，注意右边也要是2D的，上面传入0.0就不需要考虑这个
    return amp

# 生成圆孔振幅分布 U(x0, y0, 0)
U0 = generate_RoundHole(X0, Y0, lamda, R_max, k, f)
U0 = U0.astype(np.complex64)

plt.figure(figsize=(6, 5))
plt.imshow(np.real(U0), cmap='gray', extent=[x_min, x_max, y_min, y_max], origin='lower') 
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.title('Round Hole Initial Amplitude')
plt.show()


# ===================== 3. RS衍射积分 =====================
intensity = np.zeros((Nz, Nx))   # 初始化光强数组

# 生成卷积核的二维坐标网格
X_kernel, Y_kernel = np.meshgrid(x0, y0)

# 提前计算初始场的FFT
U0_fft = np.fft.fft2(U0)   

# 遍历每个z位置计算衍射场
for iz in range(Nz):
    z_current = z[iz]   
  
    if np.isclose(z_current, 0):
        U = U0
    else:
        # RS卷积核计算
        r = np.sqrt(X_kernel**2 + Y_kernel**2 + z_current**2) 
        r = np.where(r == 0, 1e-12, r)   
        kernel = -(1/(2*np.pi)) * np.exp(1j * k * r) * (1j * k * z_current / r**2 - z_current / r**3)
        
        kernel_fft = np.fft.fft2(np.fft.ifftshift(kernel))
        U_fft = U0_fft * kernel_fft          
        U = np.fft.ifft2(U_fft)              
        U = U * (dx0**2)                     

    # 计算光强，并提取 y=0 的 x-z 截面
    intensity_full = np.abs(U)**2 
    intensity[iz, :] = intensity_full[Ny//2, :]   


# ===================== 4. 绘制 x-z 平面光强分布 =====================
intensity_cpu = intensity

# 将矩阵转置，使得行变成x，列变成z
intensity_transposed = intensity_cpu.T

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.figure(figsize=(12, 5))

extent = [z_min, z_max, x_min, x_max]

# 画图
im = plt.imshow(
    intensity_transposed,    
    extent=extent,      
    aspect='auto',      
    cmap='jet',         
    origin='lower'      
)

plt.colorbar(im, label='Intensity (a.u.)')

plt.xlabel('z (μm)', fontsize=12)
plt.ylabel('x (μm)', fontsize=12)
plt.title('Phase Round Hole - RS Diffraction (Z-X Plane)', fontsize=14)

plt.tight_layout()
plt.show()