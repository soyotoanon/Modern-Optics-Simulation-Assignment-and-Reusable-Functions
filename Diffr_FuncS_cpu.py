import numpy as np
import matplotlib.pyplot as plt

'''
    包含的衍射函数种类：
    1.瑞利索莫非衍射
    2.菲涅尔近似下的角谱衍射
'''

#1111111111111111111111111111111111111111111111111111111111111111111111111111111
def RS_diffr(U0, x0, y0, z, lamda, show_plot=True):
    """
    计算基于卷积的 Rayleigh-Sommerfeld衍射积分，并支持直接绘图。
    
    参数:
    U0        : ndarray, 初始平面(z=0)的复振幅分布，二维数组，形状应为 (Ny, Nx)
    x0        : ndarray, 入射面 x 方向的坐标数组 (1D)
    y0        : ndarray, 入射面 y 方向的坐标数组 (1D)
    z         : ndarray 或 float, 观测面的 z 坐标。
                - 如果是浮点数，计算单一平面的二维光场。
                - 如果是数组，计算沿 z 轴传播的截面光场。
    lamda     : float, 光波长 (单位需与坐标系物理单位保持一致，如 μm)
    show_plot : bool, 是否自动显示结果图，默认为 True
    
    返回:
    intensity : ndarray，衍射光强分布矩阵
    """
    # 1. 基础物理参数与网格参数提取
    k = 2 * np.pi / lamda          # 波数 k = 2π/λ
    Nx = len(x0)                   # x方向采样点数
    Ny = len(y0)                   # y方向采样点数
    dx0 = x0[1] - x0[0]            # x方向空间采样间隔 (像素尺寸)
    dy0 = y0[1] - y0[0]            # y方向空间采样间隔

    # 2. 统一 z 的输入类型，方便后续使用统一的 for 循环处理
    is_scalar_z = np.isscalar(z)
    z_array = np.array([z]) if is_scalar_z else np.asarray(z)
    Nz = len(z_array)              # 记录需要计算的 z 平面数量

    # 3. 预处理：构建空域网格与初始场的频域分布
    X_kernel, Y_kernel = np.meshgrid(x0, y0)  # 生成用于计算距离 r 的二维坐标网格
    U0_fft = np.fft.fft2(U0)                  # 提前计算初始光场的二维FFT。因为每一层 z 的积分初始场相同，移到循环外可节省大量算力

    # 寻找 y 坐标最接近 0 的索引，用于在计算多层 z 时，精准提取 y=0 的 x-z 截面
    y_center_idx = np.argmin(np.abs(y0))
    
    # 初始化变量，用于存储 x-z 截面光强（若输入为 z 数组）和计算中的临时复振幅
    intensity_xz = np.zeros((Nz, Nx))
    U = np.zeros_like(U0, dtype=np.complex128)

    # 4. 核心计算：遍历所有给定的观测距离 z
    for iz in range(Nz):
        z_current = z_array[iz]
        
        # 边界条件：如果 z=0，衍射场就是初始场本身
        if np.isclose(z_current, 0):
            U = np.copy(U0)
        else:
            # 4.1 计算空间任意点到原点的距离 r = sqrt(x^2 + y^2 + z^2)
            r = np.sqrt(X_kernel**2 + Y_kernel**2 + z_current**2)
            r = np.where(r == 0, 1e-12, r)  # 避免极点处的除零错误 (r=0)

            # 4.2 计算第一类 Rayleigh-Sommerfeld 衍射系统的脉冲响应 (空域卷积核 h)
            # 严格公式推导结果包含两个项：近场项和远场项。
            # h(x,y) = - (1 / 2π) * exp(ikr) * (ikz/r^2 - z/r^3)
            kernel = -(1 / (2 * np.pi)) * np.exp(1j * k * r) * (1j * k * z_current / r**2 - z_current / r**3)

            # 4.3 利用 FFT 定理计算卷积: U = U0 * h  <=>  U = IFFT( FFT(U0) · FFT(h) )
            # 注意：np.fft.ifftshift(kernel) 非常关键。由于 h 是以中心对称的，
            # 需要用 ifftshift 将零频点移动到矩阵的四个角上，否则卷积后图像会发生循环平移(即图像错位)。
            kernel_fft = np.fft.fft2(np.fft.ifftshift(kernel))
            
            U_fft = U0_fft * kernel_fft
            
            # 乘上离散积分的面积元 (dx0 * dy0)，保证能量量纲正确
            U = np.fft.ifft2(U_fft) * (dx0 * dy0)  
            
        # 提取当前 z 平面下，中心行 (y=0) 的一维光强分布，存入 x-z 切片矩阵
        intensity_xz[iz, :] = (np.abs(U)**2)[y_center_idx, :]

    # 5. 绘图与返回逻辑
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    if is_scalar_z:
        # 如果输入的是单个距离 z，说明用户想看这个特定面的完整二维光斑
        intensity_xy = np.abs(U)**2
        if show_plot:
            plt.figure(figsize=(6, 5))
            extent = [x0.min(), x0.max(), y0.min(), y0.max()] # 设定坐标轴物理范围
            im = plt.imshow(intensity_xy, extent=extent, cmap='jet', origin='lower')
            plt.colorbar(im, label='Intensity (a.u.)')
            plt.xlabel('x (μm)', fontsize=12)
            plt.ylabel('y (μm)', fontsize=12)
            plt.title(f'RS Diffraction (X-Y Plane at z={z} μm)', fontsize=14)
            plt.tight_layout()
            plt.show()
        return intensity_xy
    else:
        # 如果输入的是 z 数组，说明用户想看光束在空间中的传播过程 (z-x 截面)
        if show_plot:
            plt.figure(figsize=(12, 5))
            extent = [z_array.min(), z_array.max(), x0.min(), x0.max()]
            # intensity_xz 是 (Nz, Nx) 的矩阵。imshow 默认行是Y轴，列是X轴，
            # 因此需要转置 (.T) 让 z 对应 X 轴(列)，x 对应 Y 轴(行)
            im = plt.imshow(intensity_xz.T, extent=extent, aspect='auto', cmap='jet', origin='lower')
            plt.colorbar(im, label='Intensity (a.u.)')
            plt.xlabel('z (μm)', fontsize=12)
            plt.ylabel('x (μm)', fontsize=12)
            plt.title('RS Diffraction (Z-X Plane at y=0)', fontsize=14)
            plt.tight_layout()
            plt.show()
        return intensity_xz
    

#2222222222222222222222222222222222222222222222222222222222222222222222222222222
import numpy as np
import matplotlib.pyplot as plt

def Fres_AS_diffr(U0, x0, y0, z, lamda, show_plot=True):
    """
    计算基于 Fresnel 近似的角谱衍射 (Angular Spectrum) 积分。
    纯 NumPy 版本，适合在 CPU 上运行。
    
    参数:
    U0        : ndarray, 初始复振幅分布，二维数组形状应为 (Ny, Nx)
    x0        : ndarray, 入射面 x 方向的坐标数组 (1D)
    y0        : ndarray, 入射面 y 方向的坐标数组 (1D)
    z         : ndarray 或 float, 观测面 z 坐标。
                - 如果是浮点数，计算单一平面的二维光场。
                - 如果是数组，计算沿 z 轴传播的截面光场。
    lamda     : float, 光波长 (单位需与坐标系保持一致)
    show_plot : bool, 是否自动显示结果图，默认为 True
    
    返回:
    intensity : ndarray，衍射光强分布矩阵
    """
    k = 2 * np.pi / lamda
    Nx = len(x0)
    Ny = len(y0)
    dx0 = x0[1] - x0[0]
    dy0 = y0[1] - y0[0]
    
    # 统一 z 的输入类型
    is_scalar_z = np.isscalar(z) or isinstance(z, (int, float))
    z_array = np.array([z]) if is_scalar_z else np.asarray(z)
    Nz = len(z_array)
    
    # 生成频域坐标网格
    # fftfreq 自动将零频放在两端，并对应 FFT 的输出格式
    fx = np.fft.fftfreq(Nx, d=dx0)
    fy = np.fft.fftfreq(Ny, d=dy0)
    FX, FY = np.meshgrid(fx, fy)
    
    # 提前计算初始场的 FFT
    U0_fft = np.fft.fft2(U0)
    
    # 寻找中心索引，用于提取 y=0 的截面
    y_center_idx = np.argmin(np.abs(y0))
    
    # 初始化变量
    intensity_xz = np.zeros((Nz, Nx))
    U = np.zeros_like(U0, dtype=np.complex128)
    
    # 遍历每个 z 位置计算衍射场
    for iz in range(Nz):
        z_current = z_array[iz]
        
        if np.isclose(z_current, 0):
            U = np.copy(U0)
        else:
            # Fresnel 近似下的传递函数 (角谱形式)
            # H(fx, fy) = exp(j * k * z) * exp(-j * pi * lamda * z * (fx^2 + fy^2))
            phase_fresnel = 1j * k * z_current - 1j * np.pi * lamda * z_current * (FX**2 + FY**2)
            H = np.exp(phase_fresnel)
            
            # 频域相乘后直接 IFFT 回到空域
            U = np.fft.ifft2(U0_fft * H)
            
        intensity_xz[iz, :] = (np.abs(U)**2)[y_center_idx, :]
        
    # --- 绘图与返回逻辑 ---
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    if is_scalar_z:
        intensity_xy = np.abs(U)**2
        if show_plot:
            plt.figure(figsize=(6, 5))
            extent = [x0.min(), x0.max(), y0.min(), y0.max()]
            im = plt.imshow(intensity_xy, extent=extent, cmap='jet', origin='lower')
            plt.colorbar(im, label='Intensity (a.u.)')
            plt.xlabel('x (μm)', fontsize=12)
            plt.ylabel('y (μm)', fontsize=12)
            plt.title(f'Fresnel AS Diffraction (X-Y Plane at z={z} μm)', fontsize=14)
            plt.tight_layout()
            plt.show()
        return intensity_xy
    else:
        if show_plot:
            plt.figure(figsize=(12, 5))
            # 修改1：extent 的顺序变更为 [横轴最小, 横轴最大, 纵轴最小, 纵轴最大]
            # 也就是 [z_min, z_max, x_min, x_max]
            extent = [z_array.min(), z_array.max(), x0.min(), x0.max()]
            
            # 修改2：将 intensity_xz 矩阵转置 (.T)
            # 转置后矩阵形状变为 (Nx, Nz)，即行对应 x，列对应 z
            im = plt.imshow(intensity_xz.T, extent=extent, aspect='auto', cmap='jet', origin='lower')
            plt.colorbar(im, label='Intensity (a.u.)')
            
            # 修改3：互换坐标轴标签
            plt.xlabel('z (μm)', fontsize=12)
            plt.ylabel('x (μm)', fontsize=12)
            plt.title('Fresnel AS Diffraction (Z-X Plane at y=0)', fontsize=14)
            plt.tight_layout()
            plt.show()
        return intensity_xz