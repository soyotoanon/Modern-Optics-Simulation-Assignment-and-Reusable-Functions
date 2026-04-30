import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

'''包含的函数种类：
    1.瑞利索莫非衍射
    2.菲涅尔近似下的角谱衍射
'''

# 222222222222222222222222222222222222222222222222222222222222222222222222222222
def Fresnel_AS_diffr(U0, x0, y0, z, lamda, show_plot=True):
    """
    计算基于 Fresnel 近似的角谱衍射 (Angular Spectrum) 积分。
    使用 GPU (CuPy) 加速计算。
    
    参数:
    U0        : cupy.ndarray, 初始复振幅分布，二维数组形状应为 (Ny, Nx)
    x0        : cupy.ndarray, 入射面 x 方向的坐标数组 (1D)
    y0        : cupy.ndarray, 入射面 y 方向的坐标数组 (1D)
    z         : cupy.ndarray 或 float, 观测面 z 坐标。
                - 如果是浮点数，计算单一平面的二维光场。
                - 如果是数组，计算沿 z 轴传播的截面光场。
    lamda     : float, 光波长 (单位需与坐标系保持一致)
    show_plot : bool, 是否自动显示结果图，默认为 True
    
    返回:
    intensity : cupy.ndarray，衍射光强分布矩阵 (在GPU内存中)
    """
    k = 2 * np.pi / lamda
    Nx = len(x0)
    Ny = len(y0)
    dx0 = x0[1] - x0[0]
    dy0 = y0[1] - y0[0]
    
    # 统一 z 的输入类型
    is_scalar_z = cp.isscalar(z) or isinstance(z, (int, float))
    z_array = cp.array([z]) if is_scalar_z else cp.asarray(z)
    Nz = len(z_array)
    
    # 生成频域坐标网格
    # fftfreq 自动将零频放在两端，并对应 FFT 的输出格式
    fx = cp.fft.fftfreq(Nx, d=dx0)
    fy = cp.fft.fftfreq(Ny, d=dy0)
    FX, FY = cp.meshgrid(fx, fy)
    
    # 提前计算初始场的 FFT
    U0_fft = cp.fft.fft2(U0)
    
    # 寻找中心索引
    y_center_idx = cp.argmin(cp.abs(y0))
    
    # 初始化变量
    intensity_xz = cp.zeros((Nz, Nx))
    U = cp.zeros_like(U0, dtype=cp.complex128)
    
    # 遍历每个 z 位置计算衍射场
    for iz in range(Nz):
        z_current = z_array[iz]
        
        if cp.isclose(z_current, 0):
            U = cp.copy(U0)
        else:
            # Fresnel 近似下的传递函数 (角谱形式)
            # H(fx, fy) = exp(j * k * z) * exp(-j * pi * lamda * z * (fx^2 + fy^2))
            phase_fresnel = 1j * k * z_current - 1j * np.pi * lamda * z_current * (FX**2 + FY**2)
            H = cp.exp(phase_fresnel)
            
            # 频域相乘后直接 IFFT 回到空域
            U = cp.fft.ifft2(U0_fft * H)
            
        intensity_xz[iz, :] = (cp.abs(U)**2)[y_center_idx, :]
        
    # --- 绘图与返回逻辑 ---
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    if is_scalar_z:
        intensity_xy = cp.abs(U)**2
        if show_plot:
            # 绘图前转回 CPU
            intensity_xy_cpu = cp.asnumpy(intensity_xy)
            x0_cpu = cp.asnumpy(x0)
            y0_cpu = cp.asnumpy(y0)
            
            plt.figure(figsize=(6, 5))
            extent = [x0_cpu.min(), x0_cpu.max(), y0_cpu.min(), y0_cpu.max()]
            im = plt.imshow(intensity_xy_cpu, extent=extent, cmap='jet', origin='lower')
            plt.colorbar(im, label='Intensity (a.u.)')
            plt.xlabel('x (μm)', fontsize=12)
            plt.ylabel('y (μm)', fontsize=12)
            plt.title(f'Fresnel AS Diffraction (X-Y Plane at z={z} μm)', fontsize=14)
            plt.tight_layout()
            plt.show()
        return intensity_xy
    else:
        if show_plot:
            # 绘图前转回 CPU
            intensity_xz_cpu = cp.asnumpy(intensity_xz)
            x0_cpu = cp.asnumpy(x0)
            z_array_cpu = cp.asnumpy(z_array)
            
            plt.figure(figsize=(12, 5))
            # 注意 extent: [xmin, xmax, zmin, zmax]，并且数据转置
            extent = [x0_cpu.min(), x0_cpu.max(), z_array_cpu.min(), z_array_cpu.max()]
            # 你的原代码中 x 轴和 z 轴画图方式是 extent=[x_min, x_max, z_min, z_max]
            im = plt.imshow(intensity_xz_cpu, extent=extent, aspect='auto', cmap='jet', origin='lower')
            plt.colorbar(im, label='Intensity (a.u.)')
            plt.xlabel('x (μm)', fontsize=12)
            plt.ylabel('z (μm)', fontsize=12)
            plt.title('Fresnel AS Diffraction (X-Z Plane at y=0)', fontsize=14)
            plt.tight_layout()
            plt.show()
        return intensity_xz