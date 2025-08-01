import numpy as np
import h5py
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

np.random.seed(44)
# PFR 反应器函数（保持不变）
def pfr_reactor_mol(nt, nz, t_max, z_max, r, u, k0, Ea, R, dH, rho, Cp, U, Tc, C_A0, T0):
    dz = z_max / (nz - 1)
    z = np.linspace(0, z_max, nz)

    As = 2 * np.pi * r * z_max
    Vol = np.pi * r ** 2 * z_max

    # 初始条件
    C_A = np.zeros(nz)
    T = np.ones(nz) * T0  # 使用随机初始温度 T0
    C_A[0] = C_A0  # 入口浓度
    T[0] = T0  # 入口温度

    y0 = np.concatenate([C_A, T])

    def dydt(t, y):
        C_A = y[:nz]
        T = y[nz:]

        rate = k0 * np.exp(-Ea * 1e3 / (R * T)) * C_A

        dC_A_dt = np.zeros(nz)
        dC_A_dt[0] = 0
        dC_A_dt[1:-1] = -u * (C_A[1:-1] - C_A[:-2]) / dz - rate[1:-1]
        dC_A_dt[-1] = dC_A_dt[-2]
        # 出口自由流出修正
        # dC_A_dt[-1] = -u * (C_A[-1] - C_A[-2]) / dz - rate[-1]

        dT_dt = np.zeros(nz)
        dT_dt[0] = 0
        dT_dt[1:-1] = (
                -u * (T[1:-1] - T[:-2]) / dz +
                (-dH / (rho * Cp)) * rate[1:-1] +
                (U * As / (rho * Cp * Vol)) * (Tc - T[1:-1])
        )
        dT_dt[-1] = dT_dt[-2]
        # 出口自由流出修正
        # dT_dt[-1] = (
        #     -u * (T[-1] - T[-2]) / dz +
        #     (-dH / (rho * Cp)) * rate[-1] +
        #     (U * As / (rho * Cp * Vol)) * (Tc - T[-1])
        # )

        return np.concatenate([dC_A_dt, dT_dt])

    t_eval = np.linspace(0, t_max, nt)
    sol = solve_ivp(dydt, [0, t_max], y0, t_eval=t_eval, method='BDF')

    results = np.zeros((nt, nz, 2))
    results[:, :, 0] = sol.y[:nz, :].T  # 浓度
    results[:, :, 1] = sol.y[nz:, :].T  # 温度

    return results, z, t_eval


# 参数设置
nt = 201
nz = 128
t_max = 20
# z_max = 1
# r = 0.1
R = 8.314
# As = 2 * np.pi * r * z_max
# Vol = np.pi * r ** 2 * z_max
Tc = 300

num_samples = 5

# 随机参数范围（均值和标准差）
params = {
    'u': {'mean': 0.08, 'sigma': 0.01, 'min': 0.05, 'max': 0.1},  # 速度 (m/s)
    'k0': {'mean': 1e11, 'sigma': 2e10, 'min': 8e10, 'max': 1.2e11},  # 反应速率常数 (1/s)
    'Ea': {'mean': 80, 'sigma': 5, 'min': 78, 'max': 82},  # 活化能 (kJ/mol)
    'dH': {'mean': -100, 'sigma': 20, 'min': -130, 'max': -70},  # 反应热 (kJ/mol)
    'rho': {'mean': 1000, 'sigma': 50, 'min': 900, 'max': 1100},  # 密度 (kg/m³)
    'Cp': {'mean': 1, 'sigma': 0.2, 'min': 0.8, 'max': 1.2},  # 比热容 (kJ/kg·K)
    'U': {'mean': 1, 'sigma': 0.2, 'min': 0.5, 'max': 1.2},  # 传热系数 (kW/m²·K)
    'C_A0': {'mean': 750, 'sigma': 150, 'min': 500, 'max': 1000},  # 初始浓度 (mol/m³)
    'T0': {'mean': 320, 'sigma': 10, 'min': 300, 'max': 340},  # 初始温度 (K)
    'z_max': {'mean': 1, 'sigma': 0.2, 'min': 0.8, 'max': 1.2},  # 长度
    'r': {'mean': 0.1, 'sigma': 0.02, 'min': 0.08, 'max': 0.12} # 半径
}

# 检查并创建目录
import os
output_dir = 'D:\\Python_Projects\\Reactor_Models'
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, 'pfr_1st.h5')
# 删除已有文件
if os.path.exists(file_path):
    print(f"Removing existing file: {file_path}")
    os.remove(file_path)

# 生成数据并保存
with h5py.File(file_path, "w") as f:
    for i in range(num_samples):
        # 生成随机参数
        u = np.clip(np.random.normal(params['u']['mean'], params['u']['sigma']), params['u']['min'], params['u']['max'])
        k0 = np.clip(np.random.normal(params['k0']['mean'], params['k0']['sigma']), params['k0']['min'],
                     params['k0']['max'])
        Ea = np.clip(np.random.normal(params['Ea']['mean'], params['Ea']['sigma']), params['Ea']['min'],
                     params['Ea']['max'])
        dH = np.clip(np.random.normal(params['dH']['mean'], params['dH']['sigma']), params['dH']['min'],
                     params['dH']['max'])
        rho = np.clip(np.random.normal(params['rho']['mean'], params['rho']['sigma']), params['rho']['min'],
                      params['rho']['max'])
        Cp = np.clip(np.random.normal(params['Cp']['mean'], params['Cp']['sigma']), params['Cp']['min'],
                     params['Cp']['max'])
        U = np.clip(np.random.normal(params['U']['mean'], params['U']['sigma']), params['U']['min'], params['U']['max'])
        C_A0 = np.clip(np.random.normal(params['C_A0']['mean'], params['C_A0']['sigma']), params['C_A0']['min'],
                       params['C_A0']['max'])
        T0 = np.clip(np.random.normal(params['T0']['mean'], params['T0']['sigma']), params['T0']['min'],
                     params['T0']['max'])
        z_max = np.clip(np.random.normal(params['z_max']['mean'], params['z_max']['sigma']), params['z_max']['min'],
                     params['z_max']['max'])
        r = np.clip(np.random.normal(params['r']['mean'], params['r']['sigma']), params['r']['min'],
                     params['r']['max'])

        # 求解
        results, z, t = pfr_reactor_mol(nt, nz, t_max, z_max, r, u, k0, Ea, R, dH, rho, Cp, U, Tc, C_A0, T0)

        # 创建 group
        group_name = f"{i:04d}"
        group = f.create_group(group_name)
        group.create_dataset("data", data=results)
        grid = group.create_group("grid")
        grid.create_dataset("t", data=t)
        grid.create_dataset("x", data=z)

        # 存储随机参数
        group.attrs['u'] = u
        group.attrs['k0'] = k0
        group.attrs['Ea'] = Ea
        group.attrs['dH'] = dH
        group.attrs['rho'] = rho
        group.attrs['Cp'] = Cp
        group.attrs['U'] = U
        group.attrs['C_A0'] = C_A0
        group.attrs['T0'] = T0
        group.attrs['z_max'] = z_max
        group.attrs['r'] = r

    print(f"计算完成，结果已保存到 pfr_1st.h5")

# 可视化检查结果
with h5py.File("pfr_1st.h5", "r") as f:
    sample_ids = ['0000', '0001', '0002', '0003']
    t = f[sample_ids[0]]['grid']['t'][:]
    z = f[sample_ids[0]]['grid']['x'][:]

    # 设置绘图参数
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14

    # 可视化浓度和温度随时间变化（固定空间点）
    space_indices = [0, 32, 64, 96, 127]  # z = 0, 0.25, 0.5, 0.75, 1
    space_labels = [f'z = {z[idx]:.2f}' for idx in space_indices]

    plt.figure(figsize=(6, 10))
    for sid in sample_ids:
        data = f[sid]['data'][:]
        CA = data[:, :, 0]
        T = data[:, :, 1]

        plt.subplot(2, 1, 1)
        plt.plot(t, CA[:, space_indices[4]], label=f'Sample {sid} (z={z[space_indices[4]]:.2f})', linewidth=2)
        plt.title('PFR 1st Concentration (CA) at Exit')
        plt.xlabel('t (s)')
        plt.ylabel('CA (mol/m³)')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t, T[:, space_indices[2]], label=f'Sample {sid} (z={z[space_indices[4]]:.2f})', linewidth=2)
        plt.title('Temperature (T) at Exit')
        plt.xlabel('t (s)')
        plt.ylabel('T (K)')
        plt.legend()

    plt.tight_layout()
    plt.show()

    # 子图 1: CA 随 t
    sample_id = '0004'
    data = f[sample_id]['data'][:]
    CA = data[:, :, 0]
    T = data[:, :, 1]
    plt.figure(figsize=(6, 10))
    plt.subplot(2, 1, 1)
    for idx, label in zip(space_indices, space_labels):
        plt.plot(t, CA[:, idx], label=label, linewidth=2)
    plt.title(f'PFR 1st Concentration (CA) - Sample {sample_id}')
    plt.xlabel('t (s)')
    plt.ylabel('CA (mol/m³)')
    plt.legend()

    # 子图 2: T 随 t
    plt.subplot(2, 1, 2)
    for idx, label in zip(space_indices, space_labels):
        plt.plot(t, T[:, idx], label=label, linewidth=2)
    plt.title(f'Temperature (T) - Sample {sample_id}')
    plt.xlabel('t (s)')
    plt.ylabel('T (K)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 可视化单个样本的空间分布（固定时间点）

    time_indices = [0, 50, 100, 150, 200]  # t = 0, 5, 10, 15, 20 s
    time_labels = [f't = {t[idx]:.1f} s' for idx in time_indices]

    plt.figure(figsize=(6, 10))
    plt.subplot(2, 1, 1)
    for idx, label in zip(time_indices, time_labels):
        plt.plot(z, CA[idx, :], label=label, linewidth=2)
    plt.title(f'PFR 1st Concentration (CA) - Sample {sample_id}')
    plt.xlabel('z (m)')
    plt.ylabel('CA (mol/m³)')
    plt.legend()

    plt.subplot(2, 1, 2)
    for idx, label in zip(time_indices, time_labels):
        plt.plot(z, T[idx, :], label=label, linewidth=2)
    plt.title(f'Temperature (T) - Sample {sample_id}')
    plt.xlabel('z (m)')
    plt.ylabel('T (K)')
    plt.legend()

    plt.tight_layout()
    plt.show()


    # 检查合理性
    print(f"Checking Sample {sample_id}:")
    print(f"CA min: {CA.min():.2f}, CA max: {CA.max():.2f}")
    print(f"T min: {T.min():.2f}, T max: {T.max():.2f}")
    if CA.min() < 0:
        print("Warning: Negative concentration detected!")
    if T.min() < 0:
        print("Warning: Negative temperature detected!")
    elif T.min() < 200 or T.max() > 500:
        print("Warning: Temperature range might be unrealistic (outside 200-500 K)!")
    print(f"Parameters: u={f[sample_id].attrs['u']:.3f}, k0={f[sample_id].attrs['k0']:.2e}, "
          f"Ea={f[sample_id].attrs['Ea']:.1f}, dH={f[sample_id].attrs['dH']:.1f}, "
          f"rho={f[sample_id].attrs['rho']:.1f}, Cp={f[sample_id].attrs['Cp']:.2f}, "
          f"U={f[sample_id].attrs['U']:.2f}, C_A0={f[sample_id].attrs['C_A0']:.1f}, "
          f"T0={f[sample_id].attrs['T0']:.1f}, z_max={f[sample_id].attrs['z_max']:.2f}, r={f[sample_id].attrs['r']:.2f}")