
import numpy as np
import h5py
from scipy.integrate import solve_ivp
import os

np.random.seed(53)


# 基本参数
# V = 0.1
Qs = -1000
# T_in = 320
R = 8.314
A = 10
Tc = 300
# CA_in = 1000

# GRF 参数（标量）
# sigma_F = 0.004
sigma_V = 0.02
sigma_CA = 200
sigma_T = 10
sigma_k0 = 2e9
sigma_Ea = 5
sigma_H_rxn = 20
sigma_U = 0.2
sigma_rho = 50
sigma_Cp = 0.5

# mean_F = 0.02
mean_V = 0.1
mean_CA = 800
mean_T = 320
mean_k0 = 1e10
mean_Ea = 70
mean_H_rxn = -100
mean_U = 1
mean_rho = 1000
mean_Cp = 3

num_samples = 5000

def generate_scalar_grf(sigma, mean):
    """生成标量随机值"""
    return mean + sigma * np.random.normal(0, 1)

# 反应器方程（经典 BR，无空间依赖）
def br_reactor(t, y, V, k0, Ea, H_rxn, U, rho, Cp):
    CA, T = y
    dCA_dt = - k0 * np.exp(-Ea *1e3 / (R * T)) * CA
    dT_dt = (-H_rxn) / (rho * Cp) * k0 * np.exp(-Ea *1e3 / (R * T)) * CA + (U * A * (Tc - T)) / \
                (rho * Cp * V)
    return [dCA_dt, dT_dt]

# 时间和虚拟空间坐标
t_eval = np.linspace(0, 20, 201)
t_span = (0, 20)
x = np.linspace(0, 1, 128)  # 虚拟 x 轴，仅用于形状匹配


# 检查并创建目录
output_dir = 'D:\\Python_Projects\\Reactor_Models'
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, 'br_1st.h5')
# 删除已有文件
if os.path.exists(file_path):
    print(f"Removing existing file: {file_path}")
    os.remove(file_path)

# 创建 HDF5 文件
try:
    with h5py.File(file_path, 'w') as f:
        for i in range(num_samples):
            # 生成标量随机初始条件和参数
            V = generate_scalar_grf(sigma_V, mean_V)
            CA0 = generate_scalar_grf(sigma_CA, mean_CA)
            T0 = generate_scalar_grf(sigma_T, mean_T)
            k0 = generate_scalar_grf(sigma_k0, mean_k0)
            Ea = generate_scalar_grf(sigma_Ea, mean_Ea)
            H_rxn = generate_scalar_grf(sigma_H_rxn, mean_H_rxn)
            U = generate_scalar_grf(sigma_U, mean_U)
            rho = generate_scalar_grf(sigma_rho, mean_rho)
            Cp = generate_scalar_grf(sigma_Cp, mean_Cp)

            # 确保参数物理合理
            V = np.clip(V, 0.08, 0.12)
            CA0 = np.clip(CA0, 500, 1000)
            T0 = np.clip(T0, 300, 350)
            k0 = np.clip(k0, 8e9, 1.2e10)
            Ea = np.clip(Ea, 68, 72)
            H_rxn = np.clip(H_rxn, -120, -80)
            U = np.clip(U, 0.5, 1.2)
            rho = np.clip(rho, 900, 1100)
            Cp = np.clip(Cp, 2.5, 3.5)

            # 求解（结果形状: [201, 2]）
            sol = solve_ivp(
                lambda t, y: br_reactor(t, y, V, k0, Ea, H_rxn, U, rho, Cp),
                t_span, [CA0, T0],
                method='BDF', t_eval=t_eval
            )
            data = np.stack([sol.y[0], sol.y[1]], axis=-1)  # 形状: [201, 2]

            # 沿虚拟 x 轴重复，扩展为 [201, 128, 2]
            data_expanded = np.tile(data[:, np.newaxis, :], (1, 128, 1))  # 重复 128 次

            # 创建 group
            group_name = f'{i:04d}'
            group = f.create_group(group_name)
            group.create_dataset('data', data=data_expanded)
            # 保存标量参数
            group.attrs['V'] = V
            group.attrs['k0'] = k0
            group.attrs['Ea'] = Ea
            group.attrs['H_rxn'] = H_rxn
            group.attrs['U'] = U
            group.attrs['rho'] = rho
            group.attrs['Cp'] = Cp
            # 保存时间和虚拟空间坐标
            grid = group.create_group('grid')
            grid.create_dataset('t', data=t_eval)
            grid.create_dataset('x', data=x)

    print(f"HDF5 file saved successfully at: {file_path}")

except OSError as e:
    print(f"Failed to create HDF5 file: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")



import matplotlib.pyplot as plt

# 设置字体和大小
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# 读取 HDF5 文件
file_path = 'D:\\Python_Projects\\Reactor_Models\\br_1st.h5'
with h5py.File(file_path, 'r') as f:
    # 可视化演化
    sample_ids = ['0000', '1000', '2000', '4000']  # 展示多个样本
    t = f[sample_ids[0]]['grid']['t'][:]
    x = f[sample_ids[0]]['grid']['x'][:]  # 虚拟 x 轴

    # 选择几个虚拟空间点（值相同，仅展示一致性）
    space_indices = [0, 32, 64, 96, 127]
    space_labels = [f'x = {x[idx]:.2f}' for idx in space_indices]

    plt.figure(figsize=(6, 10))

    # 子图 1: CA 随 t
    plt.subplot(2, 1, 1)
    for sid in sample_ids:
        CA = f[sid]['data'][:, 0, 0]  # 取第一个虚拟空间点（所有点相同）
        plt.plot(t, CA, label=f'Sample {sid}', linewidth=2)
    plt.title('BR 1st Concentration (CA)')
    plt.xlabel('t (s)')
    plt.ylabel('CA (mol/m³)')
    plt.legend()

    # 子图 2: T 随 t
    plt.subplot(2, 1, 2)
    for sid in sample_ids:
        T = f[sid]['data'][:, 0, 1]  # 取第一个虚拟空间点（所有点相同）
        plt.plot(t, T, label=f'Sample {sid}', linewidth=2)
    plt.title('Temperature (T)')
    plt.xlabel('t (s)')
    plt.ylabel('T (K)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 检查数据一致性（可选）
    sample_id = '0000'
    data = f[sample_id]['data'][:]
    print(f"Sample {sample_id} shape: {data.shape}")
    print(f"CA variation: {np.max(data[:, :, 0]) - np.min(data[:, :, 0]):.1f}")
    print(f"T variation: {np.max(data[:, :, 1]) - np.min(data[:, :, 1]):.1f}")

