from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as ticker

'''
。
'''

N = 4
r = 2
omega = np.exp (4 * r)
U = np.exp (4 * r) * 3
omegaz = omega
Delta = omegaz / 30
omegax = Delta
K = 0
chi = (omegax ) * 1e-2
gamma = chi * 0.0001
omegacz = chi




a = destroy(N)
adag = a.dag()
I = qeye(N)
a1 = tensor(a, I)
a1dag = a1.dag()
n1 = a1dag * a1
a2 = tensor(I, a)
a2dag = a2.dag()
n2 = a2dag * a2
c_op_single = np.sqrt(gamma) * a + np.sqrt(gamma) * adag + np.sqrt(gamma) * adag * a
c_ops_2q = [tensor(c_op_single, I), tensor(I, c_op_single)]

H_Rz_single = omega * adag * a + U * adag * adag * a * a
H_Rzx_single = -omegax * adag * a + (omegax / 2.0) * (a + adag) + U * adag * adag * a * a

H_int = K * (a1dag * a2 + a1 * a2dag) + chi * n1 * n2

# Rz
H_Rz_global = tensor(H_Rz_single, I) + tensor(I, H_Rz_single) + H_int
# Rxz on qubi 1
H_Rzx_Q1 = tensor(H_Rzx_single, I) + tensor(I, H_Rz_single) + H_int
# Rxz on qubit 2
H_Rzx_Q2 = tensor(H_Rz_single, I) + tensor(I, H_Rzx_single) + H_int
# 都有
H_Rzx_Q12 = tensor(H_Rzx_single, I) + tensor(I, H_Rzx_single) + H_int
print("Hamiltonian H_Rz_global shape:", H_Rz_global.shape)

opts = {
    "nsteps": 500000000,
    "atol": 1e-11,
    "rtol": 1e-11
}
  # 所有哈密顿量。 这里这个Rcz就是对于 Cz旋转特殊优化的 不然 Rz 转太多圈
def apply_Rz_evolution(psi_in, t_duration):
    t_list = np.linspace(0, t_duration, 20)
    result = mesolve(H_Rz_global, psi_in, t_list, c_ops_2q, [], options=opts)
    return result

'''def apply_Rcz_evolution(psi_in, t_duration):
    t_list = np.linspace(0, t_duration, 20)
    result = mesolve(H_int, psi_in, t_list, c_ops_2q, [], options=opts)
    tz = ((t_duration * omegaz) % (2 * np.pi)) / omegaz
    tz_list = np.linspace(0, tz, 2)
    result1 = mesolve(tensor(H_Rz_single, I) + tensor(I, H_Rz_single), result.states[-1], tz_list, c_ops_2q, [], options=opts)
    return result1'''

def apply_Rcz_evolution(psi_in, t_duration):
    t_list = np.linspace(0, t_duration, 20)
    result = mesolve(H_Rz_global, psi_in, t_list, c_ops_2q, [], options=opts)
    return result

def apply_Rzx_Q1(psi_in, t_duration):
    t_list = np.linspace(0, t_duration, 20)
    result = mesolve(H_Rzx_Q1, psi_in, t_list, c_ops_2q, [], options=opts)
    return result


def apply_Rzx_Q2(psi_in, t_duration):
    t_list = np.linspace(0, t_duration, 20)
    result = mesolve(H_Rzx_Q2, psi_in, t_list, c_ops_2q, [], options=opts)
    return result


def apply_Rzx_Q12(psi_in, t_duration):
    t_list = np.linspace(0, t_duration, 20)
    result = mesolve(H_Rzx_Q12, psi_in, t_list, c_ops_2q, [], options=opts)
    return result



def run_sequence(sequence, psi_init):
    current_psi = psi_init
    results_list = []
    gate_names = []
    def print_state_readable(state):
        N_dim = state.dims[0][0]
        data = state.full()
        is_ket = state.type == 'ket'

        coeffs = np.zeros(N_dim * N_dim, dtype=complex)

        if is_ket:
            for i in range(N_dim * N_dim):
                coeffs[i] = data[i][0]
        else:
            diags = np.real(np.diag(data))
            k = np.argmax(diags)
            if diags[k] > 1e-10:
                c_k = np.sqrt(diags[k])
                for i in range(N_dim * N_dim):
                    coeffs[i] = data[i, k] / c_k

        printed_anything = False
        for i in range(N_dim):
            for j in range(N_dim):
                idx = i * N_dim + j
                val = coeffs[idx]
                if abs(val) > 1e-4:
                    print(f"  |{i}{j}> : {val.real:+.4f}{val.imag:+.4f}j")
                    printed_anything = True

        if not printed_anything:
            print("All zero")


    print("\n" + "=" * 40)
    print("Initial State:")
    print_state_readable(current_psi)
    print("=" * 40 + "\n")

    for step, (gate, target, t_dur) in enumerate(sequence):
        print(f"Step {step + 1}: {gate} on {target} for time {t_dur:.4e}s")

        if gate == 'Rzx':
            if target == 1:
                res = apply_Rzx_Q1(current_psi, t_dur)
                gate_names.append("Rzx(Q1)")
            elif target == 2:
                res = apply_Rzx_Q2(current_psi, t_dur)
                gate_names.append("Rzx(Q2)")
            elif target == 12:
                res = apply_Rzx_Q12(current_psi, t_dur)
                gate_names.append("Rzx(Q1+Q2)")

        elif gate == 'Rz':
            res = apply_Rz_evolution(current_psi, t_dur)
            gate_names.append(f"Rz(Global)")
        elif gate == 'Rcz':
            res = apply_Rcz_evolution(current_psi, t_dur)
            gate_names.append(f"Rcz(Global)")

        results_list.append(res)
        current_psi = res.states[-1]  # 状态接力
        print(f"\n--- State after Step {step + 1} ({gate} on {target}) ---")
        print_state_readable(current_psi)
        print("-" * 40 + "\n")

    return results_list, gate_names


def plot_2q_population_final(results_list, gate_names, save_pdf=False, filename="CNOT_gate（有耗散）.pdf"):
    # 1. 样式配置
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.labelsize": 18,
        "legend.fontsize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 13,
        "pdf.fonttype": 42,
    })

    # 2. 提取数据（按门序号归一化）
    full_steps = []
    pop_dict = {'00': [], '01': [], '10': [], '11': []}
    boundaries = []  # 记录每个门的分界点

    N_val = results_list[0].states[0].dims[0][0]
    p_ops = {
        '00': tensor(basis(N_val, 0), basis(N_val, 0)).proj(),
        '01': tensor(basis(N_val, 0), basis(N_val, 1)).proj(),
        '10': tensor(basis(N_val, 1), basis(N_val, 0)).proj(),
        '11': tensor(basis(N_val, 1), basis(N_val, 1)).proj()
    }

    for k, res in enumerate(results_list):
        t_orig = np.array(res.times)
        t_start, t_end = t_orig[0], t_orig[-1]

        # 将当前门的时间 [t_start, t_end] 映射到 [k, k+1]
        if t_end > t_start:
            step_array = k + (t_orig - t_start) / (t_end - t_start)
        else:
            # 处理时间为 0 的特殊情况
            step_array = np.linspace(k, k + 1, len(t_orig))

        full_steps.extend(step_array.tolist())
        boundaries.append(k + 1)  # 每一个整数点都是边界

        for key in p_ops:
            pop_dict[key].extend(expect(p_ops[key], res.states))
    # 3. 绘图
    fig, ax = plt.subplots(figsize=(10, 5))  # 步骤较多时，宽度设长一点

    # 按照你之前的颜色/线型设置：00/01 蓝色系，10/11 橙色系
    colors = ['#000000', '#000000', '#f20c00', '#f20c00']
    styles = ['-', '--', '-', '--']
    labels = [r'$|00\rangle$', r'$|01\rangle$', r'$|10\rangle$', r'$|11\rangle$']

    for i, key in enumerate(['00', '01', '10', '11']):
        ax.plot(full_steps, pop_dict[key], label=labels[i],
                color=colors[i], lw=2.2, ls=styles[i])

    # 4. 坐标轴修饰
    ax.set_ylabel("Population")
    ax.set_xlabel("Gate Sequence")
    ax.set_ylim(-0.02, 1.05)  # 建议设回 1.05 以看清 P=1 的曲线
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.grid(True, ls=':', alpha=0.5)

    # 5. 设置横坐标为门的名字
    # 将刻度放在每个门的正中间
    ax.set_xticks(np.arange(len(gate_names)) + 0.5)
    ax.set_xticklabels(gate_names, rotation=45, ha='right')

    # 6. 画分界斜线
    for b in boundaries[:-1]:
        ax.axvline(x=b, color='gray', lw=1.0, ls='--', alpha=0.5)

    # 7. 图例
    ax.legend(loc='best', ncol=2, frameon=True, edgecolor='black', fancybox=False, framealpha=0.9)

    plt.tight_layout()
    if save_pdf:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def calculate_gate_fidelity(sequence, U_ideal_logical): # 计算保真度 用演化单位矩阵的方法
    U_total = qeye(N * N)
    U_total.dims = [[N, N], [N, N]]

    for gate, target, t_dur in sequence:
        if gate == 'Rzx':
            H = H_Rzx_Q1 if target == 1 else (H_Rzx_Q2 if target == 2 else H_Rzx_Q12)
        elif gate == 'Rz':
            H = H_Rz_global
        elif gate == 'Rcz':
            H_step1 = H_int
            U_step1 = (-1j * H_step1 * t_dur).expm()

            tz = ((t_dur * omegaz) % (2 * np.pi)) / omegaz
            H_step2 = tensor(H_Rz_single, I) + tensor(I, H_Rz_single)
            U_step2 = (-1j * H_step2 * tz).expm()

            U_current_gate = U_step2 * U_step1
        else:
            H = H_int if gate == 'Rczcl' else H_Rz_global
            U_current_gate = (-1j * H * t_dur).expm()
        if gate != 'Rcz':
            U_current_gate = (-1j * H * t_dur).expm()
        U_total = U_current_gate * U_total

    logical_indices = [0, 1, N, N + 1]
    U_actual_mat = U_total.full()[np.ix_(logical_indices, logical_indices)]
    basis_names = ["|00>", "|01>", "|10>", "|11>"]

    for i in range(4):
        col = U_actual_mat[:, i]
        print(f"\nInitial State {basis_names[i]} evolved to:")
        for j in range(4):
            val = col[j]
            if abs(val) > 1e-3:
                print(f"  -> {basis_names[j]} : {val.real:+.4f}{val.imag:+.4f}j  (Amp: {abs(val):.4f})")
    norm_sum = np.sum(np.abs(U_actual_mat) ** 2, axis=0)
    U_ideal_mat = U_ideal_logical.full()
    d = 4
    trace_overlap = np.trace(U_ideal_mat.conj().T @ U_actual_mat)
    return (np.abs(trace_overlap) ** 2) / (d ** 2)




'''
哦 这里根本用不到
这里还有问题
def pure_Rcz(target_phase=None, target_time=None, t_prior=0.0):
    if target_time is not None:
        theta_target = (chi * target_time) % (2 * np.pi)
    elif target_phase is not None:
        theta_target = target_phase % (2 * np.pi)

    phi_z_prior = (omegaz * t_prior) % (2 * np.pi)
    t_min_z = (2 * np.pi - phi_z_prior) % (2 * np.pi) / omegaz

    phi_cz_base = (chi * (t_prior + t_min_z)) % (2 * np.pi)
    delta_phi_cz = (theta_target - phi_cz_base) % (2 * np.pi)
    d_phi_cz_per_rev = (chi * 2 * np.pi) / omegaz
    k = delta_phi_cz / d_phi_cz_per_rev
    t_total_new = t_min_z + k * (2 * np.pi / omegaz)

    t1 = 0.01 * t_total_new
    t2 = 0.99 * t_total_new

    sequence = [
        ('Rz', 12, t1),
        ('Rcz', 12, t2)
    ]

    return sequence
哦    
'''


tZ = np.pi / omegaz
tH = np.pi / (omegax * np.sqrt(2))
tpi2 = np.pi / (2 * omegacz)
psi = (2 * np.pi ) - (tpi2 * omegaz) % (2 * np.pi)
tpsi = psi / omegaz
thetacz = (2 * tpsi) * omegacz
tcz = (2 * np.pi - thetacz) / omegacz

tx = 2 * tH + tZ

CZ = [
    ('Rcz', 12,2 * tpi2),
    ('Rz', 12,2 * tpsi),
]

H11 = [('Rzx', 1,tH)]

H22 = [('Rzx', 2,tH)]

H12 = [('Rzx', 12,tH)]

X11 = [
    ('Rzx', 1,tH),
    ('Rz', 12,tZ),
    ('Rzx', 1,tH),
]

X22 = [
    ('Rzx', 2,tH),
    ('Rz', 12,tZ),
    ('Rzx', 2,tH),
]

X12 = [
    ('Rzx', 12,tH),
    ('Rz', 12,tZ),
    ('Rzx', 12,tH),
]

#下面来搞一搞单个的 Rz 门

'''
t1_2 = (2 *((4* np.pi) - 2 * (((2 * tH + tZ) * (omegaz ) )% (2 * np.pi)) )) / (3 * omegaz)

RZ1 = [('Rz', 12,t1_2),] + X1 + [('Rz', 12,t1_2),] + X1
'''


def tint (theta):
    ton = (20 * np.pi - theta) / (2 * omegaz) - tx

    ton = ton % ((2 * np.pi) / omegaz)

    return ton

def Rz1 (theta):
    return [
    ('Rz', 12, tint(theta)),
    ] + X22 + [
    ('Rz', 12, tint(theta)),
    ] + X22

def Rz2 (theta):
    return [
    ('Rz', 12, tint(theta)),
    ] + X11 + [
    ('Rz', 12, tint(theta)),
    ] + X11


Z2 = Rz2(np.pi)
Z1 = Rz1(np.pi)

'''
这里overkill了
X1 = X11 + [
    ('Rz', 12, tonly(thetax)),
] + X11 + [
    ('Rz', 12, tonly(thetax)),
] + X11

X2 = X22 + [
    ('Rz', 12, tonly(thetax)),
] + X22 + [
    ('Rz', 12, tonly(thetax)),
] + X22
'''

thetaH = (tH) * omegaz


H2 = H11 + Rz2(thetaH)

# H2 = Rz2(np.pi) + H22 + Rz1(thetaH) + Rz2(np.pi)

H2 = H22 + Rz1(thetaH)


my_sequence = H2 + CZ + H2

alpha00 = 1
alpha01 = 1
alpha10 = 1
alpha11 = 2

psi_init = (  alpha00 * tensor(basis(N, 0), basis(N, 0)) +
              alpha01 * tensor(basis(N, 0), basis(N, 1)) +
              alpha10 * tensor(basis(N, 1), basis(N, 0)) +
              alpha11 * tensor(basis(N, 1), basis(N, 1)) ).unit()

gate_matrix = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
]
U_gate = Qobj(gate_matrix, dims=[[2, 2], [2, 2]])

fid_x1 = calculate_gate_fidelity(my_sequence, U_gate)
print(f"\nGate Fidelity: {fid_x1:.6f}")
# results, names = run_sequence(my_sequence, psi_init)

# plot_2q_population_final(results, names)

