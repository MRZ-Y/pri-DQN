import control
import matplotlib.pyplot as plt
import numpy as np
from pyswarm import pso

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义一阶系统
num = [1]
den = [1, 1]
sys = control.TransferFunction(num, den)

# 积分限幅参数
integral_limit = 10

# 定义时间范围
t = np.linspace(0, 10, 1000)
dt = t[1] - t[0]

# 构造幅度为 5 的阶跃输入信号
u = 5 * np.ones_like(t)

# 噪声标准差
noise_std = 0.1

# 全局变量，用于记录迭代次数
iteration_count = 0

# 全局变量，粒子群数量
swarmsize = 50

# 全局变量，最大迭代次数
max_iterations = 10

# 记录上一轮的评分
last_score = 0

# 自定义 PID 控制器函数
def pid_controller(error, prev_error, integral, Kp, Ki, Kd):
    # 计算积分项
    integral += error
    # 积分限幅
    integral = np.clip(integral, -integral_limit, integral_limit)
    # 计算微分项
    derivative = error - prev_error
    # 计算控制输出
    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral

# 仿真函数
def simulate(Kp, Ki, Kd):
    # 固化噪声影响
    np.random.seed(42)
    # 初始化变量
    prev_error = 0
    integral = 0
    y = np.zeros_like(t)
    y[0] = 0

    # 仿真循环
    for i in range(1, len(t)):
        error = u[i] - y[i - 1]
        control_output, integral = pid_controller(error, prev_error, integral, Kp, Ki, Kd)
        # 计算系统响应
        _, yout = control.forced_response(sys, T=[t[i - 1], t[i]], U=[control_output, control_output], X0=y[i - 1])
        y[i] = yout[-1]
        # 添加高斯白噪声
        y[i] += np.random.normal(0, noise_std)
        prev_error = error

    return y

# 评价函数
def evaluation_function(params):
    global iteration_count, last_score
    iteration_count += 1
    Kp, Ki, Kd = params
    y = simulate(Kp, Ki, Kd)

    # 计算评价指标
    # 上升时间：系统输出首次达到稳态值的 90% 所需的时间
    rise_time = None
    for i in range(len(t)):
        if y[i] >= 0.9 * u[-1]:
            rise_time = t[i]
            break
    if rise_time is None:
        rise_time = 10  # 假设最大为 10s

    # 峰值时间：系统输出达到第一个峰值所需的时间
    peak_time = t[np.argmax(y)]

    # 超调量：系统输出的最大值超过稳态值的百分比
    overshoot = (np.max(y) - u[-1]) / u[-1] * 100

    # 调整时间：系统输出进入并保持在稳态值的 ±2% 范围内所需的时间
    settling_time = None
    for i in range(len(t)):
        if np.all(np.abs(y[i:] - u[-1]) <= 0.02 * u[-1]):
            settling_time = t[i]
            break
    if settling_time is None:
        settling_time = 10  # 假设最大为 10s

    # 稳态误差：系统达到稳态后，输出值与设定值之间的误差（按比例）
    steady_state_error = np.abs(y[-1] - u[-1]) / u[-1] * 100

    # 各项指标评分
    rise_time_score = max(0, 20 - rise_time * 2)
    peak_time_score = max(0, 20 - peak_time * 2)
    overshoot_score = max(0, 20 - overshoot * 0.5)
    settling_time_score = max(0, 20 - settling_time * 2)
    # 稳态误差评分，error = 0 得 20 分，error 为 2% 得 0 分
    steady_state_error_score = max(0, 20 - steady_state_error * 20)

    total_score = rise_time_score + peak_time_score + overshoot_score + settling_time_score + steady_state_error_score
    last_score = total_score

    if iteration_count % swarmsize == 1 and iteration_count > 1:
        print(f"当前迭代轮次: {iteration_count // swarmsize}/{max_iterations}, 上一轮评分: {last_score:.2f}")
    elif iteration_count == 1:
        print(f"当前迭代轮次: {iteration_count // swarmsize + 1}/{max_iterations}, 初始评分: {last_score:.2f}")

    return -total_score  # 因为 pso 是求最小值，所以取负

# 粒子群优化算法
lb = [0, 0, 0]  # 下限
ub = [10, 10, 10]  # 上限
xopt, fopt = pso(evaluation_function, lb, ub, swarmsize=swarmsize, maxiter=max_iterations)

Kp_opt, Ki_opt, Kd_opt = xopt

# 打印优化后的参数
print(f"优化后的 Kp: {Kp_opt:.2f}")
print(f"优化后的 Ki: {Ki_opt:.2f}")
print(f"优化后的 Kd: {Kd_opt:.2f}")

# 使用优化后的参数进行仿真
y_opt = simulate(Kp_opt, Ki_opt, Kd_opt)

# 绘制响应曲线
plt.plot(t, y_opt)
plt.title('优化后 PID 控制的一阶系统阶跃响应（设定值为 5，添加积分限幅和噪声）')
plt.xlabel('时间 (s)')
plt.ylabel('输出')
plt.grid(True)
plt.show()