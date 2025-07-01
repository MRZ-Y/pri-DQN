import control
import matplotlib.pyplot as plt
import numpy as np

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义一阶系统
num = [1]
den = [1, 1]
sys = control.TransferFunction(num, den)

# 定义 PID 控制器参数
Kp = 3.0
Ki = 5
Kd = 0.1

# 积分限幅参数
integral_limit = 10


# 自定义 PID 控制器函数
def pid_controller(error, prev_error, integral):
    global Kp, Ki, Kd
    # 计算积分项
    integral += error
    # 积分限幅
    integral = np.clip(integral, -integral_limit, integral_limit)
    # 计算微分项
    derivative = error - prev_error
    # 计算控制输出
    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral


# 定义时间范围
t = np.linspace(0, 10, 1000)
dt = t[1] - t[0]

# 构造幅度为 5 的阶跃输入信号
u = 5 * np.ones_like(t)

# 初始化变量
prev_error = 0
integral = 0
y = np.zeros_like(t)
y[0] = 0

# 仿真循环
noise_std = 0.1  # 噪声标准差
for i in range(1, len(t)):
    error = u[i] - y[i - 1]
    control_output, integral = pid_controller(error, prev_error, integral)
    # 计算系统响应
    _, yout = control.forced_response(sys, T=[t[i - 1], t[i]], U=[control_output, control_output], X0=y[i - 1])
    y[i] = yout[-1]
    # 添加高斯白噪声
    y[i] += np.random.normal(0, noise_std)
    prev_error = error

# 计算评价指标
# 上升时间：系统输出首次达到稳态值的 90% 所需的时间
rise_time = None
for i in range(len(t)):
    if y[i] >= 0.9 * u[-1]:
        rise_time = t[i]
        break

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

# 稳态误差：系统达到稳态后，输出值与设定值之间的误差
steady_state_error = np.abs(y[-1] - u[-1])

# 打印评价指标
print(f"上升时间: {rise_time:.2f} s")
print(f"峰值时间: {peak_time:.2f} s")
print(f"超调量: {overshoot:.2f} %")
if settling_time is not None:
    print(f"调整时间: {settling_time:.2f} s")
else:
    print("调整时间: 未在仿真时间内达到稳定条件")
print(f"稳态误差: {steady_state_error:.2f}")

# 绘制响应曲线
plt.plot(t, y)
plt.title('PID 控制的一阶系统阶跃响应（设定值为 5，添加积分限幅和噪声）')
plt.xlabel('时间 (s)')
plt.ylabel('输出')
plt.grid(True)
plt.show()
