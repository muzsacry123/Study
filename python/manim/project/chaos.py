from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
 
# 设置重力参数为9.8
G = 9.8
# 设置摆1长度为1，5厘米
L1 = 1.5
# 设置摆2长度为1，0厘米
L2 = 1.0
# 设置摆1质量为1
M1 = 1.0
# 设置摆2质量为1
M2 = 1.0
 
 
def derivs(init_state, init_time):
    dydx = np.zeros_like(init_state)

    dydx[0] = init_state[1]
 
    del_ = init_state[2] - init_state[0]
 
    den1 = (M1 + M2) * L1 - M2 * L1 * cos(del_) * cos(del_)
 
    dydx[1] = (M2 * L1 * init_state[1] * init_state[1] * sin(del_) * cos(del_) +
 
               M2 * G * sin(init_state[2]) * cos(del_) +
 
               M2 * L2 * init_state[3] * state[3] * sin(del_) -
 
               (M1 + M2) * G * sin(init_state[0])) / den1
 
    dydx[2] = init_state[3]
 
    den2 = (L2 / L1) * den1
 
    dydx[3] = (-M2 * L2 * init_state[3] * init_state[3] * sin(del_) * cos(del_) +
 
               (M1 + M2) * G * sin(init_state[0]) * cos(del_) -
 
               (M1 + M2) * L1 * init_state[1] * init_state[1] * sin(del_) -
 
               (M1 + M2) * G * sin(init_state[2])) / den2
 
    return dydx
 
 
t_steep = 0.01  # 时间采样步长0.01
 
t = np.arange(0.0, 100, t_steep)
 
th1 = 90.0  # 摆1的初始摆角
 
w1 = 0.0  # 摆1的初始角速度
 
th2 = 0.0  # 摆2的初始摆角
 
w2 = 0.0  # 摆2的初始角速度
 
state = np.radians([th1, w1, th2, w2])
 
y = integrate.odeint(derivs, state, t)
 
x1 = L1 * sin(y[:, 0])
 
y1 = -L1 * cos(y[:, 0])
 
x2 = L2 * sin(y[:, 2]) + x1
 
y2 = -L2 * cos(y[:, 2]) + y1
 
fig = plt.figure()
 
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-3, 3), ylim=(-3, 3))
 
ax.grid()  # 坐标系中是否需要网格
 
line, = ax.plot([], [], 'o-', lw=2)
 
time_template = 'time = %.1fs'
 
time_text = ax.text(0.1, 0.9, '', transform=ax.transAxes)
 
# 可添加一个新的曲线对象来绘制摆2的轨迹
line2, = ax.plot([], [], '-', lw=1, color='black')
 
 
def init():
    line.set_data([], [])
    line2.set_data([], [])  # 初始化摆2的轨迹曲线
    time_text.set_text('')
    return line, line2, time_text
 
 
def animate(i):
    this_x = [0, x1[i], x2[i]]
    this_y = [0, y1[i], y2[i]]
    line.set_data(this_x, this_y)
 
    # 更新摆2的轨迹曲线数据
    traj_x = x2[:i + 1]
    traj_y = y2[:i + 1]
    line2.set_data(traj_x, traj_y)
 
    time_text.set_text(time_template % (i * t_steep))
    return line, line2, time_text
 
 
# 输出双摆动画,并且设置动画为每秒40帧（interval=40）
ani = animation.FuncAnimation(fig, animate, len(y), interval=60, blit=True, init_func=init)
 
# 用于保存输出的动画，调用ffmpeg写入磁盘，动画fps=40
file_path = 'E:/double_pendulum.mp4'
ani.save(file_path, fps=40, writer="ffmpeg")
 
plt.show()