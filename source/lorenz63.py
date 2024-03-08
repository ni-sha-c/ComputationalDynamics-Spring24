import torch
import matplotlib.pyplot as plt

# Define the Lorenz-63 equations
def lorenz63(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return torch.tensor([dxdt, dydt, dzdt])

# Fourth-order Runge-Kutta method
def rk4_step(func, t, y, dt):
    k1 = func(t, y) * dt
    k2 = func(t + 0.5*dt, y + 0.5*k1) * dt
    k3 = func(t + 0.5*dt, y + 0.5*k2) * dt
    k4 = func(t + dt, y + k3) * dt
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

# Integrate the Lorenz-63 equations using RK4
def integrate_lorenz63(func, y0, t0, tf, dt):
    t = torch.arange(t0, tf, dt)
    y = torch.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = rk4_step(func, t[i-1], y[i-1], dt)
    return t, y

# Initial conditions
y0 = torch.tensor([1.0, 1.0, 1.0])

# Time integration parameters
t0 = 0.0
tf = 40.0
dt = 0.01

# Integrate the Lorenz-63 system
t, y = integrate_lorenz63(lorenz63, y0, t0, tf, dt)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, y[:, 0], label='x')
plt.plot(t, y[:, 1], label='y')
plt.plot(t, y[:, 2], label='z')
plt.title('Lorenz-63 System')
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.legend()
plt.grid(True)
plt.show()


