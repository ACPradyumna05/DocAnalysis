import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

losses = np.array([1.2224,0.9741,0.8533,0.7512,0.649,0.5915,0.4963,0.4183,0.3683,
    0.318,0.2965,0.2413,0.1958,0.1933,0.149,0.1315,0.1276,0.1104,0.0838,
    0.0904,0.0776,0.0675,0.0623,0.0619])
lrs = np.array([
    0.0001921348,0.0001841091,0.0001760834,0.0001682182,0.0001601926,
    0.0001521669,0.0001441412,0.0001361155,0.0001280898,0.0001200642,
    0.0001120385,0.0001040128,9.5987e-05,8.7961e-05,7.9935e-05,7.1910e-05,
    6.3884e-05,5.5858e-05,4.7833e-05,3.9807e-05,3.1781e-05,2.3756e-05,
    1.5730e-05,7.704e-06
])
epochs = np.array([
    0.28,0.56,0.85,1.12,1.41,1.69,1.97,2.25,2.53,2.81,3.09,
    3.37,3.65,3.94,4.21,4.50,4.78,5.06,5.34,5.62,5.90,6.18,6.46,6.74
])

TOTAL_STEPS = 1246

steps = np.linspace(0, TOTAL_STEPS, len(losses))

smooth_loss = gaussian_filter1d(losses, sigma=2)
smooth_lr = gaussian_filter1d(lrs, sigma=1)

epoch_steps = (epochs / epochs[-1]) * TOTAL_STEPS


plt.style.use("dark_background")
fig, axs = plt.subplots(3, 1, figsize=(15, 18))

axs[0].plot(steps, smooth_loss, color="red", linewidth=2)
axs[0].set_title("Smoothed Loss Curve")
axs[0].set_xlabel("Step")
axs[0].set_ylabel("Loss")

for s in epoch_steps:
    axs[0].axvline(s, color="gray", linestyle="--", alpha=0.2)

axs[1].plot(steps, smooth_lr, color="lime", linewidth=2)
axs[1].set_title("Learning Rate Schedule")
axs[1].set_xlabel("Step")
axs[1].set_ylabel("LR")

axs[2].plot(steps, epochs, color="cyan", linewidth=2)
axs[2].set_title("Epoch Progression Timeline")
axs[2].set_xlabel("Step")
axs[2].set_ylabel("Epoch")
for s in epoch_steps:
    axs[2].axvline(s, color="gray", linestyle="--", alpha=0.2)

plt.tight_layout()
plt.show()


window = 3
mov_avg = np.convolve(losses, np.ones(window)/window, mode='same')

upper = mov_avg + (smooth_loss - mov_avg) * 0.5
lower = mov_avg - (smooth_loss - mov_avg) * 0.5

plt.figure(figsize=(15, 7))
plt.title("Advanced Loss Visualization (Moving Avg + Confidence Band)")

plt.plot(steps, mov_avg, color="orange", linewidth=3, label="Moving Average Loss")
plt.fill_between(steps, lower, upper, color="orange", alpha=0.2, label="Confidence Shading")

plt.plot(steps, smooth_loss, color="red", linewidth=1, alpha=0.6, label="Smoothed Loss")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.2)
plt.show()
