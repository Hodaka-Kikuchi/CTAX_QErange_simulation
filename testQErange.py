import numpy as np
import matplotlib.pyplot as plt

conv = 0.6947

Ei_list = np.arange(5, 22, 2)

plt.figure(figsize=(7,7))

for Ei in Ei_list:

    ki = conv*np.sqrt(Ei)

    hw = np.linspace(Ei-5, Ei-3, 300)

    qmin = []
    qmax = []

    for h in hw:

        Ef = Ei - h
        kf = conv*np.sqrt(Ef)

        # 2theta = 0°
        q0 = abs(ki-kf)

        # 2theta = 90°
        q90 = np.sqrt(ki**2+kf**2)

        qmin.append(q0)
        qmax.append(q90)

    plt.fill_betweenx(hw, qmin, qmax,
                      alpha=0.35,
                      label=f"Ei={Ei:g} meV")

plt.xlabel(r"$Q\ (\AA^{-1})$", fontsize=14)
plt.ylabel(r"$\hbar\omega$ (meV)", fontsize=14)

plt.xlim(left=0)
plt.ylim(0,18)
plt.xlim(0,4)
plt.grid(alpha=0.3)
plt.legend(ncol=2)

plt.tight_layout()
plt.show()

#######################################

import numpy as np
import matplotlib.pyplot as plt

conv = 0.6947

Ef_list = [3,4,5]

colors = ['tab:blue','tab:orange','tab:green']

plt.figure(figsize=(7,7))

for Ef, color in zip(Ef_list, colors):

    hw = np.linspace(0, 20-Ef, 500)

    kf = conv*np.sqrt(Ef)

    qmin = []
    qmax = []

    for h in hw:

        Ei = Ef + h
        ki = conv*np.sqrt(Ei)

        q0 = abs(ki-kf)
        q90 = np.sqrt(ki**2 + kf**2)

        qmin.append(q0)
        qmax.append(q90)

    plt.fill_betweenx(hw,
                      qmin,
                      qmax,
                      color=color,
                      alpha=0.35,
                      label=f"Ef = {Ef} meV")

    # 境界線を描くと見やすい
    plt.plot(qmin, hw, color=color, lw=2)
    plt.plot(qmax, hw, color=color, lw=2)

plt.xlabel(r"$Q\ (\AA^{-1})$", fontsize=14)
plt.ylabel(r"$\hbar\omega$ (meV)", fontsize=14)

plt.xlim(0,4)
plt.ylim(0,18)

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()