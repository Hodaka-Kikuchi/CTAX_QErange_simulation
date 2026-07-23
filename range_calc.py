import numpy as np
import matplotlib.pyplot as plt

# データ（M2, S2limit, Ei）
data = np.array([
[-35.01,27,20.08],
[-36.03,30,19.00],
[-36.53,32,18.50],
[-37,32,18.05],
[-37.05,33,18.00],
[-37.6,34,17.50],
[-38.17,35,17.00],
[-38.76,36,16.50],
[-39,36,16.31],
[-39.39,38,16.00],
[-40.04,39,15.50],
[-40,38,15.53],
[-41.01,41,14.81],
[-41.46,42,14.50],
[-42.23,44,14.00],
[-43,45,13.53],
[-43.05,46,13.50],
[-43.91,47,13.00],
[-44.82,49,12.50],
[-45,50,12.41],
[-45.8,52,12.00],
[-46.84,54,11.50],
[-47,54,11.43],
[-47.96,56,11.00],
[-48.43,57,10.80],
[-48.92,58,10.60],
[-48.99,58,10.57],
[-49.41,59,10.40],
[-49.93,60,10.20],
[-50.01,60,10.17],
[-50.46,62,10.00],
[-51.01,63,9.80],
[-51.58,64,9.60],
[-52.16,65,9.40],
[-52.77,67,9.20],
[-52.99,68,9.13],
[-53.4,69,9.00],
[-54.05,72,8.80],
[-54.73,73,8.60],
[-55.01,75,8.52],
[-55.43,79,8.40],
[-56.16,82,8.20],
[-56.92,87,8.00],
[-57,89,7.98],
[-57.72,89,7.80],
[-58.54,90,7.60],
[-59.02,90,7.49],
[-59.41,90,7.40],
[-60,90,7.27],
[-60.31,94,7.20],
[-61.02,96,7.05],
[-61.26,97,7.00],
[-62.25,98,6.80],
[-62.98,99,6.66],
[-63.3,99,6.60],
[-64.39,102,6.40],
[-64.97,102,6.30],
[-65.55,106,6.20],
[-66.78,109,6.00],
[-67.03,109,5.96],
[-68.75,111,5.70],
[-68.96,112,5.67],
[-70.02,112,5.52],
[-75.03,112,4.90],
[-84.36,112,4.03],
[-89.28,112,3.68],
])

M2 = data[:,0]
S2 = data[:,1]
Ei = data[:,2]

Ef = 4.8  # meV

conv = 0.6947
kf = conv * np.sqrt(Ef)


Q_min = []
Q_max = []
hw = []


for M2, S2limit, Ei in data:

    ki = conv * np.sqrt(Ei)

    # energy transfer
    w = Ei - Ef

    # scattering angle range
    theta_min = np.deg2rad(5)
    theta_max = np.deg2rad(S2limit)

    # Q minimum
    qmin = np.sqrt(
        ki**2 + kf**2
        - 2*ki*kf*np.cos(theta_min)
    )

    # Q maximum
    qmax = np.sqrt(
        ki**2 + kf**2
        - 2*ki*kf*np.cos(theta_max)
    )

    Q_min.append(qmin)
    Q_max.append(qmax)
    hw.append(w)


Q_min = np.array(Q_min)
Q_max = np.array(Q_max)
hw = np.array(hw)


# energy transfer順に並べ替え
idx = np.argsort(hw)

hw = hw[idx]
Q_min = Q_min[idx]
Q_max = Q_max[idx]


plt.figure(figsize=(7,5))


# QE range fill
plt.fill_betweenx(
    hw,
    Q_min,
    Q_max,
    color="lightgray",
    alpha=0.5,
    label="Accessible QE range"
)


# minimum Q boundary
plt.plot(
    Q_min,
    hw,
    linewidth=2,
    label=r"$Q_{min}$ ($2\theta=5^\circ$)"
)


# maximum Q boundary
plt.plot(
    Q_max,
    hw,
    linewidth=2,
    label=r"$Q_{max}$ ($2\theta=S2_{limit}$)"
)


# 上下端を閉じる線
plt.plot(
    [Q_min[-1], Q_max[-1]],
    [hw[-1], hw[-1]],
    linewidth=1
)

plt.plot(
    [Q_min[0], Q_max[0]],
    [hw[0], hw[0]],
    linewidth=1
)


plt.xlabel(r"$Q$ ($\AA^{-1}$)")
plt.ylabel(r"$\hbar\omega$ (meV)")

plt.title(
    f"CTAX Q-E range ($E_f={Ef}$ meV)"
)

plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np

def reciprocal_lattice(a,b,c,alpha,beta,gamma):

    alpha=np.deg2rad(alpha)
    beta=np.deg2rad(beta)
    gamma=np.deg2rad(gamma)

    avec=np.array([a,0,0])

    bvec=np.array([
        b*np.cos(gamma),
        b*np.sin(gamma),
        0
    ])

    cx=c*np.cos(beta)

    cy=c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)

    cz=np.sqrt(c**2-cx**2-cy**2)

    cvec=np.array([cx,cy,cz])

    V=np.dot(avec,np.cross(bvec,cvec))

    astar=2*np.pi*np.cross(bvec,cvec)/V
    bstar=2*np.pi*np.cross(cvec,avec)/V
    cstar=2*np.pi*np.cross(avec,bvec)/V

    return astar,bstar,cstar

# Lattice
a, b, c = 4.82,4.82,23.38
alpha,beta,gamma = 90,90,120

# Scattering plane
u = [1,0,0]
v = [0,1,0]

# Bragg peak
h0,k0,l0 = 0,0,0

# Instrument
Ef = 4.8          # meV
hw = 5.0          # Energy transfer

# Sample rotation
S1min = -180
S1max = 180
S1step = 1

from scipy.interpolate import interp1d

S2interp = interp1d(
    data[:,2],      # Ei
    data[:,1],      # S2 limit
    kind="linear",
    fill_value="extrapolate"
)

S2max = float(S2interp(Ei))

