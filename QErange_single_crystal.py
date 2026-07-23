import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import plotly.graph_objects as go

# ==============================
# CTAX table
# [M2, S2limit, Ei]
# ==============================

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

#----------------------------------------
# Instrument condition
#----------------------------------------

Ef = 5.0          # meV
hw = 10          # meV
Ei = Ef + hw

conv = 0.6947

ki = conv*np.sqrt(Ei)
kf = conv*np.sqrt(Ef)

#----------------------------------------
# interpolate S2 limit
#----------------------------------------

idx = np.argsort(data[:,2])

interp = interp1d(
    data[idx,2],
    data[idx,1],
    kind='linear',
    fill_value='extrapolate'
)

S2max = float(interp(Ei))
S2min = 8.0

print(f"Ei = {Ei:.2f} meV")
print(f"S2max = {S2max:.2f} deg")

#----------------------------------------
# Q vector in laboratory frame
#----------------------------------------

def Qvector(two_theta_deg):

    tt = np.deg2rad(two_theta_deg)

    qx = ki - kf*np.cos(tt)
    qy = -kf*np.sin(tt)

    return np.array([qx,qy])

Qmin = Qvector(S2min)
Qmax = Qvector(S2max)

#=========================
# Crystal
#=========================

a = 4.82
b = 4.82
c = 23.38

alpha = 90
beta  = 90
gamma = 120

# scattering plane

u_hkl = np.array([1,0,0])
v_hkl = np.array([0,1,0])

def reciprocal_vectors(a,b,c,alpha,beta,gamma):

    alpha=np.deg2rad(alpha)
    beta=np.deg2rad(beta)
    gamma=np.deg2rad(gamma)

    A=np.array([a,0,0])

    B=np.array([
        b*np.cos(gamma),
        b*np.sin(gamma),
        0
    ])

    cx=c*np.cos(beta)

    cy=c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)

    cz=np.sqrt(c*c-cx*cx-cy*cy)

    C=np.array([cx,cy,cz])

    V=np.dot(A,np.cross(B,C))

    astar=2*np.pi*np.cross(B,C)/V
    bstar=2*np.pi*np.cross(C,A)/V
    cstar=2*np.pi*np.cross(A,B)/V

    return astar,bstar,cstar

astar,bstar,cstar = reciprocal_vectors(
    a,b,c,
    alpha,beta,gamma
)

u = (
    u_hkl[0]*astar +
    u_hkl[1]*bstar +
    u_hkl[2]*cstar
)

v = (
    v_hkl[0]*astar +
    v_hkl[1]*bstar +
    v_hkl[2]*cstar
)

ex = u/np.linalg.norm(u)

ey = v - np.dot(v,ex)*ex
ey = ey/np.linalg.norm(ey)

#----------------------------------------
# rotate by sample angle
#----------------------------------------

phi = np.linspace(0,360,721)

xmin=[]
ymin=[]

xmax=[]
ymax=[]

for p in phi:

    r=np.deg2rad(p)

    R=np.array([
        [np.cos(r),-np.sin(r)],
        [np.sin(r), np.cos(r)]
    ])

    q=R@Qmin
    xmin.append(q[0])
    ymin.append(q[1])

    q=R@Qmax
    xmax.append(q[0])
    ymax.append(q[1])

xmin=np.array(xmin)
ymin=np.array(ymin)

xmax=np.array(xmax)
ymax=np.array(ymax)

#----------------------------------------
# fill accessible region
#----------------------------------------

polyx=np.concatenate([xmax,xmin[::-1]])
polyy=np.concatenate([ymax,ymin[::-1]])

plt.figure(figsize=(7,7))

plt.fill(
    polyx,
    polyy,
    color="lightgray",
    alpha=0.6,
    label="Accessible region"
)

# Reverse lattice

Qmax = np.linalg.norm(Qvector(S2max))

U = (
    u_hkl[0]*astar +
    u_hkl[1]*bstar +
    u_hkl[2]*cstar
)

V = (
    v_hkl[0]*astar +
    v_hkl[1]*bstar +
    v_hkl[2]*cstar
)

U_len = np.linalg.norm(U)
V_len = np.linalg.norm(V)
Mmax = int(np.ceil(Qmax/U_len))+1
Nmax = int(np.ceil(Qmax/V_len))+1

for m in range(-Mmax,Mmax+1):
    for n in range(-Nmax,Nmax+1):

        hkl = m*u_hkl + n*v_hkl

        G = (
            hkl[0]*astar +
            hkl[1]*bstar +
            hkl[2]*cstar
        )

        # 念のためQ範囲外を除外
        if np.linalg.norm(G) > Qmax:
            continue

        x=np.dot(G,ex)
        y=np.dot(G,ey)

        plt.plot(x,y,'ko',ms=5)

plt.plot(xmax,ymax,lw=2,label="S2 max")
plt.plot(xmin,ymin,lw=2,label="S2 min")

plt.scatter(0,0,c='r',s=50)

plt.gca().set_aspect("equal")

plt.xlabel(r"$Q_x$ ($\AA^{-1}$)")
plt.ylabel(r"$Q_y$ ($\AA^{-1}$)")

title = (
    f"a={a:.3f} Å, b={b:.3f} Å, c={c:.3f} Å\n"
    f"α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}°\n"
    f"Plane: ({u_hkl[0]},{u_hkl[1]},{u_hkl[2]}) / "
    f"({v_hkl[0]},{v_hkl[1]},{v_hkl[2]})\n"
    f"Ef={Ef:.2f} meV, ℏω={hw:.2f} meV"
)

plt.title(title)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()