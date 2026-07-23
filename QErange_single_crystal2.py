import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import plotly.graph_objects as go

# ==============================
# parameters
# ==============================

a = 4.82
b = 4.82
c = 23.38

alpha = 90
beta  = 90
gamma = 120

# scattering plane

u_hkl = np.array([1,0,0])
v_hkl = np.array([0,1,0])

Ef = 5.0          # meV

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
# interpolate S2 limit
#----------------------------------------

idx = np.argsort(data[:,2])

S2interp = interp1d(
    data[idx,2],      # Ei
    data[idx,1],      # S2 limit
    kind="linear",
    fill_value="extrapolate"
)

S2min = 5

#----------------------------------------
# calculation range
#----------------------------------------

def Qvector(two_theta_deg, Ei):

    ki = 0.6947*np.sqrt(Ei)
    kf = 0.6947*np.sqrt(Ef)

    tt = np.deg2rad(two_theta_deg)

    qx = ki - kf*np.cos(tt)
    qy = -kf*np.sin(tt)

    return np.array([qx,qy])

def calc_Q_region(hw):

    Ei = Ef + hw

    S2max = float(S2interp(Ei))

    qmin = Qvector(S2min, Ei)
    qmax = Qvector(S2max, Ei)

    phi=np.linspace(0,360,721)

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

        q=R@qmin
        xmin.append(q[0])
        ymin.append(q[1])

        q=R@qmax
        xmax.append(q[0])
        ymax.append(q[1])

    return (
        np.array(xmin),
        np.array(ymin),
        np.array(xmax),
        np.array(ymax),
        S2max
    )

hw_list=np.arange(3.6-Ef,20.1-Ef,0.2)

regions=[]
S2_list=[]
Qmax_list=[]

for hw in hw_list:
    Ei = Ef + hw
    result = calc_Q_region(hw)
    
    regions.append(result[:4])
    S2_list.append(result[4])

    S2max = float(S2interp(Ei))

    qmax = np.linalg.norm(
            Qvector(S2max,Ei)
        )
    Qmax_list.append(qmax)

#=========================
# Crystal
#=========================

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
    Qmin = Qvector(5, Ei)
    Qmax = Qvector(S2max, Ei)
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

Qmax = np.linalg.norm(Qvector(S2max, Ei))

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

fig=go.Figure()

xmin,ymin,xmax,ymax=regions[0]

Gx_points = []
Gy_points = []
labels = []
Qplot = 2 * Qmax

for m in range(-Mmax, Mmax+1):
    for n in range(-Nmax, Nmax+1):

        hkl = m*u_hkl + n*v_hkl

        G = (
            hkl[0]*astar +
            hkl[1]*bstar +
            hkl[2]*cstar
        )

        if np.linalg.norm(G) > Qplot:
            continue

        x = np.dot(G, ex)
        y = np.dot(G, ey)

        Gx_points.append(x)
        Gy_points.append(y)
        labels.append(
            f"({int(hkl[0])},{int(hkl[1])},{int(hkl[2])})"
        )

fig.add_trace(
    go.Scatter(
        x=np.concatenate([xmax,xmin[::-1]]),
        y=np.concatenate([ymax,ymin[::-1]]),
        fill="toself",
        name="Accessible Q",
        line=dict(width=0)
    )
)

fig.add_trace(
    go.Scatter(
        x=Gx_points,
        y=Gy_points,
        mode="markers+text",
        text=labels,
        textposition="top center",
        name="Reciprocal lattice"
    )
)

fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        name=f"S2 range = {S2min:.1f} - {S2max:.1f}°"
    )
)

steps=[]

for i,hw in enumerate(hw_list):

    xmin,ymin,xmax,ymax = regions[i]

    step=dict(
        method="update",
        args=[
            {
                "x":[
                    np.concatenate([xmax,xmin[::-1]]),
                    Gx_points,
                    [None]
                ],
                "y":[
                    np.concatenate([ymax,ymin[::-1]]),
                    Gy_points,
                    [None]
                ],
                "name":[
                    "Accessible Q",
                    "Reciprocal lattice",
                    f"S2 range = {S2min:.1f} - {S2_list[i]:.1f}°"
                ]
            }
        ],
        label=f"{hw:.1f} meV"
    )

    steps.append(step)

fig.update_layout(
    title=dict(
        text=(
            f"Ef={Ef:.1f} meV | "
            f"a={a:.3f}, b={b:.3f}, c={c:.3f} Å | "
            f"α={alpha:.1f}, β={beta:.1f}, γ={gamma:.1f}°<br>"
            f"Plane: ({u_hkl[0]},{u_hkl[1]},{u_hkl[2]})-({v_hkl[0]},{v_hkl[1]},{v_hkl[2]})"
        ),
        x=0.5
    )
)

fig.update_layout(
    sliders=[
        dict(
            active=0,
            steps=steps,
            currentvalue=dict(
                prefix="ħω = ",
                #suffix=" meV",
                font=dict(size=16)
            )
        )
    ]
)

Qplot = 1.2 * max(Qmax_list)

fig.update_layout(
    xaxis_title="Qx (Å⁻¹)",
    yaxis_title="Qy (Å⁻¹)",
    yaxis_scaleanchor="x",
    width=700,
    height=700,

    xaxis=dict(
        range=[-Qplot, Qplot],
        zeroline=True,
        dtick=1
    ),

    yaxis=dict(
        range=[-Qplot, Qplot],
        zeroline=True,
        dtick=1
    )
)

fig.show()