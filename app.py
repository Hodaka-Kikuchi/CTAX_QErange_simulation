import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import json
import os

# デバックの手順
# cd C:\Users\h34\Documents\Python\CTAX
# python -m streamlit run app.py

# QErange_single_crystal2.pyのweb版

# global data

# ==============================
# CTAX table
# [M2, S2limit, Ei]
# ==============================

# common function
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

#----------------------------------------
# file
#----------------------------------------

# Instrument folder
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

INSTRUMENT_DIR = os.path.join(
    BASE_DIR,
    "instruments"
)

instrument_files = sorted(
    [
        f.replace(".json","")
        for f in os.listdir(INSTRUMENT_DIR)
        if f.endswith(".json")
    ]
)

# ==============================
# UI
# ==============================

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 350px;
            max-width: 350px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(
    page_title="CTAX QErange simulator",
    layout="wide"
)

st.title("TAS Q-E Range Simulator")

display_names = {
    "CTAX": "CTAX@HFIR",
    "HB1": "HB-1@HFIR",
    "HODACA": "HODACA@JRR-3"
}

with st.sidebar:

    col1, col2 = st.columns([1,1])

    with col1:

        instrument = st.selectbox(
            "Instrument",
            instrument_files,
            format_func=lambda x: display_names.get(x, x)
        )

    with col2:

        mode = st.radio(
            "Type",
            [
                "Single crystal",
                "Powder"
            ],
            horizontal=True
        )

#----------------------------------------
# interpolate S2 limit
#----------------------------------------

with open(
    os.path.join(
        INSTRUMENT_DIR,
        f"{instrument}.json"
    ),
    "r"
) as f:
    instrument_data = json.load(f)


data = np.array(
    [
        [
            x["S2limit"],
            x["Ei"]
        ]
        for x in instrument_data["configuration"]
    ]
)

idx = np.argsort(data[:,1])

S2interp = interp1d(
    data[idx,1],      # Ei
    data[idx,0],      # S2 limit
    kind="linear",
    fill_value="extrapolate"
)

if mode == "Single crystal":

    with st.sidebar:

        st.header("Lattice constant")

        col1, col2, col3 = st.columns(3)

        with col1:
            a = st.number_input(
                "a (Å)",
                value=5.00,
                step=0.01
            )

        with col2:
            b = st.number_input(
                "b (Å)",
                value=6.00,
                step=0.01
            )

        with col3:
            c = st.number_input(
                "c (Å)",
                value=7.00,
                step=0.01
            )


        #st.write("Lattice angle (deg)")

        col1, col2, col3 = st.columns(3)

        with col1:
            alpha = st.number_input(
                "α (deg)",
                value=90.0,
                step=1.0
            )

        with col2:
            beta = st.number_input(
                "β (deg)",
                value=90.0,
                step=1.0
            )

        with col3:
            gamma = st.number_input(
                "γ (deg)",
                value=90.0,
                step=1.0
            )

        st.sidebar.header("Propagation vector")

        col1, col2, col3, col4 = st.sidebar.columns(4)

        with col1:
            show_k = st.checkbox(
                "show",
                value=False
            )
            
        with col2:
            k_h = st.number_input(
                "h",
                value=0.00,
                step=0.01,
                format="%.3f",
                key="k_h"
            )

        with col3:
            k_k = st.number_input(
                "k",
                value=0.00,
                step=0.01,
                format="%.3f",
                key="k_k"
            )

        with col4:
            k_l = st.number_input(
                "l",
                value=0.00,
                step=0.01,
                format="%.3f",
                key="k_l"
            )

        st.header("Scattering plane")

        Ucol1, Ucol2, Ucol3, Ucol4 = st.columns(4)

        with Ucol1:
            st.write("U vector")

        with Ucol2:
            U_h = st.number_input(
                "h",
                value=1,
                step=1,
                key="U_h"
            )

        with Ucol3:
            U_k = st.number_input(
                "k",
                value=0,
                step=1,
                key="U_k"
            )

        with Ucol4:
            U_l = st.number_input(
                "l",
                value=0,
                step=1,
                key="U_l"
            )

        Vcol1, Vcol2, Vcol3, Vcol4 = st.columns(4)

        with Vcol1:
            st.write("V vector")

        with Vcol2:
            V_h = st.number_input(
                "h",
                value=0,
                step=1,
                key="V_h"
            )

        with Vcol3:
            V_k = st.number_input(
                "k",
                value=1,
                step=1,
                key="V_k"
            )

        with Vcol4:
            V_l = st.number_input(
                "l",
                value=0,
                step=1,
                key="V_l"
            )

        #st.write(instrument)
        #st.write(instrument_data)

        st.header("Configuration")

        default_mode = instrument_data.get(
            "energy_mode",
            "Ef fixed"
        )

        default_energy = instrument_data.get(
            "default_energy",
            4.8
        )

        default_S2min = instrument_data.get(
            "default_S2min",
            8.0
        )


        col0, col1, col2 = st.sidebar.columns([2, 2, 1])


        with col0:
            mode = st.radio(
                "",
                ["Ef fixed", "Ei fixed"],
                index=[
                    "Ef fixed",
                    "Ei fixed"
                ].index(default_mode),
                label_visibility="collapsed",
                key=f"energy_mode_{instrument}"
            )


        with col1:
            energy_input = st.number_input(
                f"{'Ef' if mode=='Ef fixed' else 'Ei'} (meV)",
                value=float(default_energy),
                step=0.5,
                key=f"energy_{instrument}"
            )


        with col2:
            lambda_half = st.checkbox(
                "λ/2",
                value=False,
                key=f"lambda_{instrument}"
            )


        if mode == "Ef fixed":

            if lambda_half:
                Ef = 4 * energy_input

                # 元データを直接変更しない方が安全
                data[:,2] = 4 * data[:,2]

            else:
                Ef = energy_input


        else:

            if lambda_half:
                Ei = 4 * energy_input

                data[:,2] = 4 * data[:,2]

            else:
                Ei = energy_input



        S2min = st.number_input(
            "minimum 2θ (deg)",
            value=float(default_S2min),
            step=0.1,
            key=f"S2min_{instrument}"
        )

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

    def calc_Q_region(Ei, Ef, hw, S2min):

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

    #hw_list=np.arange(3.6-Ef,20.1-Ef,0.1)
    if mode=='Ef fixed':
        hw_list=np.arange(0,np.max(data[:,1])-Ef,0.2)
    else:
        hw_list=np.arange(0,Ei,0.2)

    regions=[]
    S2_list=[]
    Qmax_list=[]

    for hw in hw_list:
        if mode=='Ef fixed':
            Ei = Ef + hw
        else:
            Ef = Ei - hw

        result = calc_Q_region(
            Ei,
            Ef,
            hw,
            S2min
        )
        
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

    astar,bstar,cstar = reciprocal_vectors(
        a,b,c,
        alpha,beta,gamma
    )

    u = (
        U_h*astar +
        U_k*bstar +
        U_l*cstar
    )

    v = (
        V_h*astar +
        V_k*bstar +
        V_l*cstar
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
        Qmin = Qvector(S2min, Ei)
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
        U_h*astar +
        U_k*bstar +
        U_l*cstar
    )

    V = (
        V_h*astar +
        V_k*bstar +
        V_l*cstar
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

    mag_x = []
    mag_y = []
    mag_label = []

    for m in range(-Mmax, Mmax+1):
        for n in range(-Nmax, Nmax+1):

            hkl = m*np.array([U_h, U_k, U_l]) + n*np.array([V_h,V_k,V_l])

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

            if show_k:
                        
                kvec = np.array([k_h, k_k, k_l])

                for sign in (+1, -1):

                    hkl_mag = (
                        m*np.array([U_h, U_k, U_l])
                        + n*np.array([V_h, V_k, V_l])
                        + sign*kvec
                    )

                    Gmag = (
                        hkl_mag[0]*astar
                        + hkl_mag[1]*bstar
                        + hkl_mag[2]*cstar
                    )

                    if np.linalg.norm(Gmag) > Qplot:
                        continue

                    x = np.dot(Gmag, ex)
                    y = np.dot(Gmag, ey)

                    mag_x.append(x)
                    mag_y.append(y)
                    
                    mag_label.append(
                        f"({hkl_mag[0]:.2f},{hkl_mag[1]:.2f},{hkl_mag[2]:.2f})"
                    )
                    

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([xmax,xmin[::-1]]),
            y=np.concatenate([ymax,ymin[::-1]]),
            fill="toself",
            name="Accessible Q",
            line=dict(width=0),
            fillcolor="rgba(255,0,0,0.15)"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=Gx_points,
            y=Gy_points,
            mode="markers+text",
            text=labels,
            textposition="top center",
            name="Nuclear Bragg peaks",
            marker=dict(
                color="black",
                size=6
            ),
            textfont=dict(
                color="black",
                size=12
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=mag_x,
            y=mag_y,
            mode="markers",
            marker=dict(
                color="red",
                size=6
            ),
            #hovertext=mag_label,
            #hovertemplate="%{hovertext}<extra></extra>",
            name="Magnetic Bragg peaks"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=f"S2 range = {S2min:.1f} - {S2_list[0]:.1f}°"
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
                        mag_x,
                        [None]
                    ],
                    "y":[
                        np.concatenate([ymax,ymin[::-1]]),
                        Gy_points,
                        mag_y,
                        [None]
                    ],
                    "name":[
                        "Accessible Q",
                        "Nuclear Bragg peaks",
                        "Magnetic Bragg peaks",
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
                f"a={a:.3f}, b={b:.3f}, c={c:.3f} Å<br>"
                f"α={alpha:.1f}, β={beta:.1f}, γ={gamma:.1f}° | "
                f"Plane: ({U_h},{U_k},{U_l})-({V_h},{V_k},{V_l})"
            ),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=14)
        )
    )

    fig.update_layout(
        sliders=[
            dict(
                active=0,
                steps=steps,
                currentvalue=dict(
                    prefix="ħω = ",
                    font=dict(
                        size=16,
                        color="black"
                    )
                ),
                font=dict(
                    size=14,
                    color="black"
                )
            )
        ]
    )

    Qplot = 1.2 * max(Qmax_list)

    fig.update_layout(

        xaxis_title="Qx (Å⁻¹)",
        yaxis_title="Qy (Å⁻¹)",

        xaxis=dict(
            range=[-Qplot, Qplot],
            dtick=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            zeroline=True,
        ),

        yaxis=dict(
            range=[-Qplot, Qplot],
            dtick=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            zeroline=True,
            scaleanchor="x",
            scaleratio=1
        ),

        width=700,
        height=700
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

###############################################################################################################################

else:

    with st.sidebar:

        st.header("Lattice constant")

        col1, col2, col3 = st.columns(3)

        with col1:
            a = st.number_input(
                "a (Å)",
                value=5.00,
                step=0.01
            )

        with col2:
            b = st.number_input(
                "b (Å)",
                value=6.00,
                step=0.01
            )

        with col3:
            c = st.number_input(
                "c (Å)",
                value=7.00,
                step=0.01
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            alpha = st.number_input(
                "α (deg)",
                value=90.0,
                step=1.0
            )

        with col2:
            beta = st.number_input(
                "β (deg)",
                value=90.0,
                step=1.0
            )

        with col3:
            gamma = st.number_input(
                "γ (deg)",
                value=90.0,
                step=1.0
            )

        st.sidebar.header("Propagation vector")
        
        col1, col2, col3, col4 = st.sidebar.columns(4)

        with col1:
            show_k = st.checkbox(
                "show",
                value=False
            )
            
        with col2:
            k_h = st.number_input(
                "h",
                value=0.00,
                step=0.01,
                format="%.3f",
                key="k_h"
            )

        with col3:
            k_k = st.number_input(
                "k",
                value=0.00,
                step=0.01,
                format="%.3f",
                key="k_k"
            )

        with col4:
            k_l = st.number_input(
                "l",
                value=0.00,
                step=0.01,
                format="%.3f",
                key="k_l"
            )

        st.header("Configuration")

        default_mode = instrument_data.get(
            "energy_mode",
            "Ef fixed"
        )

        default_energy = instrument_data.get(
            "default_energy",
            4.8
        )

        default_S2min = instrument_data.get(
            "default_S2min",
            8.0
        )

        col0, col1 = st.columns([2,2])

        with col0:

            mode = st.radio(
                "",
                ["Ef fixed","Ei fixed"],
                index=["Ef fixed","Ei fixed"].index(default_mode),
                label_visibility="collapsed",
                key=f"powder_mode_{instrument}"
            )

        with col1:

            energy_input = st.number_input(
                f"{'Ef' if mode=='Ef fixed' else 'Ei'} (meV)",
                value=float(default_energy),
                step=0.1,
                key=f"powder_energy_{instrument}"
            )

        S2min = st.number_input(
            "minimum 2θ (deg)",
            value=float(default_S2min),
            step=0.1,
            key=f"powder_S2min_{instrument}"
        )

    #############################

    astar,bstar,cstar = reciprocal_vectors(
            a,b,c,
            alpha,beta,gamma
        )

    astar_len = np.linalg.norm(astar)
    bstar_len = np.linalg.norm(bstar)
    cstar_len = np.linalg.norm(cstar)

    conv = 0.6947

    Q_min = []
    Q_max = []
    hw = []

    if mode == "Ef fixed":

        Ef = energy_input

        # Ei rangeを連続化
        Ei_list = np.arange(
            Ef + 0.01,
            np.max(data[:,1]),
            0.2
        )

        for Ei in Ei_list:

            # S2limitを補間
            S2limit = float(
                S2interp(Ei)
            )

            ki = conv * np.sqrt(Ei)
            kf = conv * np.sqrt(Ef)

            # energy transfer
            w = Ei - Ef

            # scattering angle
            theta_min = np.deg2rad(S2min)
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

    else:

        # Ei fixed:
        # Eiを固定してhwをscan
        Ei = energy_input

        # S2 limitはEiから決定
        S2limit = float(S2interp(Ei))

        hw_list = np.arange(
            0,
            Ei-0.01,
            0.2
        )


        for w in hw_list:

            Ef = Ei - w

            if Ef <= 0:
                continue

            ki = conv * np.sqrt(Ei)
            kf = conv * np.sqrt(Ef)

            theta_min = np.deg2rad(S2min)
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

    fig = go.Figure()

    for n in range(1,10):

        q=n*astar_len

        if q>np.max(Q_max):
            break

        fig.add_vline(
            x=q,
            line_dash="dot",
            line_color="red",
            opacity=0.5
        )

        fig.add_annotation(
            x=q,
            y=max(hw),
            text=f"{n}a*",
            showarrow=False,
            xshift=15,
            yshift=30,
            font=dict(color="red")
        )

    for n in range(1,10):

        q=n*bstar_len

        if q>np.max(Q_max):
            break

        fig.add_vline(
            x=q,
            line_dash="dot",
            line_color="blue",
            opacity=0.5
        )

        fig.add_annotation(
            x=q,
            y=max(hw),
            text=f"{n}b*",
            showarrow=False,
            xshift=15,
            yshift=30,
            font=dict(color="blue")
        )

    for n in range(1,10):

        q=n*cstar_len

        if q>np.max(Q_max):
            break

        fig.add_vline(
            x=q,
            line_dash="dot",
            line_color="green",
            opacity=0.5
        )

        fig.add_annotation(
            x=q,
            y=max(hw),
            text=f"{n}c*",
            showarrow=False,
            xshift=15,
            yshift=30,
            font=dict(color="green")
        )

    if show_k:

        kvec = np.array([
            k_h,
            k_k,
            k_l
        ])

        mag_Q = []

        # Q範囲
        Qlimit = np.max(Q_max)

        # h,k,l の探索範囲
        hmax = 10
        kmax = 10
        lmax = 10

        for h in range(-hmax, hmax+1):
            for k in range(-kmax, kmax+1):
                for l in range(-lmax, lmax+1):

                    G = (
                        h*astar +
                        k*bstar +
                        l*cstar
                    )

                    # +k と -k
                    for sign in (+1, -1):

                        Qmag = G + sign * (
                            kvec[0]*astar +
                            kvec[1]*bstar +
                            kvec[2]*cstar
                        )

                        q = np.linalg.norm(Qmag)

                        if q < 1e-6:
                            continue

                        if q <= Qlimit:
                            mag_Q.append(q)


        # 重複除去
        mag_Q = sorted(list(set(np.round(mag_Q,6))))

        for q in mag_Q:

            fig.add_vline(
                x=q,
                line_dash="dot",
                line_color="black",
                opacity=0.6
            )

            fig.add_annotation(
                x=q,
                y=max(hw),
                text="k*",
                showarrow=False,
                xshift=15,
                yshift=15,
                font=dict(
                    color="black"
                )
            )

    # ==============================
    # Auto axis range
    # ==============================

    Qmax_plot = np.max(Q_max)

    Q_margin = 0.1 * Qmax_plot

    xmin = 0
    xmax = Qmax_plot + Q_margin


    hw_min = np.min(hw)
    hw_max = np.max(hw)

    hw_margin = 0.1 * (hw_max - hw_min)

    ymin = hw_min - hw_margin
    ymax = hw_max + hw_margin

    # Accessible region
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([Q_min, Q_max[::-1]]),
            y=np.concatenate([hw, hw[::-1]]),
            fill="toself",
            fillcolor="rgba(255,150,150,0.35)",
            line=dict(width=0),
            name="Accessible QE range"
        )
    )

    fig.update_layout(
        
        title=dict(
            text=(
                f"Ef={Ef:.1f} meV | "
                f"a={a:.3f}, b={b:.3f}, c={c:.3f} Å<br>"
                f"α={alpha:.1f}, β={beta:.1f}, γ={gamma:.1f}°"
            ),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=14)
        ),

        xaxis_title="Q (Å⁻¹)",
        yaxis_title="ħω (meV)",

        width=800,
        height=650,

        xaxis=dict(
            range=[0, xmax],
            showgrid=True,
            gridcolor="lightgray",
            zeroline=False,

            linecolor="black",
            tickfont=dict(color="black"),
            mirror=True
        ),

        yaxis=dict(
            range=[0, ymax],
            showgrid=True,
            gridcolor="lightgray",
            zeroline=False,

            linecolor="black",
            tickfont=dict(color="black"),
            mirror=True
        ),

        plot_bgcolor="white",
        paper_bgcolor="white",

        legend=dict(
            x=0.02,
            y=0.98
        )
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    