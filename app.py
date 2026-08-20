import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import json
import os

# デバックの手順(powershell上で動かす。)
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

    /* --------------------------------------------------------
       Calculation Point sticky area
       -------------------------------------------------------- */

    .calculation-sticky {
        position: sticky;
        top: 0rem;
        z-index: 999;
        background-color: var(--background-color);
        padding-top: 0.5rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(128,128,128,0.35);
    }


    /* Sidebar heading spacing */
    section[data-testid="stSidebar"] h3 {
        margin-top: 0.8rem;
    }


    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(
    page_title="TAS QErange simulator",
    layout="wide"
)

st.title("TAS Q-E Range Simulator")

display_names = {
    "CTAX": "CTAX@HFIR",
    "HB1": "HB-1@HFIR",
    "HB3": "HB-3@HFIR",
    "HODACA": "HODACA/HER@JRR3",
    "PoplarL": "Poplar(Larmor)@HFIR",
    "PoplarS": "Poplar(Standard)@HFIR"
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

if len(data) == 1:

    S2_fixed = data[0,0]

    def S2interp(Ei):
        return S2_fixed

else:

    idx = np.argsort(data[:,1])

    S2interp = interp1d(
        data[idx,1],
        data[idx,0],
        kind="linear",
        fill_value="extrapolate"
    )

# λ/2用
idx_half = np.argsort(data[:,1]*4)

S2interp_half = interp1d(
    data[idx_half,1]*4,
    data[idx_half,0],
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
                value=5.00,
                step=0.01
            )

        with col3:
            c = st.number_input(
                "c (Å)",
                value=5.00,
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
            "S2_min",
            8.0
        )

        default_sense = instrument_data.get("sense", "-+-")

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
                #data[:,1] = 4 * data[:,1]

            else:
                Ef = energy_input


        else:

            if lambda_half:
                Ei = 4 * energy_input

                #data[:,1] = 4 * data[:,1]

            else:
                Ei = energy_input


        S2min = st.number_input(
            "minimum 2θ (deg)",
            value=float(default_S2min),
            step=0.1,
            key=f"S2min_{instrument}"
        )

        st.header("Dark angle")

        # ============================================================
        # Reference & sense
        # ============================================================

        col1, col2 = st.columns([1, 1])

        with col1:
            dark_angle_reference = st.radio(
                "Reference",
                ["Reference Q", "Direct beam"],
                horizontal=True,
                index=0,
                key="dark_angle_reference"
            )

        with col2:
            sense_options = ["+-+", "-+-"]
            
            default_sense_index = (
                sense_options.index(default_sense)
                if default_sense in sense_options
                else 0
            )

            instrument_sense = st.radio(
                "Sense",
                sense_options,
                horizontal=True,
                index=default_sense_index,
                key=f"instrument_sense_{instrument}"
            )

        # ============================================================
        # Reference Q
        # ============================================================

        st.markdown("**Reference Q**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            ref_h = st.number_input(
                "h",
                value=0.0,
                step=0.1,
                key="ref_h"
            )

        with col2:
            ref_k = st.number_input(
                "k",
                value=0.0,
                step=0.1,
                key="ref_k"
            )

        with col3:
            ref_l = st.number_input(
                "l",
                value=0.0,
                step=0.1,
                key="ref_l"
            )

        with col4:
            ref_s1 = st.number_input(
                "s1 (deg)",
                value=0.0,
                step=0.1,
                key="ref_s1"
            )

        # ============================================================
        # Dark angle range
        # ============================================================

        st.markdown("**Range**")

        col0, col1, col2, col3 = st.columns([0.5, 1, 1, 1])

        with col0:
            st.markdown("**No.**")

        with col1:
            st.markdown("**From (deg)**")

        with col2:
            st.markdown("**To (deg)**")

        with col3:
            st.markdown("**Offset (deg)**")

        dark_angle_ranges = []

        for i in range(4):

            col0, col1, col2, col3 = st.columns([0.5, 1, 1, 1])

            with col0:
                st.markdown(f"**{i + 1}**")

            with col1:
                angle_from = st.number_input(
                    "From (deg)",
                    value=0.0,
                    step=1.0,
                    key=f"dark_from_{i}",
                    label_visibility="collapsed"
                )

            with col2:
                angle_to = st.number_input(
                    "To (deg)",
                    value=0.0,
                    step=1.0,
                    key=f"dark_to_{i}",
                    label_visibility="collapsed"
                )

            with col3:
                offset = st.number_input(
                    "Offset (deg)",
                    value=0.0,
                    step=1.0,
                    key=f"dark_offset_{i}",
                    label_visibility="collapsed"
                )

            dark_angle_ranges.append(
                (angle_from, angle_to, offset)
            )

        # --------------------------------------------------------
        # Add dark angle
        # --------------------------------------------------------

        add_dark_angle = st.button("Add dark angle")

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

    def calc_Q_region(Ei, Ef, hw, S2min, S2interp_use):

        S2max = float(S2interp_use(Ei))

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
    if lambda_half:

        hw_list = np.array([0.0])

    else:

        if mode=='Ef fixed':

            Ei_max = np.max(data[:,1])

            hw_list=np.arange(
                0,
                Ei_max-Ef,
                0.2
            )

        else:

            hw_list=np.arange(
                0,
                Ei,
                0.2
            )

    regions=[]
    S2_list=[]
    Qmax_list=[]

    for hw in hw_list:
        if mode=='Ef fixed':
            Ei = Ef + hw
        else:
            Ef = Ei - hw

        if lambda_half:
            S2interp_use = S2interp_half
        else:
            S2interp_use = S2interp

        result = calc_Q_region(
            Ei,
            Ef,
            hw,
            S2min,
            S2interp_use
        )
        
        regions.append(result[:4])
        S2_list.append(result[4])

        S2max = float(S2interp_use(Ei))

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

    if add_dark_angle:

        # ============================================================
        # Dark angle
        # ============================================================

        dark_regions_kf = []
        dark_regions_ki = []

        # ----------------------------------------
        # Calculate offset
        # ----------------------------------------

        # Reciprocal lattice vector of reference Bragg position
        Q_ref = (
            ref_h * astar
            + ref_k * bstar
            + ref_l * cstar
        )

        d_ref = 2.0 * np.pi / np.linalg.norm(Q_ref)

        # ----------------------------------------
        # Q_ref angle in the (u, v) coordinate
        # ----------------------------------------

        Q_ref_x = np.dot(Q_ref, ex)
        Q_ref_y = np.dot(Q_ref, ey)

        phi_ref = np.degrees(
            np.arctan2(Q_ref_y, Q_ref_x)
        )

        # ----------------------------------------
        # Neutron wavelength
        # ----------------------------------------

        if mode == 'Ef fixed':
            wavelength = 9.044 / np.sqrt(Ef)
        else:
            wavelength = 9.044 / np.sqrt(Ei)

        # ----------------------------------------
        # Bragg angle
        # ----------------------------------------

        theta_ref = np.degrees(
            np.arcsin(
                wavelength / (2.0 * d_ref)
            )
        )

        # ----------------------------------------
        # angle between Q and kf, 0 deg means just block kf
        # ----------------------------------------
        if dark_angle_reference == "Reference Q":

            Q_offset = 90 + theta_ref

        else:

            Q_offset = 2.0 * theta_ref

        # ----------------------------------------
        # Crystal angle offset
        # ----------------------------------------

        # s1_offset = (ref_s1 - theta_ref + 180.0 - phi_ref)
        s1_offset = ( - theta_ref + 180.0 - phi_ref)

        # --------------------------------------------------------
        # Q calculation with S1
        # --------------------------------------------------------

        def calc_q(s1, s2, ki, kf, s1_offset):

            ki_angle = np.deg2rad(
                -s1 + s1_offset
            )

            kf_angle = np.deg2rad(
                s2 - s1 + s1_offset
            )

            kix = ki * np.sin(ki_angle)
            kiy = ki * np.cos(ki_angle)

            kfx = kf * np.sin(kf_angle)
            kfy = kf * np.cos(kf_angle)

            qx = kix - kfx
            qy = kiy - kfy

            if instrument_sense == "+-+":

                # Reference Q direction
                eQ = np.array([
                    Q_ref_x,
                    Q_ref_y
                ])

                eQ /= np.linalg.norm(eQ)

                # Reflection with respect to Reference-Q axis
                q = np.array([qx, qy])

                qx, qy = (
                    2.0 * np.dot(q, eQ) * eQ - q
                )

            return qx, qy

        # --------------------------------------------------------
        # Calculate Dark angle for each hw
        # --------------------------------------------------------

        for i, hw in enumerate(hw_list):

            # Energy for this hw
            if mode == 'Ef fixed':
                Ei_hw = Ef + hw
                Ef_hw = Ef
            else:
                Ei_hw = Ei
                Ef_hw = Ei - hw

            # Wave vectors
            ki = 0.6947 * np.sqrt(Ei_hw)
            kf = 0.6947 * np.sqrt(Ef_hw)

            # S2 range for this hw
            S2max_hw = S2_list[i]

            s2_dark = np.linspace(
                S2min,
                S2max_hw,
                200
            )

            # ========================================================
            # Dark angle regions for this hw
            # ========================================================

            dark_regions_hw_kf = []
            dark_regions_hw_ki = []

            # ========================================================
            # Loop over additional s1 ranges
            # ========================================================

            for angle_from, angle_to, offset in dark_angle_ranges:

                # 0, 0 は未使用
                if angle_from == 0 and angle_to == 0:
                    continue

                if instrument_sense == "+-+":
                    offset = -offset
                    angle_from = -angle_from
                    angle_to = -angle_to

                # Actual s1 positions
                s1_from = offset + angle_from - Q_offset
                s1_to   = offset + angle_to - Q_offset

                # ====================================================
                # kf side
                # ====================================================

                # S1 = From
                dark_x_from = []
                dark_y_from = []

                for s2 in s2_dark:

                    qx, qy = calc_q(
                        s1_from,
                        s2,
                        ki,
                        kf,
                        s1_offset
                    )

                    dark_x_from.append(qx)
                    dark_y_from.append(qy)

                # S1 = To
                dark_x_to = []
                dark_y_to = []

                for s2 in s2_dark:

                    qx, qy = calc_q(
                        s1_to,
                        s2,
                        ki,
                        kf,
                        s1_offset
                    )

                    dark_x_to.append(qx)
                    dark_y_to.append(qy)


                dark_x_from = np.array(dark_x_from)
                dark_y_from = np.array(dark_y_from)

                dark_x_to = np.array(dark_x_to)
                dark_y_to = np.array(dark_y_to)

                # ====================================================
                # ki side
                #
                # s1_ki = s1_kf + 180 - S2
                # ====================================================

                dark_x_from_ki = []
                dark_y_from_ki = []

                dark_x_to_ki = []
                dark_y_to_ki = []

                for s2 in s2_dark:

                    # From
                    s1_from_ki = s1_from - (180.0 - s2)

                    qx, qy = calc_q(
                        s1_from_ki,
                        s2,
                        ki,
                        kf,
                        s1_offset
                    )

                    dark_x_from_ki.append(qx)
                    dark_y_from_ki.append(qy)


                    # To
                    s1_to_ki = s1_to - (180.0 - s2)

                    qx, qy = calc_q(
                        s1_to_ki,
                        s2,
                        ki,
                        kf,
                        s1_offset
                    )

                    dark_x_to_ki.append(qx)
                    dark_y_to_ki.append(qy)


                dark_x_from_ki = np.array(dark_x_from_ki)
                dark_y_from_ki = np.array(dark_y_from_ki)

                dark_x_to_ki = np.array(dark_x_to_ki)
                dark_y_to_ki = np.array(dark_y_to_ki)


                # ====================================================
                # kf side closed region
                # ====================================================

                s1_edge_top = np.linspace(
                    s1_from,
                    s1_to,
                    100
                )

                dark_x_top = []
                dark_y_top = []

                for s1 in s1_edge_top:

                    qx, qy = calc_q(
                        s1,
                        S2max_hw,
                        ki,
                        kf,
                        s1_offset
                    )

                    dark_x_top.append(qx)
                    dark_y_top.append(qy)


                s1_edge_bottom = np.linspace(
                    s1_to,
                    s1_from,
                    100
                )

                dark_x_bottom = []
                dark_y_bottom = []

                for s1 in s1_edge_bottom:

                    qx, qy = calc_q(
                        s1,
                        S2min,
                        ki,
                        kf,
                        s1_offset
                    )

                    dark_x_bottom.append(qx)
                    dark_y_bottom.append(qy)


                dark_x_kf = np.concatenate([
                    dark_x_from,
                    dark_x_top,
                    dark_x_to[::-1],
                    dark_x_bottom
                ])

                dark_y_kf = np.concatenate([
                    dark_y_from,
                    dark_y_top,
                    dark_y_to[::-1],
                    dark_y_bottom
                ])


                # ====================================================
                # ki side closed region
                # ====================================================

                # S2 = S2max
                s1_edge_top_ki = np.linspace(
                    s1_from - (180.0 - S2max_hw),
                    s1_to   - (180.0 - S2max_hw),
                    100
                )

                dark_x_top_ki = []
                dark_y_top_ki = []

                for s1 in s1_edge_top_ki:

                    qx, qy = calc_q(
                        s1,
                        S2max_hw,
                        ki,
                        kf,
                        s1_offset
                    )

                    dark_x_top_ki.append(qx)
                    dark_y_top_ki.append(qy)


                # S2 = S2min
                s1_edge_bottom_ki = np.linspace(
                    s1_to   - (180.0 - S2min),
                    s1_from - (180.0 - S2min),
                    100
                )

                dark_x_bottom_ki = []
                dark_y_bottom_ki = []

                for s1 in s1_edge_bottom_ki:

                    qx, qy = calc_q(
                        s1,
                        S2min,
                        ki,
                        kf,
                        s1_offset
                    )

                    dark_x_bottom_ki.append(qx)
                    dark_y_bottom_ki.append(qy)


                dark_x_ki = np.concatenate([
                    dark_x_from_ki,
                    dark_x_top_ki,
                    dark_x_to_ki[::-1],
                    dark_x_bottom_ki
                ])

                dark_y_ki = np.concatenate([
                    dark_y_from_ki,
                    dark_y_top_ki,
                    dark_y_to_ki[::-1],
                    dark_y_bottom_ki
                ])


                dark_regions_hw_kf.append(
                    (
                        dark_x_kf,
                        dark_y_kf
                    )
                )

                dark_regions_hw_ki.append(
                    (
                        dark_x_ki,
                        dark_y_ki
                    )
                )


            # ========================================================
            # Store all ranges for this hw
            # ========================================================

            dark_regions_kf.append(dark_regions_hw_kf)
            dark_regions_ki.append(dark_regions_hw_ki)

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

    if add_dark_angle:

        # ============================================================
        # Dark angle trace
        # ============================================================

        for dark_x0, dark_y0 in dark_regions_kf[0]:

            fig.add_trace(
                go.Scatter(
                    x=dark_x0,
                    y=dark_y0,
                    fill="toself",
                    name="Dark angle (kf side)",
                    line=dict(width=0),
                    fillcolor="rgba(0,0,255,0.15)"
                )
            )

        for dark_x0, dark_y0 in dark_regions_ki[0]:
        
            fig.add_trace(
                go.Scatter(
                    x=dark_x0,
                    y=dark_y0,
                    fill="toself",
                    name="Dark angle (ki side)",
                    line=dict(width=0),
                    fillcolor="rgba(0,255,0,0.15)"
                )
            )


    # ============================================================
    # Slider
    # ============================================================

    steps = []

    for i, hw in enumerate(hw_list):

        xmin, ymin, xmax, ymax = regions[i]

        x_data = [
            np.concatenate([
                xmax,
                xmin[::-1]
            ]),
            Gx_points,
            mag_x,
            [None]
        ]

        y_data = [
            np.concatenate([
                ymax,
                ymin[::-1]
            ]),
            Gy_points,
            mag_y,
            [None]
        ]

        name_data = [
            "Accessible Q",
            "Nuclear Bragg peaks",
            "Magnetic Bragg peaks",
            f"S2 range = {S2min:.1f} - {S2_list[i]:.1f}°"
        ]

        if add_dark_angle:
            
            # --------------------------------------------------------
            # Dark angle
            # --------------------------------------------------------

            for dark_x, dark_y in dark_regions_kf[i]:

                x_data.append(dark_x)
                y_data.append(dark_y)

                name_data.append("Dark angle (kf side)")

            for dark_x, dark_y in dark_regions_ki[i]:
            
                x_data.append(dark_x)
                y_data.append(dark_y)

                name_data.append("Dark angle (ki side)")

        step = dict(
            method="update",
            args=[
                {
                    "x": x_data,
                    "y": y_data,
                    "name": name_data
                }
            ],
            label=f"{hw:.1f} meV"
        )

        steps.append(step)

    if mode == "Ef fixed":
        energy_text = f"Ef={Ef:.2f} meV"
    else:
        energy_text = f"Ei={Ei:.2f} meV"

    lambda_text = " | λ/2" if lambda_half else ""

    fig.update_layout(
        title=dict(
            text=(
                f"{instrument_data['name']} | "
                f"{energy_text}{lambda_text}<br>"
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

    col_main, col_geometry = st.columns([3, 1])

    with col_main:
        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col_geometry:

        # ============================================================
        # Top view of Dark angle geometry
        # ============================================================

        fig_geometry = go.Figure()

        # ------------------------------------------------------------
        # Q direction
        #
        # Q starts from (0, -1) and points to (0, 1)
        # ------------------------------------------------------------

        q_start_x = 0.0
        q_start_y = -0.2

        q_end_x = 0.0
        q_end_y = 2.0

        fig_geometry.add_annotation(
            x=q_end_x,
            y=q_end_y,
            ax=q_start_x,
            ay=q_start_y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor="red"
        )

        # Q label
        fig_geometry.add_annotation(
            x=0.8,
            y=0.8,
            text="Reference Q" if dark_angle_reference == "Reference Q" else "ki",
            showarrow=False,
            font=dict(
                size=16,
                color="red"
            )
        )

        # ------------------------------------------------------------
        # Dark angle circle
        #
        # Center = starting point of Q
        # Radius  = same length as Q arrow
        # ------------------------------------------------------------

        center_x = 0.0
        center_y = 0.0

        dark_radius = 2.0

        circle_theta = np.linspace(
            0,
            2 * np.pi,
            360
        )

        circle_x = (
            center_x
            + dark_radius * np.cos(circle_theta)
        )

        circle_y = (
            center_y
            + dark_radius * np.sin(circle_theta)
        )

        fig_geometry.add_trace(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode="lines",
                line=dict(
                    color="gray",
                    width=1
                ),
                showlegend=False,
                hoverinfo="skip"
            )
        )

        # ------------------------------------------------------------
        # Dark angle arcs
        # ------------------------------------------------------------

        for i, (angle_from, angle_to, offset) in enumerate(
            dark_angle_ranges
        ):

            # Ignore unused range
            if angle_from == 0 and angle_to == 0:
                continue

            # --------------------------------------------------------
            # Flip offset for instrument sense
            # --------------------------------------------------------

            #if instrument_sense == "+-+":
            #    offset = -offset

            # --------------------------------------------------------
            # Angle definition
            #
            # 0 deg  = Q direction
            # + angle = counter-clockwise
            # - angle = clockwise
            # --------------------------------------------------------

            angle_start = offset + angle_from
            angle_end = offset + angle_to

            if angle_end < angle_start:
                angle_end += 360.0

            angles = np.linspace(
                angle_start,
                angle_end,
                200
            )

            theta = np.deg2rad(angles)

            # 0 deg = upward
            # Positive = counter-clockwise
            arc_x = (
                center_x
                - dark_radius * np.sin(theta)
            )

            arc_y = (
                center_y
                + dark_radius * np.cos(theta)
            )

            fig_geometry.add_trace(
                go.Scatter(
                    x=arc_x,
                    y=arc_y,
                    mode="lines",
                    line=dict(
                        color="black",
                        width=6
                    ),
                    name=f"Range {i + 1}",
                    showlegend=False,
                    hoverinfo="skip"
                )
            )

        # ------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------

        fig_geometry.update_layout(

            title=dict(
                text="Dark angle (elastic, top view)",
                x=0.5,
                xanchor="center"
            ),

            xaxis=dict(
                range=[-2.3, 2.3],
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),

            yaxis=dict(
                range=[-2.3, 2.3],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="x",
                scaleratio=1
            ),

            width=350,
            height=700,

            margin=dict(
                l=10,
                r=10,
                t=50,
                b=10
            ),

            showlegend=False
        )

        st.plotly_chart(
            fig_geometry,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
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
            0.1
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
            0.1
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

    for n in range(1,20):

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

    for n in range(1,20):

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

    for n in range(1,20):

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
        hmax = 20
        kmax = 20
        lmax = 20

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
                f"{instrument_data['name']} | "
                f"Ef={Ef:.2f} meV | "
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

    