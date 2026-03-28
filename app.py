import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- SEITEN-SETUP ---
st.set_page_config(page_title="Fuzzy Logic Tipping", layout="wide")
st.title("Fuzzy Logic: Trinkgeld-Expertensystem")

# --- 1. LOGIK DEFINIEREN (Einmalig) ---
x_essen = np.arange(0, 11, 0.1)
x_service = np.arange(0, 11, 0.1)
x_tip = np.arange(0, 26, 0.1)

essen_lo = fuzz.trimf(x_essen, [0, 0, 5]); essen_md = fuzz.trimf(x_essen, [0, 5, 10]); essen_hi = fuzz.trimf(x_essen, [5, 10, 10])
serv_lo = fuzz.trimf(x_service, [0, 0, 5]); serv_md = fuzz.trimf(x_service, [0, 5, 10]); serv_hi = fuzz.trimf(x_service, [5, 10, 10])
tip_lo = fuzz.trimf(x_tip, [0, 0, 13]); tip_md = fuzz.trimf(x_tip, [0, 13, 25]); tip_hi = fuzz.trimf(x_tip, [13, 25, 25])

essen = ctrl.Antecedent(x_essen, 'essen')
service = ctrl.Antecedent(x_service, 'service')
tip = ctrl.Consequent(x_tip, 'tip')

essen['schlecht'] = essen_lo; essen['gut'] = essen_md; essen['exzellent'] = essen_hi
service['mies'] = serv_lo; service['ok'] = serv_md; service['top'] = serv_hi
tip['wenig'] = tip_lo; tip['mittel'] = tip_md; tip['hoch'] = tip_hi

rule1 = ctrl.Rule(essen['schlecht'] | service['mies'], tip['wenig'])
rule2 = ctrl.Rule(service['ok'], tip['mittel'])
rule3 = ctrl.Rule(service['top'] | essen['exzellent'], tip['hoch'])
tipping_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem([rule1, rule2, rule3]))

# 3D-Fläche berechnen (Mit st.cache_data, damit es nur 1x rechnet = superschnell!)
@st.cache_data
def get_3d_surface():
    upsampled = np.linspace(0, 10, 25)
    x_3d, y_3d = np.meshgrid(upsampled, upsampled)
    z_3d = np.zeros_like(x_3d)
    for i in range(25):
        for j in range(25):
            tipping_sim.input['essen'] = x_3d[i, j]
            tipping_sim.input['service'] = y_3d[i, j]
            try: tipping_sim.compute(); z_3d[i, j] = tipping_sim.output['tip']
            except: z_3d[i, j] = 0
    return upsampled, upsampled, z_3d

x_3d, y_3d, z_3d = get_3d_surface()

# --- 2. USER INTERFACE (SLIDER) ---
st.sidebar.header("Bitte bewerten:")
e = st.sidebar.slider("Bewertung Essen (0-10)", 0.0, 10.0, 6.5, 0.1)
s = st.sidebar.slider("Bewertung Service (0-10)", 0.0, 10.0, 7.5, 0.1)

# --- 3. BERECHNUNG DES AKTUELLEN WERTES ---
mu_e_lo = fuzz.interp_membership(x_essen, essen_lo, e); mu_e_hi = fuzz.interp_membership(x_essen, essen_hi, e)
mu_s_lo = fuzz.interp_membership(x_service, serv_lo, s); mu_s_md = fuzz.interp_membership(x_service, serv_md, s); mu_s_hi = fuzz.interp_membership(x_service, serv_hi, s)

act1 = np.fmax(mu_e_lo, mu_s_lo); out1 = np.fmin(act1, tip_lo)
act2 = mu_s_md; out2 = np.fmin(act2, tip_md)
act3 = np.fmax(mu_e_hi, mu_s_hi); out3 = np.fmin(act3, tip_hi)

agg = np.fmax(out1, np.fmax(out2, out3))
try: res = fuzz.defuzz(x_tip, agg, 'centroid')
except: res = 0

# --- 4. VISUALISIERUNG (2 Spalten Layout) ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Inferenz-Logik")
    fig, axes = plt.subplots(4, 1, figsize=(8, 10))
    plt.subplots_adjust(hspace=0.6)
    
    # Regel 1
    axes[0].plot(x_tip, tip_lo, 'k:', alpha=0.3); axes[0].fill_between(x_tip, 0, out1, color='red', alpha=0.6)
    axes[0].set_title(f"R1: Essen schlecht OR Service mies (Akt: {act1*100:.0f}%)", fontweight='bold'); axes[0].set_ylim(0, 1.1); axes[0].set_yticks([])
    # Regel 2
    axes[1].plot(x_tip, tip_md, 'k:', alpha=0.3); axes[1].fill_between(x_tip, 0, out2, color='orange', alpha=0.6)
    axes[1].set_title(f"R2: Service OK (Akt: {act2*100:.0f}%)", fontweight='bold'); axes[1].set_ylim(0, 1.1); axes[1].set_yticks([])
    # Regel 3
    axes[2].plot(x_tip, tip_hi, 'k:', alpha=0.3); axes[2].fill_between(x_tip, 0, out3, color='green', alpha=0.6)
    axes[2].set_title(f"R3: Essen exzellent OR Service Top (Akt: {act3*100:.0f}%)", fontweight='bold'); axes[2].set_ylim(0, 1.1); axes[2].set_yticks([])
    # Ergebnis
    axes[3].plot(x_tip, tip_lo, 'k:', alpha=0.1); axes[3].plot(x_tip, tip_md, 'k:', alpha=0.1); axes[3].plot(x_tip, tip_hi, 'k:', alpha=0.1)
    axes[3].fill_between(x_tip, 0, agg, color='gray', alpha=0.5); axes[3].axvline(x=res, color='blue', linewidth=3)
    axes[3].set_title(f"Ergebnis (Schwerpunkt): {res:.2f}%", fontweight='bold'); axes[3].set_ylim(0, 1.1); axes[3].set_yticks([])
    
    st.pyplot(fig)

with col2:
    st.subheader(f"3D Control Surface")
    fig_3d = go.Figure(data=[go.Surface(z=z_3d, x=x_3d, y=y_3d, colorscale='Viridis', opacity=0.8)])
    fig_3d.add_trace(go.Scatter3d(x=[e], y=[s], z=[res], mode='markers', marker=dict(color='red', size=8, line=dict(color='white', width=2))))
    fig_3d.update_layout(scene=dict(xaxis_title='Essen', yaxis_title='Service', zaxis_title='Trinkgeld (%)'), height=700, margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig_3d, use_container_width=True)