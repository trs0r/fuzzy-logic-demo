import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Fuzzy Logic Tipping", layout="wide")
st.title("Fuzzy Logic: Trinkgeld")

st.markdown("""
### Wie funktioniert diese Trinkgeldlogik?
Dieses Dashboard demonstriert den klassischen **Mamdani-Inferenz-Prozess**. Anstatt harte Grenzen zu ziehen (z.B. "Wenn Service > 5, dann gutes Trinkgeld"), nutzt die Fuzzy-Logik fließende Übergänge (Zugehörigkeitsfunktionen). 

Spiele mit den Slidern an der Seite und beobachte, wie das System in drei Schritten das Trinkgeld berechnet:

1. **Fuzzifizierung (Inputs):** Deine Bewertungen für Essen und Service (scharfe Werte von 0-10) werden in unscharfe Mengen übersetzt. Eine Bewertung von 6.5 ist beispielsweise nicht einfach nur "gut", sondern vielleicht zu 70% "gut" und zu 30% "exzellent".
2. **Inferenz (Regelauswertung):** Das System wendet drei "Wenn-Dann"-Regeln an. 
   * Bei einer **OR-Verknüpfung** nimmt das System das *Maximum*, bei einer **AND-Verknüpfung** das *Minimum* der beiden Zugehörigkeiten. 
   * Dieser Wert "schneidet" dann das resultierende Trinkgeld-Dreieck ab (die farbig gefüllten Flächen).
3. **Defuzzifizierung (Schwerpunkt):** Alle abgeschnittenen Dreiecke werden übereinandergelegt (graue Fläche). Da der Kellner am Ende echtes Geld sehen will, berechnen wir den geometrischen Schwerpunkt (Centroid) dieser Fläche (die blaue Linie). Das ist das finale Trinkgeld!

**Tipp:** Drehe den 3D-Graphen auf der rechten Seite! Er zeigt dir den kompletten Lösungsraum, also jedes erdenkliche Trinkgeld für jede mögliche Kombination aus Essen und Service.
""")
st.divider()

x_essen = np.arange(0, 11, 0.1)
x_service = np.arange(0, 11, 0.1)
x_tip = np.arange(0, 15.1, 0.1) 

essen_lo = fuzz.trimf(x_essen, [0, 0, 5]); essen_md = fuzz.trimf(x_essen, [2, 6, 10]); essen_hi = fuzz.trimf(x_essen, [6, 10, 10])
serv_lo = fuzz.trimf(x_service, [0, 0, 5]); serv_md = fuzz.trimf(x_service, [2, 6, 10]); serv_hi = fuzz.trimf(x_service, [6, 10, 10])

tip_lo = fuzz.trimf(x_tip, [0, 1, 3]); tip_md = fuzz.trimf(x_tip, [3, 6, 9]); tip_hi = fuzz.trimf(x_tip, [10, 15, 15])

essen = ctrl.Antecedent(x_essen, 'essen')
service = ctrl.Antecedent(x_service, 'service')
tip = ctrl.Consequent(x_tip, 'tip')

essen['schlecht'] = essen_lo; essen['gut'] = essen_md; essen['exzellent'] = essen_hi
service['schlecht'] = serv_lo; service['gut'] = serv_md; service['exzellent'] = serv_hi
tip['wenig'] = tip_lo; tip['mittel'] = tip_md; tip['hoch'] = tip_hi

rule1 = ctrl.Rule(essen['schlecht'] | service['schlecht'], tip['wenig'])
rule2 = ctrl.Rule(essen['gut'] & service['gut'], tip['mittel'])
rule3 = ctrl.Rule(essen['exzellent'] & service['exzellent'], tip['hoch'])
tipping_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem([rule1, rule2, rule3]))

@st.cache_data
def get_3d_surface(version=2):
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

x_3d, y_3d, z_3d = get_3d_surface(version=2)

st.sidebar.header("1. Bitte bewerten:")
e = st.sidebar.slider("Bewertung Essen (0-10)", 0.0, 10.0, 6.5, 0.1)
s = st.sidebar.slider("Bewertung Service (0-10)", 0.0, 10.0, 7.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("2. Rechnung:")
bill = st.sidebar.number_input("Rechnungsbetrag in €", min_value=0.0, value=50.0, step=1.0)

mu_e_lo = fuzz.interp_membership(x_essen, essen_lo, e); mu_e_md = fuzz.interp_membership(x_essen, essen_md, e); mu_e_hi = fuzz.interp_membership(x_essen, essen_hi, e)
mu_s_lo = fuzz.interp_membership(x_service, serv_lo, s); mu_s_md = fuzz.interp_membership(x_service, serv_md, s); mu_s_hi = fuzz.interp_membership(x_service, serv_hi, s)

act1 = np.fmax(mu_e_lo, mu_s_lo); out1 = np.fmin(act1, tip_lo) # OR = np.fmax
act2 = np.fmin(mu_e_md, mu_s_md); out2 = np.fmin(act2, tip_md) # AND = np.fmin
act3 = np.fmin(mu_e_hi, mu_s_hi); out3 = np.fmin(act3, tip_hi) # AND = np.fmin

agg = np.fmax(out1, np.fmax(out2, out3))
try: res = fuzz.defuzz(x_tip, agg, 'centroid')
except: res = 0

tip_euro = bill * (res / 100)

st.sidebar.markdown("---")
st.sidebar.header("Dein Ergebnis:")
st.sidebar.success(f"**Trinkgeld: {tip_euro:.2f} €** \n\n*(entspricht {res:.2f} %)*")
st.sidebar.info(f"**Gesamtsumme: {bill + tip_euro:.2f} €**")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Inferenz-Logik")
    fig, axes = plt.subplots(4, 1, figsize=(8, 10))
    plt.subplots_adjust(hspace=0.6)
    
    # Regel 1
    axes[0].plot(x_tip, tip_lo, 'k:', alpha=0.3); axes[0].fill_between(x_tip, 0, out1, color='red', alpha=0.6)
    axes[0].set_title(f"R1: Essen schlecht OR Service schlecht (Akt: {act1*100:.0f}%)", fontweight='bold'); axes[0].set_ylim(0, 1.1); axes[0].set_yticks([])
    
    # Regel 2
    axes[1].plot(x_tip, tip_md, 'k:', alpha=0.3); axes[1].fill_between(x_tip, 0, out2, color='orange', alpha=0.6)
    axes[1].set_title(f"R2: Essen gut AND Service gut (Akt: {act2*100:.0f}%)", fontweight='bold'); axes[1].set_ylim(0, 1.1); axes[1].set_yticks([])
    
    # Regel 3
    axes[2].plot(x_tip, tip_hi, 'k:', alpha=0.3); axes[2].fill_between(x_tip, 0, out3, color='green', alpha=0.6)
    axes[2].set_title(f"R3: Essen exzellent AND Service exzellent (Akt: {act3*100:.0f}%)", fontweight='bold'); axes[2].set_ylim(0, 1.1); axes[2].set_yticks([])
    
    # Ergebnis
    axes[3].plot(x_tip, tip_lo, 'k:', alpha=0.1); axes[3].plot(x_tip, tip_md, 'k:', alpha=0.1); axes[3].plot(x_tip, tip_hi, 'k:', alpha=0.1)
    axes[3].fill_between(x_tip, 0, agg, color='gray', alpha=0.5); axes[3].axvline(x=res, color='blue', linewidth=3)
    axes[3].set_title(f"Ergebnis (Schwerpunkt): {res:.2f}%", fontweight='bold'); axes[3].set_ylim(0, 1.1); axes[3].set_yticks([])
    
    st.pyplot(fig)

with col2:
    st.subheader(f"Trinkgeld Landschaft")
    fig_3d = go.Figure(data=[go.Surface(z=z_3d, x=x_3d, y=y_3d, colorscale='Viridis', opacity=0.8)])
    fig_3d.add_trace(go.Scatter3d(x=[e], y=[s], z=[res], mode='markers', marker=dict(color='red', size=8, line=dict(color='white', width=2))))
    fig_3d.update_layout(scene=dict(xaxis_title='Essen', yaxis_title='Service', zaxis_title='Trinkgeld (%)'), height=700, margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig_3d, use_container_width=True)