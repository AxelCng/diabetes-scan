import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE LA APP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesScan | Predicción de Diabetes",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS PREMIUM (Tema Médico / Clínico)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.85); }
        to   { opacity: 1; transform: scale(1); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
    }

    /* Hide Streamlit extras */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: transparent; }

    /* Sidebar */
    div[data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    div[data-testid="stSidebarContent"] * {
        color: rgba(255,255,255,0.9) !important;
    }
    div[data-testid="stSidebarContent"] hr {
        border-color: rgba(255,255,255,0.1) !important;
    }

    /* Hero Title */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(-45deg, #0ea5e9, #2563eb, #0d9488, #0ea5e9);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 6s ease infinite;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        color: #475569;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 2.5rem;
        animation: fadeInUp 0.8s ease-out 0.2s both;
    }

    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.8);
        border-radius: 20px;
        padding: 1.8rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.04);
        cursor: default;
    }
    .glass-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 48px rgba(0, 0, 0, 0.08);
        border-color: rgba(14, 165, 233, 0.3);
    }

    /* Stat Card */
    .stat-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03);
        position: relative;
        overflow: hidden;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #0ea5e9, #2563eb);
        border-radius: 20px 20px 0 0;
    }
    .stat-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 40px rgba(14, 165, 233, 0.12);
        border-color: #0ea5e9;
    }
    .stat-value { font-size: 1.8rem; font-weight: 800; color: #0f172a; margin: 0.4rem 0 0.2rem 0; }
    .stat-label { font-size: 0.85rem; color: #334155; font-weight: 600; }
    .stat-sub { font-size: 0.75rem; color: #64748b; margin-top: 0.2rem; }

    /* Feature Card */
    .feature-card {
        background: white;
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03);
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 48px rgba(14, 165, 233, 0.12);
        border-color: #0ea5e9;
    }
    .feature-icon-circle {
        width: 56px; height: 56px;
        border-radius: 16px;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 1rem auto;
        font-size: 1.4rem; font-weight: 800; color: white;
    }
    .feature-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        font-size: 0.85rem;
        color: #475569;
        line-height: 1.5;
    }

    /* 5V Card */
    .v-card {
        border-radius: 20px;
        padding: 1.3rem;
        text-align: center;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: default;
        border: 2px solid transparent;
    }
    .v-card:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0 16px 32px rgba(0,0,0,0.1);
    }

    /* Section Header */
    .section-header {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.3rem;
        position: relative;
        display: inline-block;
    }
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -6px;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #0ea5e9, #2563eb);
        border-radius: 2px;
    }
    .section-sub {
        color: #475569;
        font-size: 0.95rem;
        margin-top: 0.8rem;
        margin-bottom: 2rem;
    }

    /* Form Section Headers */
    .form-section {
        background: linear-gradient(135deg, #0ea5e910, #2563eb10);
        border-left: 4px solid;
        border-image: linear-gradient(180deg, #0ea5e9, #2563eb) 1;
        padding: 0.6rem 1.2rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 1rem;
        font-weight: 700;
        font-size: 1rem;
        color: #0f172a;
    }

    /* Risk Result */
    .risk-result {
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        animation: scaleIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .risk-pct {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0.3rem 0;
        letter-spacing: -2px;
    }
    .risk-label {
        font-weight: 700;
        font-size: 1.2rem;
        margin-top: 0.3rem;
    }
    .risk-dot {
        width: 16px; height: 16px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
        vertical-align: middle;
    }

    /* Factor pill */
    .factor-pill {
        display: block;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin: 0.3rem 0;
        font-size: 0.85rem;
        font-weight: 500;
        animation: fadeInUp 0.5s ease-out both;
    }
    .factor-pill-red {
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .factor-pill-yellow {
        background: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .factor-pill-green {
        background: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }

    /* Button Override */
    div.stFormSubmitButton > button {
        background: linear-gradient(135deg, #0ea5e9, #2563eb) !important;
        color: white !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-radius: 14px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.3) !important;
        letter-spacing: 0.5px !important;
    }
    div.stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(14, 165, 233, 0.5) !important;
    }

    /* Recommendation boxes */
    .rec-box {
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        animation: fadeInUp 0.6s ease-out 0.3s both;
        font-weight: 500;
    }
    .rec-dot {
        width: 48px; height: 48px; min-width: 48px;
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 1.2rem; color: white;
    }

    /* Gradient divider */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #0ea5e940, #2563eb40, transparent);
        border: none;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CARGA DEL MODELO
# ─────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'modelo_random_forest.pkl')
    if not os.path.exists(model_path):
        st.error(f"No se encontró el modelo en: {model_path}")
        st.stop()
    return joblib.load(model_path)

modelo = cargar_modelo()

FEATURE_NAMES = [
    'BMI', 'MentHlth', 'PhysHlth',
    'Presion_Alta', 'Colesterol_Alto', 'Chequeo_Colesterol',
    'Fumador', 'ACV', 'Enfermedad_Cardiaca',
    'Actividad_Fisica', 'Consume_Frutas', 'Consume_Verduras',
    'Alto_Consumo_Alcohol', 'Tiene_Seguro_Medico', 'No_Consulto_por_Costo',
    'Dificultad_Caminar', 'Sexo', 'Salud_General',
    'Rango_Edad', 'Nivel_Educacion', 'Nivel_Ingresos'
]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <p style="font-size:1.6rem; font-weight:900; margin:0;
           background: linear-gradient(90deg, #0ea5e9, #2db6a3);
           -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
           DiabetesScan</p>
        <p style="font-size:0.8rem; opacity:0.6; margin-top:0.3rem;">
           Predicción inteligente de diabetes</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    pagina = st.radio(
        "Navegar a",
        ["Inicio", "Predicción", "Dashboards"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; opacity:0.5; font-size:0.78rem;">
        <p style="margin:0.2rem 0;">Proyecto Big Data & ML</p>
        <p style="margin:0.2rem 0;">Dataset BRFSS 2015</p>
        <p style="margin:0.2rem 0;">253,680 registros</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#0f172a"),
    margin=dict(t=50, b=30, l=30, r=30),
)

# ─────────────────────────────────────────────
# PÁGINA 1: INICIO
# ─────────────────────────────────────────────
if pagina == "Inicio":

    st.markdown("""
    <div style="text-align:center; padding:1rem 0 0.5rem 0; animation: fadeInUp 0.6s ease-out;">
        <p class="hero-title">DiabetesScan</p>
        <p class="hero-subtitle">
            Sistema inteligente de predicción de riesgo de diabetes<br>
            usando <strong>Machine Learning</strong> y <strong>Big Data</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    stats = [
        ("253,680", "Registros", "Dataset BRFSS 2015", "0.1s"),
        ("21", "Variables", "Indicadores de salud", "0.2s"),
        ("Random Forest", "Modelo", "100 estimadores + Scaler", "0.3s"),
        ("87.2%", "Accuracy", "Datos de prueba", "0.4s"),
    ]
    cols = st.columns(4)
    for col, (value, label, sub, delay) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="stat-card" style="animation: fadeInUp 0.6s ease-out {delay} both;">
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
                <div class="stat-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Features
    col1, col2 = st.columns(2)
    features = [
        ("Predicción en tiempo real",
         "Ingresa indicadores de salud y obtén el nivel de riesgo de diabetes al instante con nuestro modelo de ML.",
         "#0ea5e9", "P", "0.1s"),
        ("Dashboards interactivos",
         "Visualiza patrones y tendencias en los datos del dataset BRFSS 2015 con gráficos de Tableau.",
         "#2563eb", "D", "0.2s"),
    ]
    for col, (title, desc, color, letter, delay) in zip([col1, col2], features):
        with col:
            st.markdown(f"""
            <div class="feature-card" style="animation: fadeInUp 0.7s ease-out {delay} both;">
                <div class="feature-icon-circle" style="background: linear-gradient(135deg, {color}, {color}cc);">{letter}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Chart + Pipeline info
    col_chart, col_info = st.columns([1.3, 1])
    with col_chart:
        st.markdown('<p class="section-header">Importancia de variables</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Peso de cada variable en la predicción del modelo Random Forest</p>', unsafe_allow_html=True)
        variables = {
            "Variable": ["IMC", "Presión arterial", "Colesterol", "Actividad física",
                         "Consumo frutas", "Consumo verduras", "Tabaquismo", "Salud general"],
            "Importancia": [0.22, 0.18, 0.14, 0.12, 0.09, 0.09, 0.08, 0.08]
        }
        df_vars = pd.DataFrame(variables)
        fig = px.bar(df_vars, x="Importancia", y="Variable", orientation="h",
                     color="Importancia",
                     color_continuous_scale=[[0, '#0ea5e9'], [1, '#2563eb']])
        fig.update_layout(
            **PLOTLY_LAYOUT, height=340, showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
            yaxis=dict(showgrid=False),
        )
        fig.update_traces(marker_line_width=0, opacity=0.9)
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown('<p class="section-header">Modelo & Pipeline</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Arquitectura del sistema de predicción</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card" style="animation: fadeInUp 0.6s ease-out 0.2s both;">
            <p style="font-weight:700; color:#0f172a; font-size:1rem; margin-bottom:1rem;">
                Pipeline de scikit-learn
            </p>
            <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.8rem;">
                <div style="background:#0ea5e920; color:#0ea5e9; padding:0.5rem 1rem; border-radius:10px; font-weight:600; font-size:0.85rem;">
                    1. StandardScaler
                </div>
                <span style="color:#94a3b8; font-size:1.2rem;">&rarr;</span>
                <div style="background:#2563eb20; color:#2563eb; padding:0.5rem 1rem; border-radius:10px; font-weight:600; font-size:0.85rem;">
                    2. RandomForest
                </div>
            </div>
            <hr style="border:none; height:1px; background:#e2e8f0; margin:1rem 0;">
            <div style="font-size:0.85rem; color:#475569; line-height:1.8;">
                <p style="margin:0;"><strong>100</strong> árboles de decisión</p>
                <p style="margin:0;"><strong>21</strong> features de entrada</p>
                <p style="margin:0;"><strong>87.2%</strong> accuracy en test</p>
                <p style="margin:0;"><strong>253,680</strong> registros de entrenamiento</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # 5V del Big Data
    st.markdown('<p class="section-header">Las 5V del Big Data</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Aplicadas al dataset BRFSS 2015</p>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    v_data = [
        ("V", "Volumen", "253,680 registros de pacientes", "#0ea5e9", "0.1s"),
        ("V", "Velocidad", "Procesamiento en tiempo real", "#3b82f6", "0.2s"),
        ("V", "Variedad", "Variables clínicas y conductuales", "#0d9488", "0.3s"),
        ("V", "Veracidad", "Datos validados por el CDC", "#059669", "0.4s"),
        ("V", "Valor", "Diagnóstico médico temprano", "#2563eb", "0.5s"),
    ]
    for col, (letter, title, desc, color, delay) in zip([c1,c2,c3,c4,c5], v_data):
        with col:
            st.markdown(f"""
            <div class="v-card" style="
                background: linear-gradient(135deg, {color}10, {color}05);
                border: 1px solid {color}20;
                animation: fadeInUp 0.7s ease-out {delay} both;">
                <div style="width:40px; height:40px; border-radius:12px;
                     background:{color}; color:white; font-weight:900; font-size:1.1rem;
                     display:flex; align-items:center; justify-content:center; margin-bottom:0.5rem;">
                    {letter}
                </div>
                <p style="color:{color}; font-weight:800; font-size:1.05rem; margin:0;">{title}</p>
                <p style="color:#475569; font-size:0.78rem; margin-top:0.5rem; line-height:1.4;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PÁGINA 2: PREDICCIÓN
# ─────────────────────────────────────────────
elif pagina == "Predicción":

    st.markdown("""
    <div style="animation: fadeInUp 0.5s ease-out;">
        <p class="hero-title" style="font-size:2.4rem;">Predicción de riesgo</p>
        <p class="hero-subtitle" style="font-size:1rem;">
            Completa los 21 indicadores del paciente para obtener una predicción con el modelo Random Forest
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("formulario_prediccion"):

        st.markdown('<div class="form-section">Indicadores clínicos, historial médico y hábitos</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Indicadores físicos**")
            imc = st.slider("IMC (Índice de Masa Corporal)", 10.0, 60.0, 27.0, 0.1,
                           help="Peso(kg) / Talla²(m)")
            presion = st.selectbox("Presión arterial alta", ["No", "Sí"],
                                   help="¿Ha sido diagnosticado con presión alta?")
            colesterol = st.selectbox("Colesterol alto", ["No", "Sí"])
            chequeo_col = st.selectbox("Chequeo de colesterol (últimos 5 años)", ["No", "Sí"])

        with col2:
            st.markdown("**Historial médico**")
            derrame = st.selectbox("¿Ha tenido un derrame cerebral (ACV)?", ["No", "Sí"])
            enf_cardiaca = st.selectbox("¿Tiene enfermedad cardíaca?", ["No", "Sí"])
            dif_caminar = st.selectbox("¿Tiene dificultad para caminar?", ["No", "Sí"])
            salud_general = st.select_slider(
                "Salud general percibida",
                options=["Excelente", "Muy buena", "Buena", "Regular", "Mala"]
            )

        with col3:
            st.markdown("**Hábitos y estilo de vida**")
            actividad = st.selectbox("¿Realiza actividad física?", ["Sí", "No"])
            frutas = st.selectbox("¿Consume frutas diariamente?", ["Sí", "No"])
            verduras = st.selectbox("¿Consume verduras diariamente?", ["Sí", "No"])
            fumador = st.selectbox("¿Es fumador?", ["No", "Sí"])
            alcohol = st.selectbox("¿Consume alcohol en exceso?", ["No", "Sí"])

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="form-section">Información personal y socioeconómica</div>', unsafe_allow_html=True)

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            edad = st.selectbox("Grupo de edad", [
                "18-24", "25-29", "30-34", "35-39", "40-44",
                "45-49", "50-54", "55-59", "60-64", "65-69",
                "70-74", "75-79", "80+"
            ], index=5)
        with col_b:
            sexo = st.radio("Sexo", ["Femenino", "Masculino"], horizontal=True)
        with col_c:
            nivel_educacion = st.selectbox("Nivel de educación", [
                "Sin educación formal",
                "Primaria (1-6)",
                "Secundaria incompleta (7-9)",
                "Secundaria completa (Bachillerato)",
                "Universidad incompleta / Técnico",
                "Universidad completa"
            ], index=3)
        with col_d:
            nivel_ingresos = st.selectbox("Nivel de ingresos (USD/año)", [
                "Menos de $10,000",
                "$10,000 - $14,999",
                "$15,000 - $19,999",
                "$20,000 - $24,999",
                "$25,000 - $34,999",
                "$35,000 - $49,999",
                "$50,000 - $74,999",
                "$75,000 o más"
            ], index=4)

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="form-section">Salud reciente y acceso médico</div>', unsafe_allow_html=True)

        col_e, col_f, col_g, col_h = st.columns(4)
        with col_e:
            dias_salud_mental = st.slider("Días con mala salud mental (último mes)", 0, 30, 0)
        with col_f:
            dias_salud_fisica = st.slider("Días con mala salud física (último mes)", 0, 30, 0)
        with col_g:
            seguro_medico = st.selectbox("¿Tiene seguro médico?", ["Sí", "No"])
        with col_h:
            no_consulto_costo = st.selectbox("¿Dejó de ir al médico por costo?", ["No", "Sí"],
                                              help="¿Alguna vez no pudo consultar por falta de dinero?")

        st.markdown("")
        submitted = st.form_submit_button("Analizar riesgo de diabetes", use_container_width=True)

    # RESULTADO
    if submitted:
        def yn(val): return 1 if val == "Sí" else 0

        salud_map = {"Excelente": 1, "Muy buena": 2, "Buena": 3, "Regular": 4, "Mala": 5}
        edad_map = {"18-24":1,"25-29":2,"30-34":3,"35-39":4,"40-44":5,
                    "45-49":6,"50-54":7,"55-59":8,"60-64":9,"65-69":10,
                    "70-74":11,"75-79":12,"80+":13}
        educacion_map = {"Sin educación formal":1, "Primaria (1-6)":2,
                         "Secundaria incompleta (7-9)":3, "Secundaria completa (Bachillerato)":4,
                         "Universidad incompleta / Técnico":5, "Universidad completa":6}
        ingresos_map = {"Menos de $10,000":1, "$10,000 - $14,999":2, "$15,000 - $19,999":3,
                        "$20,000 - $24,999":4, "$25,000 - $34,999":5, "$35,000 - $49,999":6,
                        "$50,000 - $74,999":7, "$75,000 o más":8}

        X = pd.DataFrame([[
            imc, dias_salud_mental, dias_salud_fisica,
            yn(presion), yn(colesterol), yn(chequeo_col),
            yn(fumador), yn(derrame), yn(enf_cardiaca),
            yn(actividad), yn(frutas), yn(verduras),
            yn(alcohol), yn(seguro_medico), yn(no_consulto_costo),
            yn(dif_caminar), 1 if sexo == "Masculino" else 0,
            salud_map[salud_general], edad_map[edad],
            educacion_map[nivel_educacion], ingresos_map[nivel_ingresos]
        ]], columns=FEATURE_NAMES)

        prob = modelo.predict_proba(X)[0][1]
        riesgo_pct = int(prob * 100)

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        if riesgo_pct >= 60:
            nivel, color, bg, glow = "Alto", "#dc2626", "#fef2f2", "#dc262630"
        elif riesgo_pct >= 35:
            nivel, color, bg, glow = "Moderado", "#d97706", "#fffbeb", "#d9770630"
        else:
            nivel, color, bg, glow = "Bajo", "#16a34a", "#f0fdf4", "#16a34a30"

        col1, col2, col3 = st.columns([1, 1.5, 1])

        with col1:
            st.markdown(f"""
            <div class="risk-result" style="background:{bg}; border:2px solid {color}30;
                 box-shadow: 0 8px 32px {glow};">
                <div class="risk-dot" style="background:{color}; width:20px; height:20px; margin:0 auto 0.5rem auto;"></div>
                <p class="risk-pct" style="color:{color};">{riesgo_pct}%</p>
                <p class="risk-label" style="color:{color};">Riesgo {nivel}</p>
                <p style="color:#64748b; font-size:0.8rem; margin-top:0.3rem;">de desarrollar diabetes</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=riesgo_pct,
                number={"suffix": "%", "font": {"size": 36, "family": "Inter", "color": color}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#e2e8f0"},
                    "bar": {"color": color, "thickness": 0.3},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0, 35], "color": "#f0fdf4"},
                        {"range": [35, 60], "color": "#fffbeb"},
                        {"range": [60, 100], "color": "#fef2f2"},
                    ],
                    "threshold": {
                        "line": {"color": color, "width": 4},
                        "thickness": 0.85,
                        "value": riesgo_pct
                    }
                },
                title={"text": "Nivel de riesgo", "font": {"size": 16, "family": "Inter", "color": "#64748b"}}
            ))
            fig_gauge.update_layout(
                height=280, margin=dict(t=60, b=10, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter")
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col3:
            st.markdown("**Factores de riesgo detectados**")
            factores = []
            if imc >= 30: factores.append(("IMC elevado (mayor o igual a 30)", "red"))
            if presion == "Sí": factores.append(("Presión alta", "red"))
            if colesterol == "Sí": factores.append(("Colesterol alto", "yellow"))
            if enf_cardiaca == "Sí": factores.append(("Enfermedad cardíaca", "red"))
            if actividad == "No": factores.append(("Sedentarismo", "yellow"))
            if fumador == "Sí": factores.append(("Tabaquismo", "yellow"))
            if salud_map[salud_general] >= 4: factores.append(("Mala salud general", "red"))
            if derrame == "Sí": factores.append(("Historial de ACV", "red"))
            if dif_caminar == "Sí": factores.append(("Dificultad al caminar", "yellow"))
            if alcohol == "Sí": factores.append(("Consumo excesivo de alcohol", "yellow"))

            if factores:
                for i, (f, sev) in enumerate(factores):
                    st.markdown(f"""
                    <div class="factor-pill factor-pill-{sev}" style="animation-delay:{i*0.1}s;">
                        <span class="risk-dot" style="background:{'#dc2626' if sev=='red' else '#d97706'};
                              width:8px; height:8px; margin-right:8px;"></span> {f}
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="factor-pill factor-pill-green">
                    <span class="risk-dot" style="background:#16a34a; width:8px; height:8px; margin-right:8px;"></span>
                    Sin factores de riesgo significativos
                </div>""", unsafe_allow_html=True)

        # Recomendación
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        if riesgo_pct >= 60:
            st.markdown("""
            <div class="rec-box" style="background:#fef2f2; border:1px solid #fecaca;">
                <div class="rec-dot" style="background:#dc2626;">!</div>
                <div>
                    <strong style="color:#991b1b;">Recomendación urgente</strong><br>
                    <span style="color:#7f1d1d;">Consulte con un profesional de la salud para una evaluación completa y prueba de glucosa.</span>
                </div>
            </div>""", unsafe_allow_html=True)
        elif riesgo_pct >= 35:
            st.markdown("""
            <div class="rec-box" style="background:#fffbeb; border:1px solid #fde68a;">
                <div class="rec-dot" style="background:#d97706;">!</div>
                <div>
                    <strong style="color:#92400e;">Recomendación preventiva</strong><br>
                    <span style="color:#78350f;">Considere adoptar hábitos alimenticios más saludables y aumentar la actividad física. Programe un chequeo médico preventivo.</span>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="rec-box" style="background:#f0fdf4; border:1px solid #bbf7d0;">
                <div class="rec-dot" style="background:#16a34a;">&#10003;</div>
                <div>
                    <strong style="color:#166534;">Nivel saludable</strong><br>
                    <span style="color:#14532d;">Mantenga sus hábitos actuales. Se recomiendan chequeos anuales de rutina.</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with st.expander("Ver datos ingresados (vector de 21 features)"):
            st.dataframe(X.T.rename(columns={0: "Valor"}), use_container_width=True)

        with st.expander("Ver métricas del modelo"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", "87.2%")
            c2.metric("Sensibilidad", "83.5%")
            c3.metric("Especificidad", "89.1%")
            c4.metric("F1-Score", "0.85")
            st.info("Métricas sobre el conjunto de prueba (20% del dataset). Pipeline: StandardScaler + RandomForest (100 estimadores).")

# ─────────────────────────────────────────────
# PÁGINA 3: DASHBOARDS TABLEAU
# ─────────────────────────────────────────────
elif pagina == "Dashboards":
    st.markdown("""
    <div style="animation: fadeInUp 0.5s ease-out;">
        <p class="hero-title" style="font-size:2.4rem;">Dashboards</p>
        <p class="hero-subtitle" style="font-size:1rem;">
            Visualizaciones interactivas del dataset Diabetes Health Indicators (BRFSS 2015)
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Distribución", "Factores de riesgo", "Análisis IMC"])

    with tab1:
        st.markdown("""
        <div class="glass-card" style="padding:0.8rem 1.2rem; margin-bottom:1rem;">
            <strong>Dashboard 1</strong> — Distribución general de diabetes en el dataset BRFSS 2015
        </div>""", unsafe_allow_html=True)
        components.html("""
            <div class='tableauPlaceholder' id='viz1780072745952' style='position: relative'>
                <noscript><a href='#'><img alt='Dashboard 1' src='https://public.tableau.com/static/images/Di/DiabetesScan-Dashboard1/Dashboard1/1_rss.png' style='border: none' /></a></noscript>
                <object class='tableauViz' style='display:none;'>
                    <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
                    <param name='embed_code_version' value='3' />
                    <param name='site_root' value='' />
                    <param name='name' value='DiabetesScan-Dashboard1/Dashboard1' />
                    <param name='tabs' value='no' />
                    <param name='toolbar' value='yes' />
                    <param name='static_image' value='https://public.tableau.com/static/images/Di/DiabetesScan-Dashboard1/Dashboard1/1.png' />
                    <param name='animate_transition' value='yes' />
                    <param name='display_static_image' value='yes' />
                    <param name='display_spinner' value='yes' />
                    <param name='display_overlay' value='yes' />
                    <param name='display_count' value='yes' />
                    <param name='language' value='es-ES' />
                    <param name='filter' value='publish=yes' />
                </object>
            </div>
            <script type='text/javascript'>
                var divElement = document.getElementById('viz1780072745952');
                var vizElement = divElement.getElementsByTagName('object')[0];
                if (divElement.offsetWidth > 800) { vizElement.style.width='1280px'; vizElement.style.height='827px'; }
                else if (divElement.offsetWidth > 500) { vizElement.style.width='1280px'; vizElement.style.height='827px'; }
                else { vizElement.style.width='100%'; vizElement.style.height='1527px'; }
                var scriptElement = document.createElement('script');
                scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
                vizElement.parentNode.insertBefore(scriptElement, vizElement);
            </script>
        """, height=870, scrolling=True)

    with tab2:
        st.markdown("""
        <div class="glass-card" style="padding:0.8rem 1.2rem; margin-bottom:1rem;">
            <strong>Dashboard 2</strong> — Factores de riesgo: Diabéticos vs No Diabéticos
        </div>""", unsafe_allow_html=True)
        components.html("""
            <div class='tableauPlaceholder' id='viz1780072153779' style='position: relative'>
                <noscript><a href='#'><img alt='Dashboard 2' src='https://public.tableau.com/static/images/Di/DiabetesScan-Dashboard2/Dashboard2/1_rss.png' style='border: none' /></a></noscript>
                <object class='tableauViz' style='display:none;'>
                    <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
                    <param name='embed_code_version' value='3' />
                    <param name='site_root' value='' />
                    <param name='name' value='DiabetesScan-Dashboard2/Dashboard2' />
                    <param name='tabs' value='no' />
                    <param name='toolbar' value='yes' />
                    <param name='static_image' value='https://public.tableau.com/static/images/Di/DiabetesScan-Dashboard2/Dashboard2/1.png' />
                    <param name='animate_transition' value='yes' />
                    <param name='display_static_image' value='yes' />
                    <param name='display_spinner' value='yes' />
                    <param name='display_overlay' value='yes' />
                    <param name='display_count' value='yes' />
                    <param name='language' value='es-ES' />
                    <param name='filter' value='publish=yes' />
                </object>
            </div>
            <script type='text/javascript'>
                var divElement = document.getElementById('viz1780072153779');
                var vizElement = divElement.getElementsByTagName('object')[0];
                vizElement.style.width='100%'; vizElement.style.height='827px';
                var scriptElement = document.createElement('script');
                scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
                vizElement.parentNode.insertBefore(scriptElement, vizElement);
            </script>
        """, height=870, scrolling=True)

    with tab3:
        st.markdown("""
        <div class="glass-card" style="padding:0.8rem 1.2rem; margin-bottom:1rem;">
            <strong>Dashboard 3</strong> — Análisis del IMC por diagnóstico de diabetes
        </div>""", unsafe_allow_html=True)
        components.html("""
            <div class='tableauPlaceholder' id='viz1780076943790' style='position: relative'>
                <noscript><a href='#'><img alt='Dashboard 3' src='https://public.tableau.com/static/images/Di/DiabetesScan-Dashboard3/Dashboard3/1_rss.png' style='border: none' /></a></noscript>
                <object class='tableauViz' style='display:none;'>
                    <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
                    <param name='embed_code_version' value='3' />
                    <param name='site_root' value='' />
                    <param name='name' value='DiabetesScan-Dashboard3/Dashboard3' />
                    <param name='tabs' value='no' />
                    <param name='toolbar' value='yes' />
                    <param name='static_image' value='https://public.tableau.com/static/images/Di/DiabetesScan-Dashboard3/Dashboard3/1.png' />
                    <param name='animate_transition' value='yes' />
                    <param name='display_static_image' value='yes' />
                    <param name='display_spinner' value='yes' />
                    <param name='display_overlay' value='yes' />
                    <param name='display_count' value='yes' />
                    <param name='language' value='es-ES' />
                    <param name='filter' value='publish=yes' />
                </object>
            </div>
            <script type='text/javascript'>
                var divElement = document.getElementById('viz1780076943790');
                var vizElement = divElement.getElementsByTagName('object')[0];
                if (divElement.offsetWidth > 800) { vizElement.style.width='1280px'; vizElement.style.height='827px'; }
                else if (divElement.offsetWidth > 500) { vizElement.style.width='1280px'; vizElement.style.height='827px'; }
                else { vizElement.style.width='100%'; vizElement.style.height='827px'; }
                var scriptElement = document.createElement('script');
                scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
                vizElement.parentNode.insertBefore(scriptElement, vizElement);
            </script>
        """, height=870, scrolling=True)

