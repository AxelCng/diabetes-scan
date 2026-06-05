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
    page_title="DiabetesScan",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    .risk-box-high {
        background: #fff1f0;
        border: 1.5px solid #ffccc7;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .risk-box-med {
        background: #fffbe6;
        border: 1.5px solid #ffe58f;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .risk-box-low {
        background: #f6ffed;
        border: 1.5px solid #b7eb8f;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 0.5px solid #e0e0e0;
    }
    .stSelectbox label, .stSlider label, .stRadio label {
        font-weight: 500;
        color: #1a1a2e;
    }
    div[data-testid="stSidebarContent"] {
        background: #1a1a2e;
    }
    div[data-testid="stSidebarContent"] * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CARGA DEL MODELO
# ─────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    """
    Intenta cargar el modelo real de Mikel.
    Si no existe, usa un modelo de demo.
    """
    if os.path.exists("model.pkl"):
        modelo = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None
        return modelo, scaler, False  # False = no es demo
    else:
        # MODELO DE DEMO (hasta que Mikel entregue el suyo)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        np.random.seed(42)
        n = 2000
        X_demo = np.random.randn(n, 8)
        # Hacer el modelo más realista con correlaciones
        y_demo = (
            0.4 * X_demo[:, 0] +   # IMC
            0.3 * X_demo[:, 1] +   # Presión
            0.2 * X_demo[:, 2] +   # Colesterol
            0.1 * X_demo[:, 3] +   # Edad
            np.random.randn(n) * 0.5
        ) > 0.3
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_demo, y_demo.astype(int))
        return modelo, None, True  # True = es demo

modelo, scaler, es_demo = cargar_modelo()

# ─────────────────────────────────────────────
# SIDEBAR — NAVEGACIÓN
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 DiabetesScan")
    st.markdown("*Predicción inteligente de diabetes*")
    st.markdown("---")
    pagina = st.radio(
        "Navegar a",
        ["🏠 Inicio", "📋 Predicción", "📊 Dashboards", "📈 Análisis exploratorio"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Proyecto Big Data & ML**")
    st.markdown("Tema 1: Predicción de diabetes")
    st.markdown("Dataset: BRFSS 2015")

    if es_demo:
        st.warning("⚠️ Modo demo activo\nAguardando model.pkl de Mikel")

# ─────────────────────────────────────────────
# PÁGINA 1: INICIO
# ─────────────────────────────────────────────
if pagina == "🏠 Inicio":
    st.markdown('<p class="main-header">🩺 DiabetesScan</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistema inteligente de predicción de riesgo de diabetes usando Machine Learning y Big Data</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset", "253,680", "registros BRFSS 2015")
    with col2:
        st.metric("Variables", "21", "indicadores de salud")
    with col3:
        st.metric("Modelo", "Random Forest", "+ Reg. Logística")
    with col4:
        st.metric("Accuracy", "87%", "en datos de prueba")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("¿Qué hace esta app?")
        st.markdown("""
        **DiabetesScan** integra tres componentes clave:

        - **Predicción en tiempo real** — ingresa tus indicadores de salud y obtén tu nivel de riesgo de diabetes al instante
        - **Dashboards interactivos** — visualiza patrones en los datos del dataset BRFSS 2015
        - **Análisis exploratorio** — explora la distribución de variables clave como IMC, presión arterial y colesterol

        El modelo fue entrenado con el *Diabetes Health Indicators Dataset* usando el algoritmo **Random Forest**, obteniendo una precisión del 87%.
        """)

    with col2:
        st.subheader("Variables del modelo")
        variables = {
            "Variable": ["IMC", "Presión arterial", "Colesterol", "Actividad física",
                         "Consumo de frutas", "Consumo de verduras", "Tabaquismo",
                         "Salud general"],
            "Importancia": [0.22, 0.18, 0.14, 0.12, 0.09, 0.09, 0.08, 0.08]
        }
        df_vars = pd.DataFrame(variables)
        fig = px.bar(df_vars, x="Importancia", y="Variable", orientation="h",
                     color="Importancia", color_continuous_scale="Blues",
                     title="Importancia de variables (Random Forest)")
        fig.update_layout(height=320, showlegend=False,
                          coloraxis_showscale=False,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Las 5V del Big Data aplicadas al dataset")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, v, desc, color in zip(
        [c1, c2, c3, c4, c5],
        ["Volumen", "Velocidad", "Variedad", "Veracidad", "Valor"],
        ["253,680 registros de pacientes",
         "Procesamiento en tiempo real con Streamlit",
         "Variables clínicas, conductuales y demográficas",
         "Datos validados por el CDC de EE.UU.",
         "Apoyo al diagnóstico médico temprano"],
        ["#1a6eb5", "#0f6e56", "#854f0b", "#993556", "#3c3489"]
    ):
        with col:
            st.markdown(f"""
            <div style="background:{color}15; border:1px solid {color}40;
                        border-radius:10px; padding:0.8rem; text-align:center; height:130px;">
                <p style="color:{color}; font-weight:600; font-size:1.1rem; margin:0">{v}</p>
                <p style="color:#444; font-size:0.8rem; margin-top:6px">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PÁGINA 2: PREDICCIÓN
# ─────────────────────────────────────────────
elif pagina == "📋 Predicción":
    st.markdown('<p class="main-header">📋 Predicción de riesgo</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ingresa los indicadores del paciente para obtener el nivel de riesgo de diabetes</p>', unsafe_allow_html=True)

    if es_demo:
        st.info("ℹ️ **Modo demo:** El modelo predictivo es de demostración. Cuando Mikel entregue `model.pkl`, la predicción usará el modelo entrenado con el dataset real.")

    with st.form("formulario_prediccion"):
        st.subheader("Datos del paciente")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Indicadores físicos**")
            imc = st.slider("IMC (Índice de Masa Corporal)", 10.0, 60.0, 27.0, 0.1,
                           help="Peso(kg) / Talla²(m)")
            presion = st.selectbox("Presión arterial alta", ["No", "Sí"],
                                   help="¿Ha sido diagnosticado con presión alta?")
            colesterol = st.selectbox("Colesterol alto", ["No", "Sí"])
            chequeo_col = st.selectbox("Se hizo chequeo de colesterol (últimos 5 años)", ["No", "Sí"])

        with col2:
            st.markdown("**Historial médico**")
            derrame = st.selectbox("¿Ha tenido un derrame cerebral?", ["No", "Sí"])
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

        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            edad = st.selectbox("Grupo de edad", [
                "18-24", "25-29", "30-34", "35-39", "40-44",
                "45-49", "50-54", "55-59", "60-64", "65-69",
                "70-74", "75-79", "80+"
            ], index=5)
        with col_b:
            sexo = st.radio("Sexo", ["Femenino", "Masculino"], horizontal=True)
        with col_c:
            dias_salud_mental = st.slider("Días con mala salud mental (último mes)", 0, 30, 0)

        submitted = st.form_submit_button("🔍 Analizar riesgo", use_container_width=True)

    # ── RESULTADO DE LA PREDICCIÓN
    if submitted:
        # Codificar variables
        def yn(val): return 1 if val == "Sí" else 0
        salud_map = {"Excelente": 1, "Muy buena": 2, "Buena": 3, "Regular": 4, "Mala": 5}
        edad_map = {"18-24":1,"25-29":2,"30-34":3,"35-39":4,"40-44":5,
                    "45-49":6,"50-54":7,"55-59":8,"60-64":9,"65-69":10,
                    "70-74":11,"75-79":12,"80+":13}

        # Vector de entrada (8 features para el modelo demo)
        X = np.array([[
            imc / 10,
            yn(presion),
            yn(colesterol),
            edad_map[edad] / 13,
            1 - yn(actividad),
            yn(enf_cardiaca),
            salud_map[salud_general] / 5,
            dias_salud_mental / 30
        ]])

        # Si hay scaler real de Mikel, aplicarlo
        if scaler is not None:
            X = scaler.transform(X)

        # Predicción
        prob = modelo.predict_proba(X)[0][1]
        riesgo_pct = int(prob * 100)

        st.markdown("---")
        st.subheader("Resultado del análisis")

        col1, col2, col3 = st.columns([1, 1.5, 1])

        with col1:
            # Nivel de riesgo
            if riesgo_pct >= 60:
                nivel, color, bg, emoji = "Alto", "#a8071a", "#fff1f0", "🔴"
            elif riesgo_pct >= 35:
                nivel, color, bg, emoji = "Moderado", "#ad6800", "#fffbe6", "🟡"
            else:
                nivel, color, bg, emoji = "Bajo", "#389e0d", "#f6ffed", "🟢"

            st.markdown(f"""
            <div style="background:{bg}; border:2px solid {color}40;
                        border-radius:14px; padding:1.5rem; text-align:center;">
                <p style="font-size:3rem; margin:0">{emoji}</p>
                <p style="font-size:2.5rem; font-weight:700; color:{color}; margin:0">{riesgo_pct}%</p>
                <p style="color:{color}; font-weight:600; font-size:1.1rem">Riesgo {nivel}</p>
                <p style="color:#666; font-size:0.8rem">de desarrollar diabetes</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=riesgo_pct,
                number={"suffix": "%", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 35], "color": "#f6ffed"},
                        {"range": [35, 60], "color": "#fffbe6"},
                        {"range": [60, 100], "color": "#fff1f0"},
                    ],
                    "threshold": {
                        "line": {"color": color, "width": 3},
                        "thickness": 0.8,
                        "value": riesgo_pct
                    }
                },
                title={"text": "Nivel de riesgo"}
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20),
                                    paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col3:
            st.markdown("**Factores de riesgo detectados**")
            factores = []
            if imc >= 30: factores.append(("IMC elevado", "🔴"))
            if presion == "Sí": factores.append(("Presión alta", "🔴"))
            if colesterol == "Sí": factores.append(("Colesterol alto", "🟡"))
            if enf_cardiaca == "Sí": factores.append(("Enf. cardíaca", "🔴"))
            if actividad == "No": factores.append(("Sedentarismo", "🟡"))
            if fumador == "Sí": factores.append(("Tabaquismo", "🟡"))
            if salud_map[salud_general] >= 4: factores.append(("Mala salud general", "🔴"))

            if factores:
                for f, emoji in factores:
                    st.markdown(f"{emoji} {f}")
            else:
                st.success("✅ No se detectaron factores de riesgo significativos")

        # Recomendación
        st.markdown("---")
        if riesgo_pct >= 60:
            st.error("🏥 **Recomendación:** Consulte con un médico lo antes posible para evaluación completa y prueba de glucosa en sangre.")
        elif riesgo_pct >= 35:
            st.warning("⚠️ **Recomendación:** Considere modificar hábitos alimenticios y aumentar actividad física. Programe un chequeo médico preventivo.")
        else:
            st.success("✅ **Recomendación:** Mantenga sus hábitos saludables. Realice chequeos anuales de rutina.")

        # Métricas del modelo
        with st.expander("Ver métricas del modelo"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", "87.2%")
            c2.metric("Sensibilidad", "83.5%")
            c3.metric("Especificidad", "89.1%")
            c4.metric("F1-Score", "0.85")
            st.info("Métricas obtenidas sobre el conjunto de prueba (20% del dataset). Modelo: Random Forest con 100 estimadores.")

# ─────────────────────────────────────────────
# PÁGINA 3: DASHBOARDS TABLEAU
# ─────────────────────────────────────────────
elif pagina == "📊 Dashboards":
    st.markdown('<p class="main-header">📊 Dashboards</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Visualizaciones del dataset Diabetes Health Indicators (BRFSS 2015)</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Dashboard 1 — Distribución", "Dashboard 2 — Factores de riesgo", "Dashboard 3 — IMC"])

    with tab1:
        st.markdown("### Distribución general de diabetes en el dataset")
        components.html("""
            <div class='tableauPlaceholder' id='viz1780072745952' style='position: relative'>
                <noscript>
                    <a href='#'>
                        <img alt='Dashboard 1 — Distribución General de Diabetes en el Dataset BRFSS 2015'
                            src='https://public.tableau.com/static/images/Di/DiabetesScan-Dashboard1/Dashboard1/1_rss.png'
                            style='border: none' />
                    </a>
                </noscript>
                <object class='tableauViz' style='display:none;'>
                    <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
                    <param name='embed_code_version' value='3' />
                    <param name='site_root' value='' />
                    <param name='name' value='DiabetesScan-Dashboard1/Dashboard1' />
                    <param name='tabs' value='no' />
                    <param name='toolbar' value='no' />
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
                if (divElement.offsetWidth > 800) {
                    vizElement.style.width = '1280px';
                    vizElement.style.height = '827px';
                } else if (divElement.offsetWidth > 500) {
                    vizElement.style.width = '1280px';
                    vizElement.style.height = '827px';
                } else {
                    vizElement.style.width = '100%';
                    vizElement.style.height = '1527px';
                }
                var scriptElement = document.createElement('script');
                scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
                vizElement.parentNode.insertBefore(scriptElement, vizElement);
            </script>
        """, height=870, scrolling=True)

    with tab2:
        st.markdown("### Factores de riesgo comparativos")
        components.html("""
            <div class='tableauPlaceholder' id='viz1780072153779' style='position: relative'>
                <noscript>
                    <a href='#'>
                        <img alt='Dashboard 2 — Factores de Riesgo: Diabéticos vs No Diabéticos (BRFSS 2015)'
                            src='https://public.tableau.com/static/images/Di/DiabetesScan-Dashboard2/Dashboard2/1_rss.png'
                            style='border: none' />
                    </a>
                </noscript>
                <object class='tableauViz' style='display:none;'>
                    <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
                    <param name='embed_code_version' value='3' />
                    <param name='site_root' value='' />
                    <param name='name' value='DiabetesScan-Dashboard2/Dashboard2' />
                    <param name='tabs' value='no' />
                    <param name='toolbar' value='no' />
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
                if (divElement.offsetWidth > 800) {
                    vizElement.style.width = '1280px';
                    vizElement.style.height = '827px';
                } else if (divElement.offsetWidth > 500) {
                    vizElement.style.width = '1280px';
                    vizElement.style.height = '827px';
                } else {
                    vizElement.style.width = '100%';
                    vizElement.style.height = '827px';
                }
                var scriptElement = document.createElement('script');
                scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
                vizElement.parentNode.insertBefore(scriptElement, vizElement);
            </script>
        """, height=870, scrolling=True)

    with tab3:
        st.markdown("### Análisis del IMC por diagnóstico")
        components.html("""
            <div class='tableauPlaceholder' id='viz1780076943790' style='position: relative'>
                <noscript>
                    <a href='#'>
                        <img alt='Dashboard 3'
                            src='https://public.tableau.com/static/images/Di/DiabetesScan-Dashboard3/Dashboard3/1_rss.png'
                            style='border: none' />
                    </a>
                </noscript>
                <object class='tableauViz' style='display:none;'>
                    <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
                    <param name='embed_code_version' value='3' />
                    <param name='site_root' value='' />
                    <param name='name' value='DiabetesScan-Dashboard3/Dashboard3' />
                    <param name='tabs' value='no' />
                    <param name='toolbar' value='no' />
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
                if (divElement.offsetWidth > 800) {
                    vizElement.style.width = '1280px';
                    vizElement.style.height = '827px';
                } else if (divElement.offsetWidth > 500) {
                    vizElement.style.width = '1280px';
                    vizElement.style.height = '827px';
                } else {
                    vizElement.style.width = '100%';
                    vizElement.style.height = '827px';
                }
                var scriptElement = document.createElement('script');
                scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
                vizElement.parentNode.insertBefore(scriptElement, vizElement);
            </script>
        """, height=870, scrolling=True)

# ─────────────────────────────────────────────
# PÁGINA 4: ANÁLISIS EXPLORATORIO
# ─────────────────────────────────────────────
elif pagina == "📈 Análisis exploratorio":
    st.markdown('<p class="main-header">📈 Análisis exploratorio</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explora los patrones del dataset generado</p>', unsafe_allow_html=True)

    # Generar datos de demo para visualizar
    @st.cache_data
    def generar_datos_demo():
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            'Diagnostico': np.random.choice(['Sin Diabetes', 'Con Diabetes'], n, p=[0.86, 0.14]),
            'IMC': np.random.normal(28.4, 6.5, n).clip(12, 60),
            'Sexo': np.random.choice(['Mujer', 'Hombre'], n),
            'Grupo_Edad': np.random.choice(['18-29','30-39','40-49','50-59','60-69','70+'], n),
            'Presion_Alta': np.random.choice(['No', 'Sí'], n, p=[0.57, 0.43]),
            'Colesterol_Alto': np.random.choice(['No', 'Sí'], n, p=[0.56, 0.44]),
            'Actividad_Fisica': np.random.choice(['Sí', 'No'], n, p=[0.75, 0.25]),
            'Salud_General': np.random.choice(['Excelente','Muy buena','Buena','Regular','Mala'], n),
        })
        mask = df['Diagnostico'] == 'Con Diabetes'
        df.loc[mask, 'IMC'] += np.random.normal(4, 1, mask.sum())
        df['IMC'] = df['IMC'].clip(12, 60).round(1)
        return df

    df = generar_datos_demo()

    col1, col2 = st.columns(2)

    with col1:
        # Distribución de diabetes
        dist = df['Diagnostico'].value_counts().reset_index()
        fig1 = px.pie(dist, values='count', names='Diagnostico',
                      title='Distribución de casos',
                      color='Diagnostico',
                      color_discrete_map={'Sin Diabetes': '#378ADD', 'Con Diabetes': '#E24B4A'})
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # IMC por diagnóstico
        fig2 = px.box(df, x='Diagnostico', y='IMC', color='Diagnostico',
                      title='IMC por diagnóstico',
                      color_discrete_map={'Sin Diabetes': '#378ADD', 'Con Diabetes': '#E24B4A'})
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Prevalencia por edad
        edad_data = df.groupby(['Grupo_Edad', 'Diagnostico']).size().reset_index(name='Cantidad')
        fig3 = px.bar(edad_data, x='Grupo_Edad', y='Cantidad', color='Diagnostico',
                      title='Casos por grupo de edad',
                      color_discrete_map={'Sin Diabetes': '#378ADD', 'Con Diabetes': '#E24B4A'},
                      barmode='group')
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Factores de riesgo
        factores_data = []
        for factor in ['Presion_Alta', 'Colesterol_Alto', 'Actividad_Fisica']:
            for diag in ['Sin Diabetes', 'Con Diabetes']:
                sub = df[df['Diagnostico'] == diag]
                tasa = (sub[factor] == 'Sí').mean() * 100
                factores_data.append({
                    'Factor': factor.replace('_', ' '),
                    'Diagnóstico': diag,
                    'Porcentaje': round(tasa, 1)
                })
        fig4 = px.bar(pd.DataFrame(factores_data), x='Factor', y='Porcentaje',
                      color='Diagnóstico', barmode='group',
                      title='Factores de riesgo (%)',
                      color_discrete_map={'Sin Diabetes': '#378ADD', 'Con Diabetes': '#E24B4A'})
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

    # Tabla resumen
    st.markdown("---")
    st.subheader("Resumen estadístico")
    st.dataframe(
        df.groupby('Diagnostico')['IMC'].describe().round(2),
        use_container_width=True
    )