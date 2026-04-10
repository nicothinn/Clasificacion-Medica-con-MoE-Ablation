"""
app.py — Dashboard Interactivo del Sistema MoE Medical Vision Router.

Funcionalidades obligatorias de la consigna (items 15-22):
  15. Carga de imagen: PNG, JPEG, NIfTI. Deteccion 2D/3D automatica.
  16. Preprocesado transparente: dimensiones originales y adaptadas.
  17. Inferencia en tiempo real: etiqueta, confianza, tiempo en ms.
  18. Attention Heatmap del Router ViT sobre imagen original.
  19. Panel del experto activado: nombre, arquitectura, dataset, gating score.
  20. Panel del ablation study: tabla comparativa de Routing Accuracy.
  21. Load Balance: grafica de barras con f_i acumulado.
  22. OOD Detection: alerta cuando la entropia supera el umbral.

Ejecutar con:
  cd dashboard
  streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from moe_inference import MoEInferenceEngine
from mock_models import EXPERT_INFO


# ======================================================================
#  Configuracion de pagina
# ======================================================================
st.set_page_config(
    page_title="MoE Medical Vision Router",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
#  CSS personalizado
# ======================================================================
st.markdown("""
<style>
    /* --- Fuente global --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* --- Header principal --- */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #a0a0c0;
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* --- Tarjetas de metricas --- */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card h4 {
        color: #7c83ff;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0 0 0.3rem 0;
        font-weight: 600;
    }
    .metric-card .value {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-card .sub {
        color: #888;
        font-size: 0.78rem;
        margin: 0.2rem 0 0 0;
    }

    /* --- Panel del experto --- */
    .expert-panel {
        background: linear-gradient(145deg, #0d1117, #161b22);
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }
    .expert-panel .expert-name {
        color: #58a6ff;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .expert-panel .detail-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .expert-panel .detail-label {
        color: #8b949e;
        font-size: 0.82rem;
    }
    .expert-panel .detail-value {
        color: #c9d1d9;
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* --- Etiqueta de prediccion --- */
    .prediction-badge {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0.8rem 0;
        box-shadow: 0 4px 12px rgba(35,134,54,0.3);
    }

    /* --- Alerta OOD --- */
    .ood-alert {
        background: linear-gradient(135deg, #b91c1c, #dc2626);
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 0.95rem;
        font-weight: 500;
        margin: 0.8rem 0;
        box-shadow: 0 4px 12px rgba(185,28,28,0.3);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }

    /* --- Barra de confianza --- */
    .confidence-bar-bg {
        background: #21262d;
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
        margin: 0.4rem 0;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
    }

    /* --- Tabla ablation --- */
    .ablation-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .ablation-table th {
        background: #161b22;
        color: #7c83ff;
        padding: 0.7rem 1rem;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        text-align: left;
        font-weight: 600;
    }
    .ablation-table td {
        background: #0d1117;
        color: #c9d1d9;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        border-top: 1px solid #21262d;
    }
    .ablation-table tr:hover td {
        background: #161b22;
    }

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #58a6ff;
    }

    /* --- Pestanas --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #161b22;
        border-radius: 8px;
        color: #8b949e;
        border: 1px solid #30363d;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: #21262d;
        color: #58a6ff;
        border-color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================
#  Inicializacion del estado de sesion
# ======================================================================
if "load_counts" not in st.session_state:
    st.session_state.load_counts = np.zeros(5)
if "total_inferences" not in st.session_state:
    st.session_state.total_inferences = 0
if "inference_history" not in st.session_state:
    st.session_state.inference_history = []


# ======================================================================
#  Carga del motor de inferencia (cacheado)
# ======================================================================
@st.cache_resource
def get_engine():
    """Carga el motor MoE una sola vez (persistente entre recargas)."""
    return MoEInferenceEngine(use_mock=True)


engine = get_engine()


# ======================================================================
#  Header
# ======================================================================
st.markdown("""
<div class="main-header">
    <h1>🧠 MoE Medical Vision Router</h1>
    <p>Sistema de Clasificacion Medica con Mixture of Experts — Dashboard Interactivo</p>
</div>
""", unsafe_allow_html=True)


# ======================================================================
#  Sidebar — Carga de archivo (#15)
# ======================================================================
with st.sidebar:
    st.markdown("### 📁 Cargar Imagen Medica")
    st.markdown(
        "Sube una imagen **PNG/JPEG** (2D) o un volumen **NIfTI** (3D). "
        "El sistema detecta automaticamente el tipo de entrada."
    )

    uploaded_file = st.file_uploader(
        "Arrastra o selecciona un archivo",
        type=["png", "jpg", "jpeg", "nii", "gz", "mha"],
        help="Formatos soportados: PNG, JPEG, NIfTI (.nii, .nii.gz), MHA",
    )

    st.markdown("---")

    st.markdown("### ⚙ Configuracion")
    ood_threshold = st.slider(
        "Umbral OOD (% entropia max)",
        min_value=50, max_value=99, value=85,
        help="Si la entropia del router supera este % de la entropia maxima, "
             "se activa la alerta OOD.",
    )

    st.markdown("---")

    # Estadisticas de sesion
    st.markdown("### 📊 Estadisticas de Sesion")
    st.metric("Total de inferencias", st.session_state.total_inferences)

    if st.button("🔄 Reiniciar contadores"):
        st.session_state.load_counts = np.zeros(5)
        st.session_state.total_inferences = 0
        st.session_state.inference_history = []
        st.rerun()


# ======================================================================
#  Contenido principal
# ======================================================================
if uploaded_file is not None:
    # ------- Ejecutar inferencia -------
    try:
        result = engine.run(uploaded_file)
    except Exception as e:
        st.error(f"Error durante la inferencia: {e}")
        st.stop()

    # Actualizar contadores de Load Balance (#21)
    st.session_state.load_counts[result.expert_id] += 1
    st.session_state.total_inferences += 1
    st.session_state.inference_history.append({
        "file": uploaded_file.name,
        "expert": result.expert_name,
        "label": result.class_label,
        "confidence": result.confidence,
        "is_ood": result.is_ood,
    })

    # ------- OOD Alert (#22) -------
    if result.is_ood:
        st.markdown(
            f'<div class="ood-alert">'
            f'⚠ ALERTA OOD — Imagen fuera de distribucion detectada<br>'
            f'<small>Entropia: {result.entropy:.3f} nats '
            f'(umbral: {result.ood_threshold:.3f})</small>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ------- Layout principal: 3 columnas -------
    col_left, col_center, col_right = st.columns([1.2, 1.5, 1.3])

    # ===== COLUMNA IZQUIERDA: Imagen + Preprocesado (#15, #16) =====
    with col_left:
        st.markdown("#### 🖼 Imagen de Entrada")

        # Mostrar imagen original
        if result.display_image is not None:
            st.image(result.display_image, use_container_width=True,
                     caption="Imagen original")

        # Info de preprocesado (#16)
        dim_type = "3D (Volumen CT)" if result.is_3d else "2D (Imagen plana)"
        st.markdown(
            f'<div class="metric-card">'
            f'<h4>Deteccion Automatica</h4>'
            f'<p class="value">{dim_type}</p>'
            f'<p class="sub">Dims originales: {result.original_shape}</p>'
            f'<p class="sub">Dims adaptadas: {result.processed_shape}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Timing (#17)
        st.markdown(
            f'<div class="metric-card">'
            f'<h4>Tiempos de Inferencia</h4>'
            f'<p class="value">{result.total_ms:.1f} ms total</p>'
            f'<p class="sub">Preprocesado: {result.preprocess_ms:.1f} ms</p>'
            f'<p class="sub">Router ViT: {result.router_ms:.1f} ms</p>'
            f'<p class="sub">Experto: {result.expert_ms:.1f} ms</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ===== COLUMNA CENTRAL: Heatmap + Gating Scores (#18) =====
    with col_center:
        st.markdown("#### 🔥 Attention Heatmap del Router")

        if result.heatmap_image is not None:
            st.image(result.heatmap_image, use_container_width=True,
                     caption="Mapa de atencion del ViT sobre la imagen")

        # Gating Scores (barra horizontal)
        st.markdown("#### 📊 Gating Scores del Router")

        expert_labels = [f"Exp {i}" for i in range(5)]
        scores = result.gating_scores

        for i in range(5):
            info = EXPERT_INFO[i]
            score = scores[i]
            color = "#58a6ff" if i == result.expert_id else "#30363d"
            indicator = " ← ACTIVADO" if i == result.expert_id else ""
            pct = score * 100

            st.markdown(
                f'<div style="margin-bottom: 6px;">'
                f'<span style="color: #8b949e; font-size: 0.78rem;">'
                f'Exp {i}: {info["dataset"]}{indicator}</span>'
                f'<div class="confidence-bar-bg">'
                f'<div class="confidence-bar-fill" '
                f'style="width: {pct}%; background: {color};"></div>'
                f'</div>'
                f'<span style="color: #c9d1d9; font-size: 0.75rem;">'
                f'{pct:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ===== COLUMNA DERECHA: Experto + Prediccion (#17, #19) =====
    with col_right:
        st.markdown("#### 🩺 Resultado de Clasificacion")

        # Prediccion Badge (#17)
        if not result.is_ood:
            st.markdown(
                f'<div class="prediction-badge">'
                f'✅ {result.class_label}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="prediction-badge" '
                f'style="background: linear-gradient(135deg, #6e4000, #9a6700);">'
                f'⚠ {result.class_label} (baja confianza)'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Confianza
        conf_pct = result.confidence * 100
        conf_color = (
            "#2ea043" if conf_pct > 80
            else "#d29922" if conf_pct > 50
            else "#f85149"
        )
        st.markdown(
            f'<div class="metric-card">'
            f'<h4>Confianza del Experto</h4>'
            f'<p class="value" style="color: {conf_color};">'
            f'{conf_pct:.1f}%</p>'
            f'<div class="confidence-bar-bg">'
            f'<div class="confidence-bar-fill" '
            f'style="width: {conf_pct}%; background: {conf_color};"></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Panel del experto activado (#19)
        st.markdown(
            f'<div class="expert-panel">'
            f'<div class="expert-name">🏥 {result.expert_name}</div>'
            f'<div class="detail-row">'
            f'  <span class="detail-label">Arquitectura</span>'
            f'  <span class="detail-value">{result.expert_arch}</span>'
            f'</div>'
            f'<div class="detail-row">'
            f'  <span class="detail-label">Dataset de origen</span>'
            f'  <span class="detail-value">{result.expert_dataset}</span>'
            f'</div>'
            f'<div class="detail-row">'
            f'  <span class="detail-label">Gating Score</span>'
            f'  <span class="detail-value">'
            f'{result.gating_scores[result.expert_id]*100:.1f}%</span>'
            f'</div>'
            f'<div class="detail-row">'
            f'  <span class="detail-label">Tipo de entrada</span>'
            f'  <span class="detail-value">'
            f'{"Volumen 3D" if result.is_3d else "Imagen 2D"}</span>'
            f'</div>'
            f'<div class="detail-row">'
            f'  <span class="detail-label">Clases del experto</span>'
            f'  <span class="detail-value">'
            f'{len(result.class_names)} clases</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Distribucion de probabilidades por clase
        st.markdown("##### Probabilidades por Clase")
        if len(result.all_class_probs) > 0 and len(result.class_names) > 0:
            df_probs = pd.DataFrame({
                "Clase": result.class_names,
                "Probabilidad": result.all_class_probs,
            }).sort_values("Probabilidad", ascending=False)
            st.dataframe(
                df_probs.style.format({"Probabilidad": "{:.3f}"}),
                hide_index=True,
                use_container_width=True,
            )

    # ======================================================================
    #  PANELES INFERIORES
    # ======================================================================
    st.markdown("---")

    tab_ablation, tab_balance, tab_history = st.tabs([
        "📋 Ablation Study", "⚖ Load Balance", "📜 Historial"
    ])

    # ===== TAB 1: Ablation Study (#20) =====
    with tab_ablation:
        st.markdown("### Ablation Study del Router — Tabla Comparativa")
        st.markdown(
            "Comparacion de los 4 mecanismos de routing operando sobre "
            "el mismo backbone ViT congelado y los mismos CLS tokens."
        )

        ablation_html = """
        <table class="ablation-table">
            <thead>
                <tr>
                    <th>Router</th>
                    <th>Tipo</th>
                    <th>Routing Acc.</th>
                    <th>Latencia</th>
                    <th>Parametros</th>
                    <th>Gradiente</th>
                    <th>VRAM</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>ViT + Linear</strong></td>
                    <td>Param. (gradiente)</td>
                    <td style="color: #2ea043;">~85-92%</td>
                    <td>~8 ms</td>
                    <td>~5.7M</td>
                    <td>Si</td>
                    <td>~1.5 GB</td>
                </tr>
                <tr>
                    <td><strong>ViT + GMM</strong></td>
                    <td>Param. (EM)</td>
                    <td style="color: #d29922;">~78-87%</td>
                    <td>~3 ms</td>
                    <td>5x(d+d&sup2;)</td>
                    <td>No</td>
                    <td>~50 MB</td>
                </tr>
                <tr>
                    <td><strong>ViT + Naive Bayes</strong></td>
                    <td>Param. (MLE)</td>
                    <td style="color: #d29922;">~72-82%</td>
                    <td>~1 ms</td>
                    <td>5x2d</td>
                    <td>No</td>
                    <td>~5 MB</td>
                </tr>
                <tr>
                    <td><strong>ViT + k-NN (FAISS)</strong></td>
                    <td>No param.</td>
                    <td style="color: #d29922;">~75-85%</td>
                    <td>~5 ms</td>
                    <td>Nxd (train)</td>
                    <td>No</td>
                    <td>~85 MB</td>
                </tr>
            </tbody>
        </table>
        <p style="color: #8b949e; font-size: 0.78rem; margin-top: 0.8rem;">
            Nota: Los valores mostrados son rangos esperados de la literatura.
            Se actualizaran con resultados reales del experimento.
        </p>
        """
        st.markdown(ablation_html, unsafe_allow_html=True)

    # ===== TAB 2: Load Balance (#21) =====
    with tab_balance:
        st.markdown("### Load Balance — Distribucion de Carga entre Expertos")

        total = st.session_state.load_counts.sum()

        if total > 0:
            fractions = st.session_state.load_counts / total

            # Calcular cociente max/min para la penalizacion
            nonzero = fractions[fractions > 0]
            if len(nonzero) > 1:
                ratio = nonzero.max() / nonzero.min()
            else:
                ratio = 1.0

            # Indicador del cociente (regla: max(fi)/min(fi) < 1.30)
            ratio_color = "#2ea043" if ratio < 1.30 else "#f85149"
            st.markdown(
                f'<div class="metric-card" style="text-align: center;">'
                f'<h4>Cociente max(f_i) / min(f_i)</h4>'
                f'<p class="value" style="color: {ratio_color}; '
                f'font-size: 2rem;">{ratio:.2f}</p>'
                f'<p class="sub">Umbral de penalizacion: 1.30 '
                f'({"OK" if ratio < 1.30 else "PENALIZACION -40%"})</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Grafica de barras
            col_chart, col_table = st.columns([2, 1])

            with col_chart:
                df_balance = pd.DataFrame({
                    "Experto": [EXPERT_INFO[i]["dataset"] for i in range(5)],
                    "Fraccion (f_i)": fractions,
                    "Conteo": st.session_state.load_counts.astype(int),
                })
                st.bar_chart(
                    df_balance.set_index("Experto")["Fraccion (f_i)"],
                    use_container_width=True,
                )

            with col_table:
                st.dataframe(
                    df_balance.style.format({
                        "Fraccion (f_i)": "{:.3f}",
                    }),
                    hide_index=True,
                    use_container_width=True,
                )
        else:
            st.info(
                "No se han realizado inferencias aun. "
                "Sube una imagen para ver la distribucion de carga."
            )

    # ===== TAB 3: Historial =====
    with tab_history:
        st.markdown("### Historial de Inferencias")

        if st.session_state.inference_history:
            df_hist = pd.DataFrame(st.session_state.inference_history)
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.info("El historial esta vacio. Sube imagenes para poblar.")

else:
    # ======================================================================
    #  Estado vacio — pantalla de bienvenida
    # ======================================================================
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="metric-card">'
            '<h4>Datasets Soportados</h4>'
            '<p class="value">5</p>'
            '<p class="sub">NIH, ISIC, Osteo, LUNA16, Pancreas</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="metric-card">'
            '<h4>Expertos Heterogeneos</h4>'
            '<p class="value">5</p>'
            '<p class="sub">ConvNeXt, EfficientNet, VGG-16, '
            'R3D-18, Swin3D</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            '<div class="metric-card">'
            '<h4>Mecanismos de Routing</h4>'
            '<p class="value">4</p>'
            '<p class="sub">Linear, GMM, Naive Bayes, k-NN</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("""
    ### Como usar este Dashboard

    1. **Sube una imagen** en el panel lateral izquierdo (PNG, JPEG, o NIfTI).
    2. El sistema **detecta automaticamente** si es 2D o 3D.
    3. El **Router ViT** analiza la imagen y decide a que experto enviarla.
    4. El **Experto activado** realiza la clasificacion clinica.
    5. Observa el **Attention Heatmap** para entender donde mira el Router.
    6. Revisa el **Load Balance** para asegurar distribucion equitativa.

    > **Nota:** Este dashboard esta en modo DEMO con modelos simulados.
    > Cuando los checkpoints (.pth) esten listos, la inferencia sera real.
    """)
