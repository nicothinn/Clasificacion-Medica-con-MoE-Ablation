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
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
from moe_inference import MoEInferenceEngine
from mock_models import EXPERT_INFO


# ======================================================================
#  Configuracion de pagina
# ======================================================================
st.set_page_config(
    page_title="MoE Medical Vision Router",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================================
#  Utilidades de imagen
# ======================================================================
def pil_to_base64(img):
    """Convierte una imagen PIL a base64 para incrustar en HTML."""
    if img is None:
        return ""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ======================================================================
#  CSS global para Streamlit (overrides del framework)
# ======================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    header[data-testid="stHeader"] { background: #080d19; }
    section[data-testid="stSidebar"] {
        background: #080d19;
        border-right: 1px solid rgba(255,255,255,0.04);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #b0bdd0;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #5a6a80;
        border: none;
        border-bottom: 2px solid transparent;
        padding: 0.6rem 1.2rem;
        font-size: 0.8rem;
        font-weight: 500;
        border-radius: 0;
    }
    .stTabs [aria-selected="true"] {
        color: #e0e7f0;
        border-bottom: 2px solid #3b82f6;
        background: transparent;
    }
    div[data-testid="stImage"] > img {
        border-radius: 6px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ======================================================================
#  Plantilla CSS compartida para st.components.v1.html
# ======================================================================
_BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Inter', sans-serif;
    background: transparent;
    color: #c0cad8;
    -webkit-font-smoothing: antialiased;
}

/* --- Tokens --- */
:root {
    --bg-primary: #0a0f1e;
    --bg-card: #0e1528;
    --bg-card-hover: #111a30;
    --border: rgba(255,255,255,0.05);
    --text-primary: #e0e7f0;
    --text-secondary: #8896a8;
    --text-muted: #4e5d72;
    --accent: #3b82f6;
    --accent-soft: rgba(59,130,246,0.12);
    --green: #22c55e;
    --green-soft: rgba(34,197,94,0.10);
    --yellow: #eab308;
    --red: #ef4444;
    --red-soft: rgba(239,68,68,0.08);
    --radius: 8px;
}

/* --- Utility classes --- */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.1rem;
}
.section-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-muted);
    margin-bottom: 0.6rem;
}
.kpi-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.4px;
}
.kpi-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.1rem;
}
"""


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
    """Carga el motor MoE real una sola vez (persistente entre recargas)."""
    return MoEInferenceEngine(use_mock=False, repo_id="Lucu1232004p/Proyecto-MoE-Pesos")


# ======================================================================
#  Header (componente HTML puro)
# ======================================================================
components.html(f"""
<html><head><style>
{_BASE_CSS}
.header {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.1rem 1.4rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
}}
.header .icon-box {{
    width: 40px; height: 40px;
    border-radius: 10px;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}}
.header .icon-box svg {{ width: 22px; height: 22px; }}
.header h1 {{
    font-size: 1.2rem; font-weight: 700;
    color: var(--text-primary); letter-spacing: -0.3px;
}}
.header p {{
    font-size: 0.78rem; color: var(--text-muted);
    margin-top: 0.1rem;
}}
.header .tag {{
    margin-left: auto;
    font-size: 0.65rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.8px;
    color: var(--green);
    background: var(--green-soft);
    padding: 0.3rem 0.7rem;
    border-radius: 4px;
}}
</style></head><body>
<div class="header">
    <div class="icon-box">
        <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2L2 7l10 5 10-5-10-5z"/>
            <path d="M2 17l10 5 10-5"/>
            <path d="M2 12l10 5 10-5"/>
        </svg>
    </div>
    <div>
        <h1>MoE Medical Vision Router</h1>
        <p>Sistema de Clasificacion Medica con Mixture of Experts</p>
    </div>
    <div class="tag">Dashboard v1.0</div>
</div>
</body></html>
""", height=82)


# ======================================================================
#  Sidebar — Carga de archivo (#15)
# ======================================================================
with st.sidebar:
    engine = get_engine()

    st.markdown("### Imagen Medica")
    st.markdown(
        "Sube una imagen **PNG/JPEG** (2D) o un volumen **NIfTI** (3D). "
        "El sistema detecta automaticamente el tipo."
    )

    uploaded_file = st.file_uploader(
        "Arrastra o selecciona un archivo",
        type=["png", "jpg", "jpeg", "nii", "gz", "mha"],
        help="Formatos: PNG, JPEG, NIfTI (.nii, .nii.gz), MHA",
    )

    st.markdown("---")

    # OOD threshold — hardcoded optimal value
    OOD_THRESHOLD = 85

    st.markdown("### Sesion")
    st.metric("Inferencias realizadas", st.session_state.total_inferences)

    if st.button("Reiniciar contadores"):
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

    # Actualizar contadores (#21)
    st.session_state.load_counts[result.expert_id] += 1
    st.session_state.total_inferences += 1
    st.session_state.inference_history.append({
        "Archivo": uploaded_file.name,
        "Experto": result.expert_name,
        "Prediccion": result.class_label,
        "Confianza": f"{result.confidence*100:.1f}%",
        "OOD": "Si" if result.is_ood else "No",
    })

    # ------- OOD Alert (#22) -------
    if result.is_ood:
        components.html(f"""
        <html><head><style>
        {_BASE_CSS}
        .ood {{
            background: var(--red-soft);
            border: 1px solid rgba(239,68,68,0.25);
            border-radius: var(--radius);
            padding: 0.85rem 1.1rem;
            display: flex; align-items: center; gap: 0.8rem;
        }}
        .ood .icon {{
            width: 32px; height: 32px; border-radius: 6px;
            background: rgba(239,68,68,0.15);
            display: flex; align-items: center; justify-content: center;
            flex-shrink: 0;
        }}
        .ood .icon svg {{ width: 18px; height: 18px; }}
        .ood .title {{ font-size: 0.82rem; font-weight: 600; color: #fca5a5; }}
        .ood .detail {{ font-size: 0.72rem; color: #9ca3af; margin-top: 0.1rem; }}
        </style></head><body>
        <div class="ood">
            <div class="icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="#f87171" stroke-width="2">
                    <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86
                             a2 2 0 00-3.42 0z"/>
                    <line x1="12" y1="9" x2="12" y2="13"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
            </div>
            <div>
                <div class="title">OOD &mdash; Imagen fuera de distribucion</div>
                <div class="detail">Entropia: {result.entropy:.3f} nats
                    &middot; Umbral: {result.ood_threshold:.3f}</div>
            </div>
        </div>
        </body></html>
        """, height=66)

    # ------- Layout: 3 columnas -------
    col_left, col_center, col_right = st.columns([1.2, 1.5, 1.3])

    # ===== COLUMNA IZQUIERDA: Imagen + Preprocesado (#15, #16) =====
    with col_left:
        # Imagen original
        if result.display_image is not None:
            st.image(result.display_image, use_container_width=True,
                     caption="Imagen original")

        # KPIs: deteccion + timing
        dim_type = "3D — Volumen CT" if result.is_3d else "2D — Imagen plana"
        components.html(f"""
        <html><head><style>
        {_BASE_CSS}
        .stack {{ display: flex; flex-direction: column; gap: 0.5rem; }}
        .card {{ padding: 0.85rem 1rem; }}
        </style></head><body>
        <div class="stack">
            <div class="card">
                <div class="section-label">Deteccion Automatica</div>
                <div class="kpi-value">{dim_type}</div>
                <div class="kpi-sub">Original: {result.original_shape}</div>
                <div class="kpi-sub">Adaptado: {result.processed_shape}</div>
            </div>
            <div class="card">
                <div class="section-label">Latencia de Inferencia</div>
                <div class="kpi-value">{result.total_ms:.1f} <span style="font-size:0.75rem;
                    font-weight:400;color:var(--text-muted)">ms</span></div>
                <div style="display:flex; gap:1rem; margin-top:0.4rem;">
                    <div>
                        <div class="kpi-sub">Preprocesado</div>
                        <div style="font-size:0.85rem; font-weight:600; color:var(--text-secondary)">
                            {result.preprocess_ms:.1f} ms</div>
                    </div>
                    <div>
                        <div class="kpi-sub">Router</div>
                        <div style="font-size:0.85rem; font-weight:600; color:var(--text-secondary)">
                            {result.router_ms:.1f} ms</div>
                    </div>
                    <div>
                        <div class="kpi-sub">Experto</div>
                        <div style="font-size:0.85rem; font-weight:600; color:var(--text-secondary)">
                            {result.expert_ms:.1f} ms</div>
                    </div>
                </div>
            </div>
        </div>
        </body></html>
        """, height=215)

    # ===== COLUMNA CENTRAL: Heatmap + Gating Scores (#18) =====
    with col_center:
        # Heatmap
        if result.heatmap_image is not None:
            st.image(result.heatmap_image, use_container_width=True,
                     caption="Attention Heatmap — Router ViT")

        # Gating scores
        scores = result.gating_scores
        bars_html = ""
        for i in range(5):
            info = EXPERT_INFO[i]
            pct = scores[i] * 100
            is_active = i == result.expert_id
            bar_color = "var(--accent)" if is_active else "#1e293b"
            active_cls = "active" if is_active else ""
            dot = '<span class="dot"></span>' if is_active else ""

            bars_html += f"""
            <div class="row {active_cls}">
                <span class="label">{dot}E{i} {info["dataset"]}</span>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct}%;background:{bar_color}"></div>
                </div>
                <span class="val">{pct:.1f}%</span>
            </div>"""

        components.html(f"""
        <html><head><style>
        {_BASE_CSS}
        .panel {{ padding: 0.9rem 1rem; }}
        .panel .section-label {{ margin-bottom: 0.7rem; }}
        .row {{
            display: flex; align-items: center; gap: 0.5rem;
            margin-bottom: 0.35rem;
        }}
        .row .label {{
            font-size: 0.7rem; color: var(--text-muted);
            width: 145px; flex-shrink: 0;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            display: flex; align-items: center; gap: 5px;
        }}
        .row .bar-track {{
            flex: 1; height: 7px; background: #111827;
            border-radius: 3px; overflow: hidden;
        }}
        .row .bar-fill {{
            height: 100%; border-radius: 3px;
            transition: width 0.4s ease;
        }}
        .row .val {{
            font-size: 0.7rem; color: var(--text-muted);
            width: 42px; text-align: right; flex-shrink: 0;
            font-variant-numeric: tabular-nums;
        }}
        .row.active .label {{ color: #93c5fd; font-weight: 600; }}
        .row.active .val {{ color: #93c5fd; font-weight: 600; }}
        .dot {{
            display: inline-block; width: 5px; height: 5px;
            border-radius: 50%; background: var(--accent);
        }}
        </style></head><body>
        <div class="card panel">
            <div class="section-label">Gating Scores del Router</div>
            {bars_html}
        </div>
        </body></html>
        """, height=190)

    # ===== COLUMNA DERECHA: Experto + Prediccion (#17, #19) =====
    with col_right:
        # Prediccion + Confianza + Panel experto en un solo componente HTML
        conf_pct = result.confidence * 100
        if conf_pct > 80:
            conf_color = "var(--green)"
            conf_border = "rgba(34,197,94,0.25)"
        elif conf_pct > 50:
            conf_color = "var(--yellow)"
            conf_border = "rgba(234,179,8,0.25)"
        else:
            conf_color = "var(--red)"
            conf_border = "rgba(239,68,68,0.25)"

        if result.is_ood:
            badge_border = "var(--yellow)"
            badge_label = f"{result.class_label} (baja confianza)"
        else:
            badge_border = "var(--green)"
            badge_label = result.class_label

        # Filas del panel de experto
        expert_rows = f"""
        <div class="detail-row">
            <span class="dlabel">Arquitectura</span>
            <span class="dvalue">{result.expert_arch}</span>
        </div>
        <div class="detail-row">
            <span class="dlabel">Dataset</span>
            <span class="dvalue">{result.expert_dataset}</span>
        </div>
        <div class="detail-row">
            <span class="dlabel">Gating Score</span>
            <span class="dvalue">{result.gating_scores[result.expert_id]*100:.1f}%</span>
        </div>
        <div class="detail-row">
            <span class="dlabel">Entrada</span>
            <span class="dvalue">{"Volumen 3D" if result.is_3d else "Imagen 2D"}</span>
        </div>
        <div class="detail-row last">
            <span class="dlabel">Clases</span>
            <span class="dvalue">{len(result.class_names)}</span>
        </div>
        """

        components.html(f"""
        <html><head><style>
        {_BASE_CSS}
        .stack {{ display: flex; flex-direction: column; gap: 0.5rem; }}

        /* Prediction badge */
        .badge {{
            text-align: center; padding: 0.7rem;
            border: 1px solid {badge_border};
            border-radius: var(--radius);
            background: var(--bg-card);
        }}
        .badge .pred {{
            font-size: 0.95rem; font-weight: 600;
            color: {badge_border};
        }}

        /* Confidence */
        .conf-card {{ padding: 0.85rem 1rem; }}
        .conf-value {{
            font-size: 1.5rem; font-weight: 700;
            color: {conf_color}; letter-spacing: -0.5px;
        }}
        .conf-track {{
            height: 6px; background: #111827;
            border-radius: 3px; overflow: hidden; margin-top: 0.35rem;
        }}
        .conf-fill {{
            height: 100%; border-radius: 3px;
            background: {conf_color}; width: {conf_pct}%;
        }}

        /* Expert panel */
        .expert {{ padding: 1rem 1.1rem; }}
        .expert-name {{
            font-size: 0.88rem; font-weight: 600;
            color: var(--text-primary);
            padding-bottom: 0.55rem; margin-bottom: 0.55rem;
            border-bottom: 1px solid var(--border);
        }}
        .detail-row {{
            display: flex; justify-content: space-between;
            padding: 0.3rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.025);
        }}
        .detail-row.last {{ border-bottom: none; }}
        .dlabel {{ font-size: 0.75rem; color: var(--text-muted); }}
        .dvalue {{ font-size: 0.75rem; color: var(--text-secondary); font-weight: 500; }}
        </style></head><body>
        <div class="stack">
            <div class="badge">
                <div class="pred">{badge_label}</div>
            </div>

            <div class="card conf-card">
                <div class="section-label">Confianza del Experto</div>
                <div class="conf-value">{conf_pct:.1f}%</div>
                <div class="conf-track"><div class="conf-fill"></div></div>
            </div>

            <div class="card expert">
                <div class="expert-name">{result.expert_name}</div>
                {expert_rows}
            </div>
        </div>
        </body></html>
        """, height=365)

        # Probabilidades por clase (Streamlit dataframe para interactividad)
        st.markdown(
            '<p style="font-size:0.65rem;font-weight:600;text-transform:uppercase;'
            'letter-spacing:1.2px;color:#4e5d72;margin:0.6rem 0 0.4rem 0;">'
            'Probabilidades por Clase</p>',
            unsafe_allow_html=True,
        )
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
        "Ablation Study", "Load Balance", "Historial"
    ])

    # ===== TAB 1: Ablation Study (#20) =====
    with tab_ablation:
        st.caption(
            "Comparacion de 4 mecanismos de routing sobre el mismo backbone "
            "ViT congelado y los mismos CLS tokens."
        )

        components.html(f"""
        <html><head><style>
        {_BASE_CSS}
        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            border-radius: var(--radius);
            overflow: hidden;
            border: 1px solid var(--border);
            font-size: 0.8rem;
        }}
        th {{
            background: #070c18;
            color: var(--text-muted);
            padding: 0.6rem 0.9rem;
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            background: var(--bg-card);
            color: var(--text-secondary);
            padding: 0.55rem 0.9rem;
            border-top: 1px solid var(--border);
        }}
        tr:hover td {{ background: var(--bg-card-hover); }}
        tr.best td {{ background: var(--green-soft); }}
        tr.best:hover td {{ background: rgba(34,197,94,0.14); }}
        .tag-best {{
            display: inline-block;
            background: var(--green-soft);
            color: var(--green);
            font-size: 0.6rem;
            padding: 0.12rem 0.4rem;
            border-radius: 3px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-left: 0.4rem;
            vertical-align: middle;
        }}
        .acc-hi {{ color: var(--green); font-weight: 600; }}
        .acc-lo {{ color: var(--red); }}
        .footer-note {{
            font-size: 0.72rem; color: var(--text-muted);
            margin-top: 0.6rem;
        }}
        .footer-note strong {{ color: var(--green); }}
        </style></head><body>
        <table>
            <thead>
                <tr>
                    <th>Router</th>
                    <th>Tipo</th>
                    <th>Routing Acc.</th>
                    <th>Balance Ratio</th>
                    <th>Gradiente</th>
                    <th>L_aux</th>
                </tr>
            </thead>
            <tbody>
                <tr class="best">
                    <td>
                        <strong>ViT + k-NN (FAISS)</strong>
                        <span class="tag-best">Best</span>
                    </td>
                    <td>No parametrico</td>
                    <td class="acc-hi">95.28%</td>
                    <td>1.93</td>
                    <td>No</td>
                    <td>No</td>
                </tr>
                <tr>
                    <td><strong>ViT + Naive Bayes</strong></td>
                    <td>Param. (MLE)</td>
                    <td class="acc-hi">81.57%</td>
                    <td>2.40</td>
                    <td>No</td>
                    <td>No</td>
                </tr>
                <tr>
                    <td><strong>ViT + Linear (DL)</strong></td>
                    <td>Param. (gradiente)</td>
                    <td class="acc-lo">34.16%</td>
                    <td>&gt;1M</td>
                    <td>Si</td>
                    <td>Si</td>
                </tr>
                <tr>
                    <td><strong>ViT + GMM</strong></td>
                    <td>Param. (EM)</td>
                    <td class="acc-lo">12.47%</td>
                    <td>9.85</td>
                    <td>No</td>
                    <td>No</td>
                </tr>
            </tbody>
        </table>
        <p class="footer-note">
            Router seleccionado: <strong>ViT + k-NN (FAISS)</strong>
            con 95.28% de Routing Accuracy.
        </p>
        </body></html>
        """, height=245)

    # ===== TAB 2: Load Balance (#21) =====
    with tab_balance:
        total = st.session_state.load_counts.sum()

        if total > 0:
            fractions = st.session_state.load_counts / total
            nonzero = fractions[fractions > 0]
            ratio = nonzero.max() / nonzero.min() if len(nonzero) > 1 else 1.0

            ratio_color = "var(--green)" if ratio < 1.30 else "var(--red)"
            status_text = "Dentro del umbral" if ratio < 1.30 else "Penalizacion -40%"

            # Barras de balance como HTML
            balance_bars = ""
            for i in range(5):
                f_val = fractions[i]
                cnt = int(st.session_state.load_counts[i])
                pct = f_val * 100
                is_max = f_val == fractions.max()
                bar_c = "var(--accent)" if not is_max else (
                    "var(--green)" if ratio < 1.30 else "var(--red)")

                balance_bars += f"""
                <div class="b-row">
                    <span class="b-label">{EXPERT_INFO[i]["dataset"]}</span>
                    <div class="b-track">
                        <div class="b-fill" style="width:{pct}%;background:{bar_c}"></div>
                    </div>
                    <span class="b-val">{f_val:.3f}</span>
                    <span class="b-cnt">({cnt})</span>
                </div>"""

            components.html(f"""
            <html><head><style>
            {_BASE_CSS}
            .wrapper {{ display: flex; flex-direction: column; gap: 0.6rem; }}
            .ratio-card {{
                text-align: center; padding: 1rem;
            }}
            .ratio-num {{
                font-size: 2rem; font-weight: 700;
                color: {ratio_color}; letter-spacing: -1px;
            }}
            .ratio-sub {{ font-size: 0.72rem; color: var(--text-muted); margin-top: 0.2rem; }}
            .bars-card {{ padding: 1rem 1.1rem; }}
            .b-row {{
                display: flex; align-items: center; gap: 0.5rem;
                margin-bottom: 0.4rem;
            }}
            .b-label {{
                font-size: 0.72rem; color: var(--text-secondary);
                width: 150px; flex-shrink: 0;
            }}
            .b-track {{
                flex: 1; height: 8px; background: #111827;
                border-radius: 4px; overflow: hidden;
            }}
            .b-fill {{ height: 100%; border-radius: 4px; }}
            .b-val {{
                font-size: 0.72rem; color: var(--text-secondary);
                font-weight: 600; width: 45px; text-align: right;
                font-variant-numeric: tabular-nums;
            }}
            .b-cnt {{
                font-size: 0.65rem; color: var(--text-muted);
                width: 30px;
            }}
            </style></head><body>
            <div class="wrapper">
                <div class="card ratio-card">
                    <div class="section-label">Cociente max(f_i) / min(f_i)</div>
                    <div class="ratio-num">{ratio:.2f}</div>
                    <div class="ratio-sub">Umbral: 1.30 &middot; {status_text}</div>
                </div>
                <div class="card bars-card">
                    <div class="section-label">Distribucion de Carga</div>
                    {balance_bars}
                </div>
            </div>
            </body></html>
            """, height=305)
        else:
            st.info(
                "No se han realizado inferencias aun. "
                "Sube una imagen para ver la distribucion de carga."
            )

    # ===== TAB 3: Historial =====
    with tab_history:
        if st.session_state.inference_history:
            df_hist = pd.DataFrame(st.session_state.inference_history)
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.info("El historial esta vacio. Sube imagenes para comenzar.")

else:
    # ======================================================================
    #  Estado vacio — pantalla de bienvenida
    # ======================================================================

    components.html(f"""
    <html><head><style>
    {_BASE_CSS}
    .grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.6rem;
        margin-bottom: 0.8rem;
    }}
    .stat-card {{
        padding: 1.2rem;
    }}
    .stat-number {{
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: -1.5px;
        line-height: 1;
    }}
    .stat-label {{
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--text-muted);
        margin-bottom: 0.3rem;
    }}
    .stat-detail {{
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 0.45rem;
        line-height: 1.5;
    }}

    /* Instructions */
    .instructions {{
        padding: 1.3rem 1.5rem;
    }}
    .instructions h3 {{
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.8rem;
    }}
    .step-list {{
        list-style: none;
        padding: 0;
        counter-reset: steps;
    }}
    .step-list li {{
        counter-increment: steps;
        display: flex;
        align-items: flex-start;
        gap: 0.7rem;
        margin-bottom: 0.55rem;
        font-size: 0.8rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }}
    .step-list li::before {{
        content: counter(steps);
        display: flex;
        align-items: center;
        justify-content: center;
        width: 22px;
        height: 22px;
        border-radius: 6px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.68rem;
        font-weight: 700;
        flex-shrink: 0;
        margin-top: 1px;
    }}
    .step-list li strong {{
        color: var(--text-primary);
        font-weight: 600;
    }}
    .note {{
        font-size: 0.75rem;
        color: var(--text-muted);
        border-top: 1px solid var(--border);
        padding-top: 0.7rem;
        margin-top: 0.8rem;
        font-style: italic;
    }}
    </style></head><body>

    <div class="grid">
        <div class="card stat-card">
            <div class="stat-label">Datasets Soportados</div>
            <div class="stat-number">5</div>
            <div class="stat-detail">NIH ChestX-ray, ISIC 2019,
                Knee Osteoarthritis, LUNA16, Pancreas CT</div>
        </div>
        <div class="card stat-card">
            <div class="stat-label">Expertos Heterogeneos</div>
            <div class="stat-number">5</div>
            <div class="stat-detail">LungMaxViT, EfficientNet-B3,
                VGG-16 BN, DCSwinB-3D, R3D-18</div>
        </div>
        <div class="card stat-card">
            <div class="stat-label">Mecanismos de Routing</div>
            <div class="stat-number">4</div>
            <div class="stat-detail">Linear (DL), GMM,
                Naive Bayes, k-NN (FAISS)</div>
        </div>
    </div>

    <div class="card instructions">
        <h3>Guia de Uso</h3>
        <ol class="step-list">
            <li><strong>Carga una imagen</strong> desde el panel lateral (PNG, JPEG o NIfTI).</li>
            <li>El sistema <strong>detecta automaticamente</strong> si la entrada es 2D o 3D.</li>
            <li>El <strong>Router ViT</strong> analiza la imagen y selecciona el experto adecuado.</li>
            <li>El <strong>experto activado</strong> realiza la clasificacion clinica.</li>
            <li>Revisa el <strong>Attention Heatmap</strong> para visualizar las regiones de interes.</li>
            <li>Consulta el <strong>Load Balance</strong> para verificar la distribucion equitativa.</li>
        </ol>
        <p class="note">
            Este dashboard conecta directamente con el repositorio de
            Hugging Face para cargar los modelos reales del sistema MoE.
        </p>
    </div>

    </body></html>
    """, height=440)
