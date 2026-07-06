"""
PomologyAI — Date Fruit Variety Classifier
Architecture: Linear(34→64) → ReLU → Linear(64→64) → ReLU → Linear(64→7)
Dataset: DateFruit (7 varieties, 34 morphological features)
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
import gradio as gr

# ── 1. Architecture (mirrors notebook exactly) ────────────────────────────────
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(34, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
        )

    def forward(self, x):
        return self.model(x)

# ── 2. Load weights + preprocessors ──────────────────────────────────────────
model = ANN()
model.load_state_dict(torch.load("pomology_model.pth", map_location="cpu"))
model.eval()

scaler = joblib.load("scaler.pkl")         # StandardScaler (fitted on X_train)
le     = joblib.load("label_encoder.pkl")  # LabelEncoder  (fitted on y)

CLASS_NAMES = list(le.classes_)
# Expected: ['BERHI','DEGLET','DOKOL','IRAQI','ROTAB','SAFAWI','SOGAY']

# ── 3. Inference function ─────────────────────────────────────────────────────
def predict(*feature_values):
    arr        = np.array(feature_values, dtype=np.float32).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    tensor     = torch.tensor(arr_scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

# ── 4. Feature definitions  [name, unit, min, max, default] ──────────────────
FEATURES = [
    # Shape
    ("AREA",           "mm²",   0,      200000, 80000),
    ("PERIMETER",      "mm",    0,      2500,   900),
    ("MAJOR_AXIS",     "mm",    0,      700,    250),
    ("MINOR_AXIS",     "mm",    0,      500,    180),
    ("ECCENTRICITY",   "—",     0.0,    1.0,    0.7),
    ("EQDIASQ",        "mm",    0,      700,    300),
    ("SOLIDITY",       "—",     0.0,    1.0,    0.99),
    ("CONVEX_AREA",    "mm²",   0,      210000, 82000),
    ("EXTENT",         "—",     0.0,    1.0,    0.75),
    ("ASPECT_RATIO",   "—",     0.0,    5.0,    1.4),
    ("ROUNDNESS",      "—",     0.0,    1.0,    0.75),
    ("COMPACTNESS",    "—",     0.0,    1.0,    0.85),
    ("SHAPEFACTOR_1",  "—",     0.0,    0.01,   0.003),
    ("SHAPEFACTOR_2",  "—",     0.0,    0.01,   0.002),
    ("SHAPEFACTOR_3",  "—",     0.0,    1.0,    0.6),
    ("SHAPEFACTOR_4",  "—",     0.0,    1.0,    0.98),
    # Color stats R/G/B
    ("MeanRR",         "0-255", 0,      255,    120),
    ("MeanRG",         "0-255", 0,      255,    100),
    ("MeanRB",         "0-255", 0,      255,    80),
    ("StdDevRR",       "—",     0,      80,     20),
    ("StdDevRG",       "—",     0,      80,     18),
    ("StdDevRB",       "—",     0,      80,     15),
    ("SkewRR",         "—",    -5,      5,      0.5),
    ("SkewRG",         "—",    -5,      5,      0.3),
    ("SkewRB",         "—",    -5,      5,      0.2),
    ("KurtosisRR",     "—",    -3,      10,     1.0),
    ("KurtosisRG",     "—",    -3,      10,     0.8),
    ("KurtosisRB",     "—",    -3,      10,     0.6),
    ("EntropyRR",      "—",     0,      8,      5.5),
    ("EntropyRG",      "—",     0,      8,      5.2),
    ("EntropyRB",      "—",     0,      8,      4.8),
    # Wavelet (Daub4)
    ("ALLdaub4RR",     "—",     0,      20,     7.0),
    ("ALLdaub4RG",     "—",     0,      20,     6.0),
    ("ALLdaub4RB",     "—",     0,      20,     5.0),
]

# ── 5. Build Gradio components ────────────────────────────────────────────────
inputs = [
    gr.Slider(
        minimum=lo,
        maximum=hi,
        value=default,
        label=f"{name} [{unit}]" if unit != "—" else name,
        step=round((hi - lo) / 1000, 6),
    )
    for name, unit, lo, hi, default in FEATURES
]

# ── 6. Theme — dark amber: deliberate, not default ───────────────────────────
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#0e0d0b",
    body_background_fill_dark="#0e0d0b",
    block_background_fill="#17150f",
    block_background_fill_dark="#17150f",
    block_border_color="#2a2720",
    block_border_color_dark="#2a2720",
    block_label_text_color="#8a7d66",
    block_label_text_color_dark="#8a7d66",
    input_background_fill="#1f1d17",
    input_background_fill_dark="#1f1d17",
    input_border_color="#36322a",
    input_border_color_dark="#36322a",
    button_primary_background_fill="#c97b0a",
    button_primary_background_fill_hover="#a86208",
    button_primary_text_color="#ffffff",
    slider_color="#c97b0a",
    body_text_color="#d4c8b4",
    body_text_color_dark="#d4c8b4",
)

CSS = """
/* ── Page shell ── */
.gradio-container { max-width: 1200px !important; margin: 0 auto; }

/* ── Hero header ── */
.pg-header {
    background: linear-gradient(135deg, #18140a 0%, #231a07 60%, #1a1208 100%);
    border: 1px solid #3d3118;
    border-radius: 14px;
    padding: 30px 36px 26px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.pg-header::before {
    content: '🌴';
    position: absolute;
    right: 36px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.08;
}
.pg-header h1 {
    font-size: 1.9rem;
    font-weight: 800;
    color: #f0b429;
    letter-spacing: -0.6px;
    margin: 0 0 4px 0;
    line-height: 1.1;
}
.pg-header .subtitle {
    color: #7a6e58;
    font-size: 0.87rem;
    margin: 0 0 14px 0;
}
.pill {
    display: inline-block;
    background: #231a07;
    border: 1px solid #5a4012;
    color: #c97b0a;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 12px;
    border-radius: 999px;
    margin-right: 8px;
}

/* ── Section eyebrows ── */
.eyebrow {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5a4f3a;
    border-left: 3px solid #c97b0a;
    padding-left: 10px;
    margin: 20px 0 12px;
}

/* ── Predict button ── */
.predict-btn {
    margin-top: 20px !important;
    height: 52px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    border-radius: 10px !important;
}

/* ── Output panel ── */
.output-panel {
    background: #17150f;
    border: 1px solid #2a2720;
    border-radius: 12px;
    padding: 20px;
    position: sticky;
    top: 20px;
}

/* ── Footer ── */
.pg-footer {
    text-align: center;
    color: #3d3828;
    font-size: 0.76rem;
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid #1e1b14;
}
.pg-footer strong { color: #c97b0a; }
"""

HEADER = """
<div class="pg-header">
  <h1>PomologyAI</h1>
  <p class="subtitle">Date Fruit Variety Classifier — Artificial Neural Network</p>
  <span class="pill">34 Morphological Features</span>
  <span class="pill">7 Date Varieties</span>
  <span class="pill">PyTorch ANN</span>
</div>
"""

FOOTER = """
<div class="pg-footer">
  Built by <strong>Karthika Krishna M</strong> &nbsp;·&nbsp;
  ANN: Linear(34→64→64→7) &nbsp;·&nbsp;
  Trained on DateFruit Dataset &nbsp;·&nbsp;
  
</div>
"""

# ── 7. Layout ─────────────────────────────────────────────────────────────────
with gr.Blocks(theme=theme, css=CSS, title="PomologyAI — Date Fruit Classifier") as demo:

    gr.HTML(HEADER)

    with gr.Row(equal_height=False):
        with gr.Column(scale=2):

            gr.HTML('<div class="eyebrow">Shape &amp; Geometry</div>')
            with gr.Row():
                gr.Slider(minimum=0,      maximum=200000, value=80000,  label="AREA [mm²]",         step=200)
                gr.Slider(minimum=0,      maximum=2500,   value=900,    label="PERIMETER [mm]",      step=2.5)
                gr.Slider(minimum=0,      maximum=700,    value=250,    label="MAJOR_AXIS [mm]",     step=0.7)
                gr.Slider(minimum=0,      maximum=500,    value=180,    label="MINOR_AXIS [mm]",     step=0.5)
            with gr.Row():
                gr.Slider(minimum=0.0,    maximum=1.0,    value=0.7,    label="ECCENTRICITY",        step=0.001)
                gr.Slider(minimum=0,      maximum=700,    value=300,    label="EQDIASQ [mm]",        step=0.7)
                gr.Slider(minimum=0.0,    maximum=1.0,    value=0.99,   label="SOLIDITY",            step=0.001)
                gr.Slider(minimum=0,      maximum=210000, value=82000,  label="CONVEX_AREA [mm²]",   step=210)
            with gr.Row():
                gr.Slider(minimum=0.0,    maximum=1.0,    value=0.75,   label="EXTENT",              step=0.001)
                gr.Slider(minimum=0.0,    maximum=5.0,    value=1.4,    label="ASPECT_RATIO",        step=0.005)
                gr.Slider(minimum=0.0,    maximum=1.0,    value=0.75,   label="ROUNDNESS",           step=0.001)
                gr.Slider(minimum=0.0,    maximum=1.0,    value=0.85,   label="COMPACTNESS",         step=0.001)
            with gr.Row():
                gr.Slider(minimum=0.0,    maximum=0.01,   value=0.003,  label="SHAPEFACTOR_1",       step=0.00001)
                gr.Slider(minimum=0.0,    maximum=0.01,   value=0.002,  label="SHAPEFACTOR_2",       step=0.00001)
                gr.Slider(minimum=0.0,    maximum=1.0,    value=0.6,    label="SHAPEFACTOR_3",       step=0.001)
                gr.Slider(minimum=0.0,    maximum=1.0,    value=0.98,   label="SHAPEFACTOR_4",       step=0.001)

            gr.HTML('<div class="eyebrow">Colour Statistics — R / G / B</div>')
            with gr.Row():
                gr.Slider(minimum=0, maximum=255, value=120, label="MeanRR [0-255]",   step=0.255)
                gr.Slider(minimum=0, maximum=255, value=100, label="MeanRG [0-255]",   step=0.255)
                gr.Slider(minimum=0, maximum=255, value=80,  label="MeanRB [0-255]",   step=0.255)
            with gr.Row():
                gr.Slider(minimum=0, maximum=80,  value=20,  label="StdDevRR",         step=0.08)
                gr.Slider(minimum=0, maximum=80,  value=18,  label="StdDevRG",         step=0.08)
                gr.Slider(minimum=0, maximum=80,  value=15,  label="StdDevRB",         step=0.08)
            with gr.Row():
                gr.Slider(minimum=-5, maximum=5,  value=0.5, label="SkewRR",           step=0.01)
                gr.Slider(minimum=-5, maximum=5,  value=0.3, label="SkewRG",           step=0.01)
                gr.Slider(minimum=-5, maximum=5,  value=0.2, label="SkewRB",           step=0.01)
            with gr.Row():
                gr.Slider(minimum=-3, maximum=10, value=1.0, label="KurtosisRR",       step=0.013)
                gr.Slider(minimum=-3, maximum=10, value=0.8, label="KurtosisRG",       step=0.013)
                gr.Slider(minimum=-3, maximum=10, value=0.6, label="KurtosisRB",       step=0.013)
            with gr.Row():
                gr.Slider(minimum=0, maximum=8,   value=5.5, label="EntropyRR",        step=0.008)
                gr.Slider(minimum=0, maximum=8,   value=5.2, label="EntropyRG",        step=0.008)
                gr.Slider(minimum=0, maximum=8,   value=4.8, label="EntropyRB",        step=0.008)

            gr.HTML('<div class="eyebrow">Wavelet Descriptors (Daub4)</div>')
            with gr.Row():
                gr.Slider(minimum=0, maximum=20, value=7.0, label="ALLdaub4RR",        step=0.02)
                gr.Slider(minimum=0, maximum=20, value=6.0, label="ALLdaub4RG",        step=0.02)
                gr.Slider(minimum=0, maximum=20, value=5.0, label="ALLdaub4RB",        step=0.02)

            btn = gr.Button("Classify Variety", variant="primary", elem_classes=["predict-btn"])

        with gr.Column(scale=1):
            gr.HTML('<div class="eyebrow">Prediction</div>')
            output = gr.Label(num_top_classes=7, label="Date Fruit Variety · Confidence")
            gr.Markdown("**7 Varieties**\n\nBERHI · DEGLET · DOKOL\nIRAQI · ROTAB · SAFAWI · SOGAY", container=False)

    all_sliders = [c for c in demo.blocks.values() if isinstance(c, gr.Slider)]
    # demo.blocks preserves insertion order (Python 3.7+), so sliders are
    # in the exact order they were created: shape[0..15], colour[16..30], wavelet[31..33]
    btn.click(fn=predict, inputs=all_sliders, outputs=output)
    gr.HTML(FOOTER)

    

if __name__ == "__main__":
    demo.launch()