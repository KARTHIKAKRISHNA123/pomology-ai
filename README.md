
<div align="center">

# 🌴 PomologyAI
### Date Fruit Variety Classifier · Artificial Neural Network

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-FF7C00?logo=gradio&logoColor=white)](https://gradio.app)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Spaces-Deployed-FFD21E)](https://huggingface.co/spaces/KARTHIKAKRISHNA123/PomologyAI)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A production-grade ANN inference pipeline that classifies date fruit varieties from 34 morphological and colour-statistical features — deployed as a zero-dependency web app on Hugging Face Spaces.**

</div>

---

## 📌 Problem Statement

Date fruit grading is traditionally a manual, labour-intensive agricultural process subject to human error and inconsistency. Automated morphological classification enables precision agriculture, improves supply-chain traceability, and reduces post-harvest losses.

**This system answers:** *Given 34 image-derived shape and colour features of a date fruit sample, which of the 7 commercial varieties does it belong to — and with what confidence?*

---

## 🎯 Solution Overview

PomologyAI wraps a trained PyTorch ANN inside a Gradio Blocks interface. Users adjust 34 sliders corresponding to morphological measurements (area, perimeter, eccentricity, RGB statistics, wavelet descriptors) and receive a real-time confidence distribution across 7 date varieties.

**No image upload required** — the system works on pre-extracted feature vectors, making inference instantaneous and compute-minimal.

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| **7-class classifier** | BERHI, DEGLET, DOKOL, IRAQI, ROTAB, SAFAWI, SOGAY |
| **34 input features** | Shape geometry + RGB channel statistics + Daub4 wavelet descriptors |
| **Confidence scores** | Softmax probabilities for all 7 classes displayed as a ranked label |
| **Live inference** | Sub-100ms inference on CPU — no GPU required |
| **Dark amber UI** | Intentional palm-gold design language; grouped feature sections |
| **Production artifact** | Portable `.pth` weights + `scaler.pkl` + `label_encoder.pkl` |

---

## 🏗️ Overall Architecture

```mermaid
flowchart LR
    subgraph Input["Input Layer"]
        U["👤 User\n(Gradio Sliders)"]
    end

    subgraph Preprocessing["Preprocessing"]
        SC["StandardScaler\nscaler.pkl"]
    end

    subgraph Model["ANN Model\npomology_model.pth"]
        L1["Linear 34→64\n+ ReLU"]
        L2["Linear 64→64\n+ ReLU"]
        L3["Linear 64→7"]
        SF["Softmax"]
        L1 --> L2 --> L3 --> SF
    end

    subgraph Postprocessing["Post-processing"]
        LE["LabelEncoder\nlabel_encoder.pkl"]
    end

    subgraph Output["Output"]
        R["Confidence\nDistribution\n(7 classes)"]
    end

    U -->|"34 float values"| SC
    SC -->|"standardized tensor [1×34]"| L1
    SF -->|"prob vector [1×7]"| LE
    LE --> R
```

---

## 🧠 System Architecture

```mermaid
flowchart TD
    subgraph HFSpace["Hugging Face Space (Gradio SDK)"]
        APP["app.py\n(Entrypoint)"]
        
        subgraph Artifacts["Serialized Artifacts"]
            PTH["pomology_model.pth\nModel Weights"]
            SCLR["scaler.pkl\nStandardScaler"]
            LBL["label_encoder.pkl\nLabelEncoder"]
        end

        subgraph GradioBlocks["Gradio Blocks UI"]
            HDR["HTML Header"]
            S1["Sliders — Shape (16)"]
            S2["Sliders — Colour (15)"]
            S3["Sliders — Wavelet (3)"]
            BTN["Classify Button"]
            OUT["gr.Label Output\n(7 classes × confidence)"]
        end

        subgraph Inference["Inference Pipeline"]
            NP["numpy.array reshape (1×34)"]
            TF["scaler.transform()"]
            TEN["torch.tensor float32"]
            FWD["model.forward()"]
            SM["torch.softmax()"]
            MAP["dict {class: prob}"]
        end
    end

    APP --> Artifacts
    APP --> GradioBlocks
    BTN -->|"on.click()"| NP
    NP --> TF --> TEN --> FWD --> SM --> MAP --> OUT
```

---

## 🧰 Technology Stack — Complete Breakdown

| Technology | Version | Category | Purpose in Project | Why Chosen | Key Features Used |
|---|---|---|---|---|---|
| **PyTorch** | 2.x | Deep Learning | ANN forward pass, weight serialization/loading | Dynamic computation graph, `state_dict` portability, `map_location='cpu'` for HF Spaces | `nn.Sequential`, `nn.Linear`, `nn.ReLU`, `model.eval()`, `torch.no_grad()`, `torch.softmax()`, `torch.save()`, `torch.load()` |
| **torch.nn** | — | Model API | Defines ANN architecture layers | Native PyTorch module; Sequential wrapping enables clean serialization | `nn.Module`, `nn.Sequential`, `nn.Linear`, `nn.ReLU` |
| **scikit-learn** | 1.x | Preprocessing | Feature standardization and label encoding | Industry-standard scalers; `joblib`-serializable for production reuse | `StandardScaler.fit_transform()`, `StandardScaler.transform()`, `LabelEncoder.fit_transform()`, `le.classes_` |
| **joblib** | — | Serialization | Persist `scaler.pkl` and `label_encoder.pkl` | Efficient numpy-array serialization, preferred over `pickle` for sklearn objects | `joblib.dump()`, `joblib.load()` |
| **NumPy** | 1.x | Numerical | Feature vector construction before tensor conversion | Bridge between Python list → sklearn scaler → PyTorch tensor | `np.array()`, `dtype=np.float32`, `.reshape(1, -1)` |
| **Gradio** | 4.x | UI / Serving | Auto-serves Gradio Blocks as HF Space web app | Native HF Spaces SDK; `gr.Blocks` enables custom layouts | `gr.Blocks`, `gr.Slider`, `gr.Label`, `gr.Button`, `gr.Row`, `gr.Column`, `gr.HTML`, `gr.Markdown`, `gr.themes.Base`, CSS injection |
| **Gradio Themes** | — | Design | Dark amber colour scheme via `.set()` token overrides | Subject-specific visual identity; avoids default grey | `gr.themes.Base`, `gr.themes.colors.amber`, `gr.themes.GoogleFont` |

---

## 🔄 Request Lifecycle

### Inference Request — User Submits Feature Vector

```
1. USER INTERACTION
   └── User adjusts 34 Gradio sliders → clicks "🔍 Classify Variety"
       → Gradio Blocks: btn.click(fn=predict, inputs=inputs, outputs=output)

2. PYTHON FUNCTION CALL
   └── predict(*feature_values) is invoked with 34 float arguments
       → args unpacked from slider components

3. PREPROCESSING
   └── np.array(feature_values, dtype=np.float32).reshape(1, -1)
       → shape: (1, 34)
       → scaler.transform(arr)  [StandardScaler loaded from scaler.pkl]
       → output shape: (1, 34) — zero-mean, unit-variance

4. TENSOR CONVERSION
   └── torch.tensor(arr_scaled, dtype=torch.float32)
       → shape: [1, 34]

5. FORWARD PASS
   └── model.eval() + torch.no_grad()
       → Linear(34→64) + ReLU
       → Linear(64→64) + ReLU
       → Linear(64→7)           [raw logits, shape: [1, 7]]

6. POSTPROCESSING
   └── torch.softmax(logits, dim=1).squeeze().numpy()
       → prob vector: [p0, p1, ..., p6], sum = 1.0
       → dict {CLASS_NAMES[i]: float(probs[i]) for i in range(7)}
       → CLASS_NAMES sourced from le.classes_ (LabelEncoder)

7. OUTPUT RENDER
   └── gr.Label displays top-3 confident classes with probability bars
       → Gradio auto-sorts by descending confidence
```

---

## 🌊 Data Flow Explanation

```
Training Data                              Inference Data
(DateFruit_Dataset.csv)                    (Gradio Sliders)
         │                                          │
         ▼                                          ▼
  DataFrame (X, y)                        feature_values: tuple[float]
         │                                          │
  StandardScaler.fit_transform(X_train)    np.array → reshape(1,34)
  LabelEncoder.fit_transform(y)                     │
         │                                  scaler.transform()
  joblib.dump(scaler)                               │
  joblib.dump(le)          ←──── reuse ────  standardized (1,34)
         │                                          │
  torch.tensor → TensorDataset                torch.tensor float32
         │                                          │
  DataLoader (batch=32)                      model.forward()   ← pomology_model.pth
         │                                          │
  ANN.forward() + CrossEntropyLoss           torch.softmax(dim=1)
         │                                          │
  Adam optimizer + 100 epochs               {class: probability}
         │                                          │
  torch.save(model.state_dict())             gr.Label renders output
```

---

<details>
<summary>📐 UML Diagram Suite — All 9 Diagrams</summary>

### 1. Use Case Diagram

```mermaid
graph TD
    U(["👤 User"])
    S(["🖥️ HF Space"])

    UC1["Adjust Feature Sliders"]
    UC2["Submit Classification Request"]
    UC3["View Confidence Distribution"]
    UC4["Interpret Top Prediction"]

    U --> UC1
    U --> UC2
    UC2 --> UC3
    UC3 --> UC4
    S --> UC2
    S --> UC3
```

### 2. Class Diagram

```mermaid
classDiagram
    class ANN {
        +model: nn.Sequential
        +__init__()
        +forward(x: Tensor) Tensor
    }

    class StandardScaler {
        +mean_: ndarray
        +scale_: ndarray
        +fit_transform(X) ndarray
        +transform(X) ndarray
    }

    class LabelEncoder {
        +classes_: ndarray
        +fit_transform(y) ndarray
    }

    class InferencePipeline {
        +model: ANN
        +scaler: StandardScaler
        +le: LabelEncoder
        +predict(*features) dict
    }

    class GradioUI {
        +inputs: list[gr.Slider]
        +output: gr.Label
        +btn: gr.Button
        +launch()
    }

    InferencePipeline --> ANN
    InferencePipeline --> StandardScaler
    InferencePipeline --> LabelEncoder
    GradioUI --> InferencePipeline
```

### 3. Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant G as Gradio UI
    participant P as predict()
    participant SC as StandardScaler
    participant M as ANN Model
    participant OUT as gr.Label

    U->>G: Adjust 34 sliders
    U->>G: Click "Classify"
    G->>P: predict(*34_floats)
    P->>P: np.array reshape(1,34)
    P->>SC: transform(arr)
    SC-->>P: scaled_arr (1,34)
    P->>P: torch.tensor(scaled_arr)
    P->>M: model.forward(tensor)
    M-->>P: logits [1,7]
    P->>P: softmax(logits)
    P-->>G: dict {class: prob}
    G-->>OUT: render ranked labels
    OUT-->>U: Confidence distribution
```

### 4. Activity Diagram

```mermaid
flowchart TD
    A([Start]) --> B[User opens HF Space]
    B --> C[Gradio UI loads with default slider values]
    C --> D[User adjusts sliders for their fruit sample]
    D --> E[Click Classify Variety]
    E --> F[predict function invoked]
    F --> G[NumPy array constructed]
    G --> H[StandardScaler transforms array]
    H --> I[Convert to float32 tensor]
    I --> J[ANN forward pass]
    J --> K[Softmax over 7 logits]
    K --> L{prob sum == 1.0?}
    L -->|Yes| M[Build class→prob dict]
    L -->|No| N([Error])
    M --> O[Gradio renders gr.Label]
    O --> P([User reads prediction])
```

### 5. Component Diagram

```mermaid
flowchart LR
    subgraph UI["Gradio Blocks UI"]
        CMP1A["Header Component\nHTML"]
        CMP2A["Shape Sliders\n16 × gr.Slider"]
        CMP3A["Colour Sliders\n15 × gr.Slider"]
        CMP4A["Wavelet Sliders\n3 × gr.Slider"]
        CMP5A["Classify Button\ngr.Button"]
        CMP6A["Label Output\ngr.Label"]
    end

    subgraph Artifacts["Serialized Artifacts"]
        CMP7A["pomology_model.pth"]
        CMP8A["scaler.pkl"]
        CMP9A["label_encoder.pkl"]
    end

    subgraph Pipeline["Inference Pipeline"]
        CMP10A["predict()"]
    end

    CMP5A -->|"click event"| CMP10A
    CMP10A --> CMP8A
    CMP10A --> CMP7A
    CMP10A --> CMP9A
    CMP10A --> CMP6A
```

### 6. Deployment Diagram

```mermaid
flowchart TD
    subgraph HF["Hugging Face Infrastructure"]
        subgraph Space["PomologyAI Space - CPU Runtime"]
            APP2["app.py\nGradio Blocks Server"]
            PTH2["pomology_model.pth"]
            SCLR2["scaler.pkl"]
            LBL2["label_encoder.pkl"]
            REQ2["requirements.txt"]
        end
        BUILD["HF Build System\nPip install + launch"]
    end

    subgraph Dev["Developer Machine"]
        NB["ANN_Classification.ipynb"]
        GITPUSH["git push"]
    end

    NB -->|"torch.save + joblib.dump"| Space
    GITPUSH -->|"CD trigger"| BUILD
    BUILD --> Space

    U2["👤 Browser"] -->|"HTTPS"| APP2
```

### 7. State Diagram

```mermaid
stateDiagram-v2
    [*] --> Idle: Space loads
    Idle --> Adjusting: User moves slider
    Adjusting --> Adjusting: More slider changes
    Adjusting --> Predicting: Click Classify
    Predicting --> Rendering: predict() returns dict
    Rendering --> Idle: gr.Label updated
    Predicting --> Error: Exception in pipeline
    Error --> Idle: Gradio shows error toast
```

### 8. Object Diagram

```mermaid
flowchart LR
    OBJ1["model: ANN\n──────────\nmodel.training = False\nparams loaded from .pth"]
    OBJ2["scaler: StandardScaler\n──────────\nmean_ = [μ₁..μ₃₄]\nscale_ = [σ₁..σ₃₄]"]
    OBJ3["le: LabelEncoder\n──────────\nclasses_ = BERHI..SOGAY"]
    OBJ4["tensor: FloatTensor\n──────────\nshape = [1, 34]\ndtype = float32"]
    OBJ5["probs: ndarray\n──────────\nshape = (7,)\nsum = 1.0"]

    OBJ4 --> OBJ1
    OBJ2 -->|"scales"| OBJ4
    OBJ1 -->|"logits → softmax"| OBJ5
    OBJ3 -->|"maps indices"| OBJ5
```

### 9. Package Diagram

```mermaid
flowchart TD
    PKG1["app.py"]
    PKG2["torch + torch.nn"]
    PKG3["numpy"]
    PKG4["joblib"]
    PKG5["scikit-learn"]
    PKG6["gradio"]

    PKG1 --> PKG2
    PKG1 --> PKG3
    PKG1 --> PKG4
    PKG1 --> PKG5
    PKG1 --> PKG6
    PKG2 --> PKG3
```

</details>

---

<details>
<summary>📊 Data Flow Diagrams — L0 and L1</summary>

### DFD Level 0 — Context Diagram

```mermaid
flowchart LR
    E1["👤 User"]
    P0(("0.0\nPomologyAI\nClassification\nSystem"))
    E2["📦 DateFruit\nDataset"]

    E1 -->|"34 morphological features"| P0
    P0 -->|"variety + confidence scores"| E1
    E2 -->|"training feature vectors + labels"| P0
```

### DFD Level 1 — System Processes

```mermaid
flowchart TD
    E1A["👤 User"]
    E2A["📦 DateFruit Dataset"]

    P1A(("1.0\nReceive Feature\nInput"))
    P2A(("2.0\nStandardize\nFeatures"))
    P3A(("3.0\nRun ANN\nForward Pass"))
    P4A(("4.0\nDecode Class\nLabels"))
    P5A(("5.0\nRender\nConfidence Output"))

    D1A[("D1: scaler.pkl\nStandardScaler")]
    D2A[("D2: pomology_model.pth\nANN Weights")]
    D3A[("D3: label_encoder.pkl\nClass Names")]

    E1A -->|"34 float slider values"| P1A
    E2A -->|"fit StandardScaler + train ANN"| D1A
    E2A -->|"fit LabelEncoder"| D3A
    P1A -->|"raw ndarray 1x34"| P2A
    D1A -->|"mean and scale params"| P2A
    P2A -->|"scaled tensor 1x34"| P3A
    D2A -->|"weights and biases"| P3A
    P3A -->|"logits 1x7 then softmax"| P4A
    D3A -->|"class name mapping"| P4A
    P4A -->|"class to prob dict"| P5A
    P5A -->|"ranked confidence labels"| E1A
```

</details>

---

## 📁 Folder Structure

```
PomologyAI/                         ← HF Space root (cloned repo)
├── app.py                          ← Entrypoint: ANN class + predict() + Gradio UI
├── requirements.txt                ← Runtime dependencies
├── pomology_model.pth              ← Serialized ANN state_dict (trained weights)
├── scaler.pkl                      ← Fitted StandardScaler (joblib)
├── label_encoder.pkl               ← Fitted LabelEncoder — maps int→class name
└── README.md                       ← This file (HF Space card + documentation)
```

---

## ⚙️ Prerequisites

- Python 3.10+
- Git with [Git LFS](https://git-lfs.github.com/) (for `.pth` files > 10MB)
- Hugging Face account + `huggingface_hub` CLI

---

## 🚀 Local Installation

```bash
# 1. Clone the Space
git clone https://huggingface.co/spaces/KARTHIKAKRISHNA123/PomologyAI
cd PomologyAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ensure artifacts are present
ls pomology_model.pth scaler.pkl label_encoder.pkl

# 4. Launch locally
python app.py
# → Opens at http://localhost:7860
```

---

## 🏋️ Training Artifacts — How to Export from Notebook

Run these cells **at the end** of `ANN_Classification.ipynb` before deploying:

```python
import torch, joblib

# Export model weights
torch.save(model.state_dict(), "pomology_model.pth")

# Export preprocessors
joblib.dump(scaler, "scaler.pkl")           # StandardScaler
joblib.dump(le, "label_encoder.pkl")        # LabelEncoder

print("Classes:", list(le.classes_))
print("✅ All artifacts exported")
```

Move all three files into the cloned HF Space directory before `git push`.

---

## 🧪 Inference Pipeline Internals

```python
# What predict() does step by step
arr        = np.array(feature_values, dtype=np.float32).reshape(1, -1)  # (1, 34)
arr_scaled = scaler.transform(arr)                                        # (1, 34) standardized
tensor     = torch.tensor(arr_scaled, dtype=torch.float32)               # FloatTensor [1, 34]

with torch.no_grad():
    logits = model(tensor)                       # [1, 7] raw scores
    probs  = torch.softmax(logits, dim=1)        # [1, 7] probabilities
    probs  = probs.squeeze().numpy()             # (7,) numpy

return {CLASS_NAMES[i]: float(probs[i]) for i in range(7)}
```

---

## 📦 Dependencies

```
torch          — ANN architecture, weight loading, tensor ops
numpy          — ndarray construction and reshaping
scikit-learn   — StandardScaler, LabelEncoder
joblib         — Artifact serialization and loading
gradio         — Blocks UI, sliders, label output, HF Spaces serving
```

---

## 🚀 Deployment

```bash
cd PomologyAI/

# Stage everything
git add app.py requirements.txt pomology_model.pth scaler.pkl label_encoder.pkl README.md

# Commit
git commit -m "feat: initialize PomologyAI inference engine"

# Push → triggers HF build pipeline
git push

# Monitor: HF dashboard shows Building → Running
```

> **Git LFS** — If `pomology_model.pth` exceeds 10MB, initialize LFS first:
> ```bash
> git lfs install
> git lfs track "*.pth" "*.pkl"
> git add .gitattributes
> ```

---

## 🔒 Security Considerations

- Model runs entirely on CPU inside HF Spaces — no GPU cost or attack surface
- All inputs are float sliders with `min/max` bounds — no raw string injection surface
- Artifacts are read-only at inference time; no user data is persisted

---

## ⚡ Performance

| Metric | Value |
|---|---|
| Inference latency | < 50ms on CPU |
| Model parameters | ~6,500 (34×64 + 64×64 + 64×7) |
| Memory footprint | < 1MB |
| Input features | 34 float32 values |
| Output | 7-class softmax distribution |

---

## 👩‍💻 Author

**Karthika Krishna M**  
B.E. Computer Science & Engineering  
Anna University Regional Campus, Tirunelveli  
Co-founder, Niranthara · AI/ML Engineer  

[![GitHub](https://img.shields.io/badge/GitHub-KARTHIKAKRISHNA123-181717?logo=github)](https://github.com/KARTHIKAKRISHNA123)
[![HuggingFace](https://img.shields.io/badge/🤗-KARTHIKAKRISHNA123-FFD21E)](https://huggingface.co/KARTHIKAKRISHNA123)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
