# ☀️ Solar Panel Segmentation — NARSS GeoAI Research

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=for-the-badge&logo=pytorch)
![SegFormer](https://img.shields.io/badge/SegFormer-B4-orange?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![NARSS](https://img.shields.io/badge/NARSS-Egypt-green?style=for-the-badge)

**End-to-end deep learning pipeline for photovoltaic panel detection from multi-resolution aerial and satellite imagery, achieving 85.03% IoU using SegFormer-B4 Transformer architecture.**

| Metric | Score |
|--------|-------|
| **IoU (Intersection over Union)** | **85.03%** |
| **Dice / F1 Score** | **91.91%** |
| **Test Set Size** | 382 images |
| **Statistical Validation** | Wilcoxon p < 0.001 |
| **Training Dataset** | PV03 (2,308 real aerial orthophotos) |

</div>

---

## 🎯 Research Overview

This project was developed during a **120-hour GeoAI practical training** at the **National Authority for Remote Sensing and Space Sciences (NARSS), Egypt**. It addresses the critical challenge of automated solar panel detection for renewable energy monitoring and urban solar potential assessment.

### 🔬 Research Contributions
1. **Super-resolution preprocessing** using SwinIR (×2.67 scale) on low-resolution satellite imagery
2. **Domain adaptation** from aerial orthophotos (PV03) to satellite imagery (PV08 — Pleiades sensor)
3. **Statistical validation** with Wilcoxon signed-rank test confirming significance (p < 0.001)
4. **Multi-resolution pipeline** handling 0.30m GSD aerial to 0.5m satellite imagery

## 🏗️ Pipeline Architecture

```
Raw Imagery (Multi-Resolution)
        │
        ▼
┌─────────────────────┐
│  SwinIR             │ ← Super-Resolution (×2.67)
│  Super-Resolution   │   For low-res satellite imagery
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Preprocessing      │ ← Normalization, patching,
│  & Augmentation     │   geometric transforms
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  SegFormer-B4       │ ← Mix Transformer encoder
│  Segmentation       │   + Lightweight MLP decoder
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Post-Processing    │ ← Morphological ops,
│  & Evaluation       │   IoU, Dice computation
└─────────────────────┘
```

## 📊 Detailed Results

### Performance Metrics
| Model Variant | IoU | Dice/F1 | Notes |
|---------------|-----|---------|-------|
| SegFormer-B2 (baseline) | ~78% | ~87% | Smaller encoder |
| SegFormer-B4 (ours) | **85.03%** | **91.91%** | Best performance |
| + SwinIR preprocessing | Improved | Improved | On satellite imagery |

### Statistical Validation
- **Test**: Wilcoxon signed-rank test (non-parametric)
- **Comparison**: SegFormer-B4 vs baseline models
- **Result**: p < 0.001 (highly statistically significant)
- **N**: 382 held-out test images (geographic split — unseen locations)

### Dataset: PV03
- **Total images**: 2,308 real aerial orthophotos
- **Resolution**: 0.30m GSD (Ground Sampling Distance)
- **Annotation**: Binary segmentation masks (solar panel / non-panel)
- **Split**: Train / Validation / Test (geographic hold-out)
- **Subsets**: Ground-mounted + Rooftop panels

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Core Model** | SegFormer-B4 (Mix Transformer) |
| **Super-Resolution** | SwinIR (Image Restoration Transformer) |
| **Framework** | PyTorch, HuggingFace Transformers |
| **Data Processing** | NumPy, PIL, OpenCV, GDAL |
| **Visualization** | Matplotlib, custom overlay tools |
| **Statistical Analysis** | SciPy (Wilcoxon test) |
| **Experiment Tracking** | Jupyter Notebooks |
| **Remote Sensing** | GDAL, rasterio |

## 🚀 Quick Start

```bash
git clone https://github.com/M12hindawy/solar-panel-segmentation-narss.git
cd solar-panel-segmentation-narss
pip install -r requirements.txt
```

### Run Inference on Your Image
```python
from inference import SolarPanelSegmentor

model = SolarPanelSegmentor("checkpoints/segformer_b4_best.pth")
mask = model.predict("your_aerial_image.jpg")
model.visualize(mask, "your_aerial_image.jpg")
```

### Run the Full Training Pipeline
```bash
jupyter notebook KAGGLE_COMPLETE_END_TO_END_RESEARCH_PIPELINE.ipynb
```

## 📁 Project Structure

```
solar-panel-segmentation-narss/
├── KAGGLE_COMPLETE_END_TO_END_RESEARCH_PIPELINE.ipynb  # Full training pipeline
├── inference.py              # Inference script for new images
├── requirements.txt          # Dependencies
├── FINAL/
│   ├── FINAL_README.md       # Detailed research documentation
│   ├── FINAL_RESEARCH_REPORT.md  # Full research report
│   └── FINAL_VALIDATION_REPORT.md  # Statistical validation details
├── results/
│   ├── MODEL_ONLY_FULL_SCENE_OUTPUT.png  # Full scene prediction
│   ├── SOLAR_PANELS_EXACT_FINAL_PERFECT_TEST.png
│   └── PLEIADES_TIF_FULL_SCENE_SEGMENTATION.png  # Satellite imagery test
└── checkpoints/              # Trained model weights
```

## 🌍 Real-World Application

The trained model was tested on:
- **PV03 Benchmark**: 2,308 real aerial orthophotos from Chinese cities
- **Pleiades Satellite Imagery**: 0.5m resolution commercial satellite data
- **Egyptian Context**: Adapted for NARSS Egypt remote sensing use cases

## 📜 Certificate

This project was developed during the **GeoAI: Segmentation of Solar Panels** practical training at:  
🏛️ **NARSS — National Authority for Remote Sensing and Space Sciences, Egypt**  
📅 September 2026 | 120 Hours | Certificate Awarded

## 👤 Author

**Mohamed Ashraf Abdelhamed**  
📍 Giza, Egypt | AI & Data Engineering Student  
🔗 [LinkedIn](https://linkedin.com/in/mohamed-ashraf-b7b5a72b6/) | [GitHub](https://github.com/M12hindawy) | [Portfolio](https://mohamedhendawy.netlify.app)

## 📄 License

MIT License — See [LICENSE](LICENSE) file.

---

> *"Automated detection of renewable energy infrastructure from satellite imagery — bridging remote sensing and deep learning for a sustainable future."*
