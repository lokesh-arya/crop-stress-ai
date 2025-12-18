# 🌿 Multi-Stage Deep Learning Pipeline for Computational Plant Pathology

A robust **three-stage Computer Vision system** designed to identify plant species and diagnose specific leaf stresses or diseases from a **raw input image**. This project demonstrates advanced techniques including **synthetic data generation**, **cascaded model inference**, and **vein-enhanced 4-channel classification** for high diagnostic accuracy.



| Project Domain    | Methodology   | Models Used   | Final Output  |
|-------------------|---------------|---------------|---------------|
| AgriTech & Computational Plant Pathology | 3-Stage Sequential Inference Pipeline |YOLOv8 (Detection), ResNet-50 (Classification) | Species, Diagnosis, Confidence Scores |

---

# 1. Project Architecture (Three-Stage Pipeline)

The system operates as a **cascade**, ensuring maximum diagnostic accuracy by refining the input at each step.

### **Stage 1 — Leaf Detection & Isolation (YOLOv8)**  
**Goal:** Detect and crop the leaf from the raw image, removing background noise.

**Methodology:**
- Model trained entirely on a **custom synthetic dataset (15k–30k images)**.
- Synthetic images generated using **Domain Randomization**.

**Output:**  
✔ Bounding Box (x1, y1, x2, y2)  
✔ Clean, isolated leaf crop

---

### **Stage 2 — Plant Identification (ResNet-50, 9 Species)**  
**Goal:** Identify plant species (e.g., Tomato, Ash Gourd).

**Input:** RGB crop + **Vein Channel** (4-channel input)

**Output:**  
✔ Species ID  
✔ Species Confidence Score

---

### **Stage 3 — Stress Classification (ResNet-50, 57 Subtypes)**  
**Goal:** Diagnose the specific crop health stress or disease.

**Input:** Same 4-channel preprocessed leaf image

**Output:**  
✔ Stress Label (e.g., `ash_gourd__N_Mg`)  
✔ Stress Confidence Score

---

# 2. Key Technical Innovations

### **2.1. Vein-Enhanced 4-Channel Input**

To improve discrimination between visually similar disease patterns, an additional **structural/vein channel** is added:

| Channels | Description         |
|---------|----------------------|
| 1–3     | RGB                 |
| 4       | Vein Channel (Laplacian + Gaussian blur) |

**Purpose:**  
Focuses the model on venation + leaf geometry — features less affected by surface color variations.

The necessary preprocessing code is modularly implemented in `data_utils.py`

---

### **2.2. Synthetic Data & Domain Randomization**

To eliminate the need for large annotated detection datasets:

### Steps:
1. **Mask Extraction → RGBA Leaf Assets**  
   `1a_create_assets.py`
2. **Synthetic Image Generation**  
   `1b_generate_synthetic_data.py`  
   Includes: Random placement, 30%–80% scaling, Full 360° rotation, Horizontal flips, Illumination jitter, Real-world backgrounds   

---

# 3. Current Performance & Scaling Roadmap

> ⚠️ **Note:** *The models provided in the repository are indicative of a Proof-of-Concept. Due to constraints, they were not trained to full convergence. This section outlines the current state and the path to production-ready accuracy.*

### **Current Performance**

| Stage | Task | Metric | POC Value | Target | Root Cause Analysis |
|-------|------|---------|------------|---------|--------|
| Stage 1 | Detection (YOLOv8) | mAP@50 | ~0.91 | >0.95 | Synthetic data successfully achieved strong boundary detection. |
| Stage 2 | Species ID (9 classes) | Accuracy | ~78.2% | >95% | Good generalization, but requires fine-tuning to maximize separation. |
| Stage 3 | Stress Classification (57 classes) | Accuracy | ~45.1% | >85% | Severe underfitting—the large number of classes demands extended training ($100+$ epochs) and more aggressive augmentation than was completed. |

---

### **Future Work and Scaling Roadmap**
- Full Training Run: Re-train Stage 2 and 3 models until convergence using optimal hardware and extended epoch counts.
- Hyperparameter Optimization: Conduct an exhaustive search for optimal learning rate decay and regularization.
- Model Export: Convert PyTorch weights (.pth) to production formats (e.g., ONNX, TorchScript) for optimized inference speed.

---

#   4. Local Setup & Execution

### **Prerequisites**
- Python **3.8+**
- Trained weights (`best.pt`, etc.) placed in appropriate folders under `runs/`

---

### **Environment Setup**

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
# OR
venv\Scripts\activate         # Windows
# Install core dependencies
pip install opencv-python-headless numpy pillow tqdm ultralytics
# Install PyTorch and Torchvision separately
```
### **Running the Full Inference Pipeline**

The `6_predict.py` script executes the full sequential inference (Detection → Species ID → Stress Classification).

```bash
# Ensure your TEST_IMAGE path is updated within 6_predict.py
python 6_predict.py
```

**Example Console Output**

```
--- Running 3-Stage Inference on: path/to/your/test_image.jpg ---

✅ Final Diagnosis:
  Species Identified: Tomato (Conf: 0.98)
  Diagnosis:          Tomato___Late_blight (Conf: 0.45)
  Processing Time:    0.095 seconds
```

# 5. Project Directory Structure

```
crop_stress_project/
├── 1a_create_assets.py                 # Leaf mask extraction
├── 1b_generate_synthetic_data.py       # Synthetic dataset generation
├── 2_train_yolo.py                     # Stage 1 training script
├── 3_crop_and_organize.py              # Crops real images using Stage 1 output
├── 4_train_species_id.py               # Stage 2 training script (9 classes)
├── 5_train_stress_class.py             # Stage 3 training script (57 subtypes)
├── 6_predict.py                        # Final 3-Stage Inference Pipeline
├── data_utils.py                       # Vein channel preprocessing and custom dataset
├── leaf_detection.yaml                 # YOLO config file
├── original_leaves/                    # Source images for asset creation
├── backgrounds/                        # Background assets
├── assets/                             # Output: RGBA leaf assets
├── synthetic_dataset/                  # Output: YOLO-ready synthetic data
├── cropped_for_species_id/             # Output: Cropped data for Stage 2 (9 species folders)
└── cropped_for_stress_class/           # Output: Cropped data for Stage 3 (57 stress subtype folders)
```
