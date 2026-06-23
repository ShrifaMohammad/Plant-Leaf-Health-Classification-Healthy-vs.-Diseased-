# Plant Leaf Health Classification (Healthy vs. Diseased)

A Convolutional Neural Network (CNN) that classifies plant leaf images as **healthy** or **diseased**, built during an AI summer training program at **Weqa'a Center** (National Center for the Control of Plant Pests and Diseases), Ministry of Environment, Water and Agriculture, Saudi Arabia.

> ⚠️ **Note on this notebook:** The original `.ipynb` file was not available — this notebook (`leaf_classification.ipynb`) was reconstructed from an exported PDF printout of the original Colab session. All code, parameters, training logs, and prediction outputs are transcribed exactly as they appeared in the original run; nothing has been re-run or altered.

## What it does

Given a photo of a plant leaf, the model predicts whether the leaf is **healthy** or **diseased** — enabling field inspectors to get an instant diagnosis from a smartphone photo, without lab analysis.

## Dataset

- Leaf images collected in the field under varying lighting conditions, angles, and backgrounds
- 2 classes: `healthy` (1,478 images) and `diseased` (997 images)
- Split: 1,981 images for training / 494 for validation (80/20 split via `ImageDataGenerator`)

## Model architecture

A CNN with 3 convolutional blocks, built in Keras/TensorFlow:

```
Conv2D(32, 3x3, relu) → MaxPooling2D
Conv2D(64, 3x3, relu) → MaxPooling2D
Conv2D(128, 3x3, relu) → MaxPooling2D
Flatten
Dense(256, relu)
Dropout(0.5)
Dense(1, sigmoid)   # binary: healthy vs. diseased
```

- Optimizer: Adam · Loss: binary cross-entropy
- Input size: 224×224×3
- Data augmentation: rotation (±20°), zoom (0.2), horizontal flip
- Trained for 20 epochs, batch size 32

## Results

- **Training accuracy: 99.99%** · **Validation accuracy: ~98.2%** (peaked at 99.6% at epoch 19)
- Validation loss dropped from 0.27 (epoch 1) to as low as 0.038 (epoch 18)
- Manually tested on new, unseen leaf photos — correctly classified both diseased and healthy samples with high confidence (e.g. 1.00 confidence on a healthy leaf test case)

The accuracy curve (training vs. validation, across all 20 epochs) is shown directly in the notebook output.

## Tech stack

- TensorFlow / Keras (`Sequential`, `Conv2D`, `ImageDataGenerator`)
- NumPy, Matplotlib

## Repo contents

```
├── leaf_classification.ipynb   # Full training & testing pipeline (reconstructed from PDF export)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/leaf-health-classification.git
cd leaf-health-classification
pip install -r requirements.txt
```

To retrain: place your leaf dataset in a `Data/` folder with two subfolders, `healthy/` and `diseased/`, then run the notebook in Jupyter or Google Colab.

## The problem

Field inspectors need a fast way to assess plant health without waiting for a lab diagnosis. Manually identifying disease symptoms from leaf appearance is time-consuming and depends on the inspector's experience, which can lead to delayed or inconsistent detection — allowing diseases to spread further before treatment begins. This model addresses that gap by giving an instant, automated healthy/diseased prediction from a single leaf photo, supporting faster decision-making in the field.

This project was built as part of a 200-hour AI summer training program at Weqa'a Center, applying machine learning to real agricultural challenges.

## What I'd improve next

- Add a confusion matrix and precision/recall breakdown on the validation set (only accuracy/loss were tracked in the original run)
- Try transfer learning (e.g. MobileNetV2) to compare against this from-scratch CNN
- Expand the dataset with more diverse disease types beyond binary healthy/diseased
