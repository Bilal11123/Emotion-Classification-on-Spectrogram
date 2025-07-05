# Emotion-Classification-on-Spectrogram
Using CNNs and ViT to train Emotional Classification model, for increased privacy
# Emotion Classification with MobileViT

This project uses the [MobileViT](https://arxiv.org/abs/2110.02178) neural network to classify facial expressions into five emotional classes using a custom dataset. It includes advanced training strategies like custom classification heads, label smoothing, learning rate scheduling, progressive layer freezing, and heavy data augmentation.

## Project Description: 

A spectrogram is a visual representation of how the frequency content of a signal (such as audio, vibration, or biomedical data) changes over time. It is created using the Short-Time Fourier Transform (STFT), which splits the signal into short, overlapping segments, computes the Fourier Transform for each segment, and then maps the frequency intensity (amplitude) as a function of time, typically using color or brightness to indicate magnitude. The problem consists of spectrograms (provided) of 162 identities of five categories "N", "F", "H", "sad", and "S". In total, the dataset contains ~ 815 samples with division of 80%, 10%, and 10% for training, validation, and test splits. The main challenge is to solve the problem using deep learning techniques with the prime aim of achieving the highest accuracy.

---

## 📁 Classes

The dataset includes the following emotion categories:

- **F** – Fear
- **H** – Happiness
- **Sad** – Sadness
- **S** – Surprise
- **N** – Neutral

---

## 📸 Dataset

- Dataset is loaded from `.pt` files (`train_dataset_full.pt`, `val_dataset_full.pt`, `test_dataset_full.pt`), which contain:
  - `images`: Tensor images
  - `labels`: Corresponding class labels
  - `filenames`: Image file identifiers (for tracking results)

---

## 🧠 Model Architecture

- **Base Model**: `mobilevit_s` from [timm](https://github.com/huggingface/pytorch-image-models)
- **Linear Learning**: Training only the Classifier Head first
- **Gradual Unfreezing of Layers**: Gradually unfreeze the CNN and ViT layers 
- **Modified Classifier Head**:
  ```python
  nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.BatchNorm1d(256),
      nn.Linear(256, NUM_CLASSES)
  )

