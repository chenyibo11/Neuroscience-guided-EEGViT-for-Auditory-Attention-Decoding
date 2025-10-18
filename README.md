# Neuroscience-guided-EEGViT-for-Auditory-Attention-Decoding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Project Overview

This project addresses a critical challenge in **Auditory Attention Decoding (AAD)**: effectively disentangling attended speech from raw Electroencephalogram (EEG) signals.

Conventional AAD models often fall short because they overlook fundamental neuroscience priors, such as:
* The brain's **hierarchical processing** of auditory stimuli.
* The **temporal asynchrony** between acoustic inputs and the corresponding neural responses.
* Interference from **unattended speech** that is also encoded in the EEG signal.

This oversight leads to suboptimal decoding accuracy and models that are difficult to interpret.

To overcome these limitations, we propose a **neuroscience-inspired framework** that explicitly integrates these priors into the model design. Our key contributions are:

1.  **EEGViT**: A novel architecture inspired by the Vision Transformer, designed to mimic the brain's hierarchical integration of auditory information by processing EEG in patches of varying temporal scales.
2.  **Hierarchical Contrastive Learning (HCL)**: A fine-grained alignment strategy that matches EEG embeddings with their corresponding attended and unattended speech embeddings (extracted by WavLM).
3.  **Hierarchical Mutual Information Minimization (HMIM)**: An objective function designed to actively suppress and disentangle the components of the unattended speech from the EEG representations.

We validated our framework on three public AAD datasets: **KUL, DTU, and NJU**. The results show that our approach substantially outperforms state-of-the-art methods, achieving higher accuracy while producing representations that are consistent with established neuroscience principles, thereby enhancing both performance and interpretability.

---

## 📂 Project Structure

Here is a breakdown of the key directories and files in this project:

```
.
├── dataset/              # Stores processed datasets (e.g., .npy, .csv, .pt files)
├── eegdata/              # Contains the raw EEG data
├── model_path/           # Directory to save and load trained model checkpoints (.pth, .h5, etc.)
├── models/               # Contains Python scripts defining the model architectures
├── PM/                   # [Please add a detailed description for the 'PM' folder here]
├── stimulus/             # Contains the visual stimulus materials (images, videos, etc.) used in the experiment
├── vit-base-patch16-224/ # Stores the pre-trained Vision Transformer model files
│
├── get_feature.py        # Script for data preprocessing and feature extraction
├── run_KUL_1s.py         # Main script to execute training, evaluation, or inference
└── README.md             # This documentation file
```

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites & Installation

First, clone the repository to your local machine:
```bash
git clone [Your Repository SSH or HTTPS URL]
cd [Your Project Folder Name]
```

It is highly recommended to use a virtual environment (like `venv` or `conda`) to manage dependencies.

```bash
# Create and activate a venv
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

Next, install the required packages.
**Note**: Please create a `requirements.txt` file based on the imports in your Python scripts (e.g., `torch`, `numpy`, `pandas`, `scikit-learn`, `mne`).

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

1.  **EEG Data**: Download the raw EEG data and place it inside the `eegdata/` directory.
2.  **Stimulus Materials**: Place all visual stimuli (images/videos) into the `stimulus/` directory.
3.  **Pre-trained Model**: Download the `vit-base-patch16-224` model from [Provide a link to the model source, e.g., Hugging Face] and ensure the files are located in the `vit-base-patch16-224/` directory.

[If necessary, add more detailed instructions here about the expected data structure.]

### 3. How to Run

1.  **(Optional) Feature Extraction**
    If your workflow requires a separate feature extraction step, run the `get_feature.py` script.
    ```bash
    python get_feature.py --input_path eegdata/ --output_path dataset/
    ```
    *[Modify the command and its arguments according to your script's implementation.]*

2.  **Train and Evaluate the Model**
    Use the main script `run_KUL_1s.py` to start the model training and evaluation process.
    ```bash
    python run_KUL_1s.py --config [path/to/config.yaml] --mode train
    ```
    *[Adjust the command and its arguments (e.g., participant ID, epochs, learning rate) as needed for your script.]*

---

## 🔧 Tech Stack

* **Programming Language**: Python 3.x
* **Core Framework**: [e.g., PyTorch, TensorFlow]
* **Key Libraries**: [e.g., NumPy, Pandas, Scikit-learn, MNE-Python, Matplotlib]

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## 🙏 Acknowledgements (Optional)

* We would like to thank [Name of the data provider, e.g., KU Leuven] for providing the dataset.
* The model implementation was inspired by [Link to a relevant paper or code repository].
