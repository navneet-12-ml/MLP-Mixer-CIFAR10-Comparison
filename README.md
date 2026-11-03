# Comparative Study of MLP-Mixer, CNN and Vision Transformer on CIFAR-10

A comparative deep learning project evaluating three different computer vision architectures — **MLP-Mixer, Convolutional Neural Network (CNN), and Vision Transformer (ViT)** — on the CIFAR-10 image classification dataset.

The objective is to investigate how different architectural approaches perform when trained under comparable experimental conditions and with approximately similar model parameter counts.

## Author

**Navneet Yadav**
Department of Computer Science and Engineering
Rajiv Gandhi Institute of Petroleum Technology

---

## Project Overview

Traditional CNNs extract spatial features using convolutional operations. Vision Transformers use self-attention to capture relationships between image patches, while MLP-Mixer relies entirely on multilayer perceptrons to mix spatial and channel information.

This project implements all three architectures and compares their performance on CIFAR-10.

### Architectures Implemented

**1. MLP-Mixer**

* Divides images into non-overlapping patches.
* Uses token-mixing MLPs to exchange information across spatial locations.
* Uses channel-mixing MLPs to process feature representations.
* Employs Layer Normalization and residual connections.

**2. Convolutional Neural Network (CNN)**

* Uses convolutional layers to extract spatial features.
* Batch Normalization and ReLU activation.
* Max pooling for spatial downsampling.
* Global Average Pooling and a fully connected classification layer.

**3. Vision Transformer (ViT)**

* Converts images into patch embeddings.
* Uses learnable positional embeddings and a CLS token.
* Applies Multi-Head Self-Attention.
* Uses Transformer encoder blocks for feature extraction.

---

## Dataset

**CIFAR-10**

* Total images: 60,000
* Training images: 45,000
* Validation images: 5,000
* Testing images: 10,000
* Image dimensions: 32 × 32
* Number of classes: 10

### Classes

Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship and Truck.

### Preprocessing

* Random cropping with padding.
* Random horizontal flipping during training.
* Tensor conversion.
* Normalization using CIFAR-10 channel statistics.
* Stratified training-validation split.

The dataset is downloaded automatically through Torchvision when running the data loader.

---

## Model Configurations

| Parameter           | MLP-Mixer |      CNN |     ViT |
| ------------------- | --------: | -------: | ------: |
| Input image         |   32 × 32 |  32 × 32 | 32 × 32 |
| Patch size          |     4 × 4 |      N/A |   4 × 4 |
| Embedding dimension |       128 |      N/A |     128 |
| Main blocks         |         4 | 3 stages |       2 |
| Attention heads     |       N/A |      N/A |       4 |
| Classes             |        10 |       10 |      10 |

The architectures are compact, task-specific implementations designed for comparison on CIFAR-10.

---

## Experimental Setup

| Hyperparameter          | Value                                     |
| ----------------------- | ----------------------------------------- |
| Epochs                  | 50                                        |
| Batch size              | 128                                       |
| Optimizer               | AdamW                                     |
| Learning rate           | 0.001                                     |
| Weight decay            | 0.01                                      |
| Loss function           | Cross Entropy Loss                        |
| Learning rate scheduler | Cosine Annealing                          |
| Framework               | PyTorch                                   |
| Hardware                | NVIDIA GeForce RTX 4050 Laptop GPU (6 GB) |

All three models were trained using the same dataset split and broadly consistent optimization settings.

---

## Results

### Overall Performance

| Metric                   | MLP-Mixer |        CNN |     ViT |
| ------------------------ | --------: | ---------: | ------: |
| Parameters               |   290,314 |    289,194 | 281,226 |
| Training Accuracy        |    92.35% |     95.62% |  83.62% |
| Best Validation Accuracy |    80.54% |     89.06% |  75.76% |
| Test Accuracy            |    80.03% | **87.99%** |  75.38% |
| Test Loss                |    0.6791 | **0.3938** |  0.7350 |

### Key Observations

* CNN achieved the highest test accuracy of **87.99%**.
* MLP-Mixer achieved **80.03%** test accuracy.
* ViT achieved **75.38%** test accuracy.
* CNN outperformed the other two models under the selected training configuration.
* Cat and dog were among the more challenging classes across the models.
* Similar parameter counts do not necessarily imply similar computational costs.

---

## Visualizations

The project includes:

* Training and validation loss curves.
* Training and validation accuracy curves.
* Normalized confusion matrices for all three models.
* Learned token-mixing weight visualizations for MLP-Mixer.

These visualizations help analyze model convergence, generalization and class-wise classification performance.

---

## Project Structure

```text
MLP_MIXER/
│
├── models/
│   ├── mlp_mixer.py
│   ├── cnn.py
│   └── vit.py
│
├── dataset/
│
├── data_loader.py
│
├── train.py
├── train_cnn.py
├── train_vit.py
│
├── evaluate.py
├── evaluate_cnn.py
├── evaluate_vit.py
│
├── visualize_mixer.py
├── confusion_matrices.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

*The structure above is representative; actual script and visualization filenames may vary.*

---

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/MLP-Mixer-CIFAR10-Comparison.git
```

### 2. Navigate into the project

```bash
cd MLP-Mixer-CIFAR10-Comparison
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the models

Run the corresponding training scripts:

```bash
python train.py
python train_cnn.py
python train_vit.py
```

### 6. Evaluate the models

```bash
python evaluate.py
python evaluate_cnn.py
python evaluate_vit.py
```

*Use the actual filenames present in your repository if they differ.*

---

## Research Paper

An accompanying research paper documents the architectures, experimental methodology, results, confusion matrices and observations.

See the `report/` directory for the PDF and editable DOCX versions, if included.

---

## Future Improvements

* Experiment with different patch sizes and embedding dimensions.
* Improve ViT performance through architecture-specific hyperparameter tuning.
* Evaluate computational complexity, inference latency and memory consumption.
* Perform repeated experiments with different random seeds.
* Explore stronger data augmentation and regularization techniques.
* Evaluate the architectures on larger image classification datasets.

---

## References

1. I. Tolstikhin et al., "MLP-Mixer: An All-MLP Architecture for Vision," NeurIPS, 2021.
2. A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR, 2021.
3. A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," University of Toronto, 2009.

---

**Built by Navneet Yadav**
Rajiv Gandhi Institute of Petroleum Technology
