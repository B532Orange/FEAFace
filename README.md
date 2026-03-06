# FEAFace

**Feature Enhancing and Adaptive Feature Space Reconstruction for Large Pose Livestock Face Recognition**

This repository contains the official implementation of **FEAFace**, a deep learning framework designed to improve livestock face recognition under large pose variations. The method enhances discriminative facial features and reconstructs an adaptive feature space to achieve more robust recognition performance.

## 📌 Overview

Large pose variations significantly affect the performance of livestock face recognition systems.


The project is implemented in **Python** using deep learning frameworks.

---

## 📂 Project Structure

```
FEAFace/
│
├── model/                # Core model architecture
│   ├── cattleface.py    # model framework
│   └── iresnet.py       # backbone network
│
├── tools/                # Data preprocessing tools
│   ├── gen_trainlist.py  # Generate training dataset list
│   └── gen_list.py       # Generate testing dataset list
│
├── dataloader.py         # Dataset loader
├── train.py              # Training script
├── test.py               # Evaluation / testing script
├── requirements.txt      # Environment dependencies
└── README.md
```

---

## ⚙️ Environment Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

We recommend using **Python 3.8+** and a GPU-enabled environment.

---

## 🗂 Dataset Preparation

Before training, you need to generate dataset lists.

### Generate Training List

```bash
python tools/gen_trainlist.py
```

### Generate Testing List

```bash
python tools/gen_list.py
```

These scripts will generate the file lists used during training and evaluation.

---

## 🚀 Training

To train the model:

```bash
python train.py
```

The training script will load the dataset using `dataloader.py` and train the FEAFace model defined in the `model/` directory.

---

## 🧪 Testing

To evaluate the trained model:

```bash
python test.py
```

Make sure the trained model checkpoint is properly configured before running the evaluation.

---

## 📄 Requirements

All dependencies are listed in:

```
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 📊 Citation

If you find this project useful in your research, please consider citing our work:

```
@article{FEAFace,
  title={Feature Enhancing and Adaptive Feature Space Reconstruction for Large Pose Livestock Face Recognition},
  author={},
  journal={},
  year={}
}
```

---

## 🤝 Acknowledgements

Thanks to the open-source community for providing valuable tools and frameworks that support this research.

---

## 📜 License

This project is released for academic research purposes. Please contact the authors for commercial usage.
