# ♻️ Garbage Classification using Deep Learning

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-blue?style=for-the-badge)


## 📌 Project Overview
This project implements a **deep learning–based garbage classification system** that classifies waste into predefined categories to support recycling and waste management automation.

The system includes:
- A trained **deep learning model**
- A **Jupyter Notebook** for training and experimentation
- A **Python API** for inference
- A simple **web-based GUI** for user interaction

---
🧠 Model & Approach
- Framework: PyTorch
- Task: **Image Classification
- Model Type: Convolutional Neural Network (CNN)
- Evaluation Metric: Classification accuracy  
- Achieved Gap: **~3%**

The model was trained to recognize different garbage categories using labeled image data, with preprocessing, training, and evaluation handled in the notebook.

---

📂 Project Structure
```text
.
├── Final_Project.ipynb        # Training, evaluation, and experiments
├── Project_API.py             # Python API for model inference
├── Final_GUI.html             # Web-based user interface
├── style.css                  # GUI styling
├── README.md                  # Project documentation
└── .gitignore                 # Ignored files (models, caches, etc.)
