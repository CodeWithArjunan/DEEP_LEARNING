# 🧠 Deep Learning Projects

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

> A hands-on collection of Deep Learning and Computer Vision projects built with Python, TensorFlow, Keras, and OpenCV — covering **neural networks, gesture recognition, drowsiness detection, OCR, object detection, licence plate recognition**, and more.

---

## 📋 Table of Contents

- [Repository Structure](#-repository-structure)
- [Projects Overview](#-projects-overview)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)
- [Learning Roadmap](#-learning-roadmap)
- [Connect](#-connect-with-me)

---

## 📁 Repository Structure

```
Deep_Learning/
│
├── 📂 Drowsiness_Detection/              # Real-time driver drowsiness alert system
│   └── main.py
│
├── 📂 Email_Spam_detect/                 # Email spam classifier (NLP + Deep Learning)
│   ├── EmailCollection
│   └── Spam.ipynb
│
├── 📂 First_Neural_Network/              # Diabetes prediction — first NN from scratch
│   ├── Training_Model.py
│   ├── model.json
│   ├── pima-indians-diabetes.csv
│   └── test.py
│
├── 📂 Hand_Gesture_Reg/                  # Hand gesture recognition with CNN
│   ├── HandGestureDataset/
│   ├── model.h5
│   ├── model.json
│   ├── model4.json
│   ├── train.py
│   ├── train - Copy.py
│   ├── test.py
│   └── test - Copy.py
│
├── 📂 Lable_Reading/                     # OCR — reads printed text from images
│   ├── opticalCharacterRecognition.py
│   ├── New.txt
│   ├── TEST1.jpg
│   ├── TEST2.jpg
│   └── TEST3.jpg
│
├── 📂 Licence_plate_Recognition/         # Automatic licence plate reading
│   ├── main.py
│   ├── Final_images.png
│   ├── car1.jpg
│   └── car2.jpg
│
├── 📂 Object_detection/                  # Real-time object detection with MobileNet SSD
│   ├── object_dect.py
│   ├── MobileNetSSD_deploy.caffemodel
│   └── MobileNetSSD_deploy.prototxt
│
├── 📂 Road_sign_Recognition/             # Road sign image classifier (CNN)
│   ├── Test/
│   └── Train/
│
├── 📂 Vechicle_detection/                # Vehicle & traffic detection via camera
│   ├── main.py
│   ├── cars.xml
│   └── Traffic_Jam.jpg
│
└── 📂 img_classifier/                    # Custom CNN image classifier
    ├── Dataset/
    ├── model.json
    ├── model.weights.h5
    ├── train.py
    ├── test.py
    └── test2.py
```

---

## 🗂️ Projects Overview

---

### 😴 Drowsiness Detection
Detects driver drowsiness in real-time using facial landmark analysis and Eye Aspect Ratio (EAR). Triggers an alert when eyes remain closed beyond a threshold — a real-world road safety application.

| File | Description |
|---|---|
| `main.py` | Real-time drowsiness detection and alert logic |

**Key concepts:** EAR (Eye Aspect Ratio), facial landmarks, dlib, OpenCV

---

### 📧 Email Spam Detection
Classifies emails as **spam or ham** using Natural Language Processing combined with a Deep Learning text classifier.

| File | Description |
|---|---|
| `Spam.ipynb` | Full notebook: preprocessing → vectorization → training → evaluation |
| `EmailCollection` | Raw email dataset for training |

**Key concepts:** Text preprocessing, TF-IDF / word embeddings, binary classification, NLP

---

### 🧬 First Neural Network — Diabetes Prediction
A foundational project: building and training a **neural network from scratch** to predict diabetes using the Pima Indians Diabetes dataset.

| File | Description |
|---|---|
| `Training_Model.py` | NN architecture definition, training loop, and model saving |
| `test.py` | Load saved model and run predictions on new data |
| `model.json` | Saved model architecture in JSON format |
| `pima-indians-diabetes.csv` | Pima Indians diabetes dataset |

**Key concepts:** Dense layers, sigmoid/ReLU activations, binary cross-entropy, model save/load

---

### 🤚 Hand Gesture Recognition
Recognizes hand gestures from images using a **CNN trained on a custom gesture dataset**. Extendable for sign language recognition or gesture-based control systems.

| File | Description |
|---|---|
| `train.py` | CNN model training on the gesture dataset |
| `test.py` | Load saved model and classify new gesture images |
| `model.h5` | Saved trained Keras model weights |
| `model.json` / `model4.json` | Saved model architectures |
| `HandGestureDataset/` | Training images organized by gesture class |

**Key concepts:** CNN, image augmentation, multi-class classification, Keras save/load

---

### 🔤 Label Reading (OCR)
Reads and extracts **printed text from real-world images** using Optical Character Recognition — useful for document scanning and automated data entry.

| File | Description |
|---|---|
| `opticalCharacterRecognition.py` | Core OCR pipeline using Tesseract + OpenCV |
| `TEST1.jpg`, `TEST2.jpg`, `TEST3.jpg` | Sample images containing printed text |
| `New.txt` | Extracted text output file |

**Key concepts:** Pytesseract, image binarization, thresholding, contour detection

---

### 🚗 Licence Plate Recognition
Automatically detects and reads **vehicle licence plates** from car photos using image processing and OCR.

| File | Description |
|---|---|
| `main.py` | Licence plate detection, ROI cropping, and OCR pipeline |
| `car1.jpg`, `car2.jpg` | Test vehicle images |
| `Final_images.png` | Processed output with detected plate region highlighted |

**Key concepts:** Contour detection, Region of Interest (ROI), morphological operations, OCR

---

### 📦 Object Detection — MobileNet SSD
Detects and labels **multiple objects in real-time** using the pre-trained **MobileNet SSD** model — a fast and lightweight deep learning detector ideal for edge deployment.

| File | Description |
|---|---|
| `object_dect.py` | Real-time object detection with bounding boxes and labels |
| `MobileNetSSD_deploy.caffemodel` | Pre-trained MobileNet SSD model weights |
| `MobileNetSSD_deploy.prototxt` | Model architecture definition (Caffe format) |

**Key concepts:** MobileNet SSD, Caffe model loading via OpenCV DNN, confidence thresholding

---

### 🛑 Road Sign Recognition
A **CNN-based image classifier** trained to detect and categorize different road signs — a key component for autonomous driving systems.

| Folder | Description |
|---|---|
| `Train/` | Training images organized by road sign category |
| `Test/` | Test images for model accuracy evaluation |

**Key concepts:** Multi-class CNN, image augmentation, transfer learning, traffic sign datasets

---

### 🚙 Vehicle Detection
Detects vehicles in traffic scenes using **Haar Cascade Classifiers** — useful for traffic monitoring, smart surveillance, and road analytics.

| File | Description |
|---|---|
| `main.py` | Vehicle detection and counting pipeline |
| `cars.xml` | Pre-trained Haar Cascade model for car detection |
| `Traffic_Jam.jpg` | Sample traffic image for testing |

**Key concepts:** Haar Cascade, multi-scale sliding window, OpenCV detection pipeline

---

### 🖼️ Image Classifier (Custom CNN)
A custom **Convolutional Neural Network** trained to classify images into categories from a self-curated dataset. Includes a complete train-to-inference pipeline.

| File | Description |
|---|---|
| `train.py` | CNN architecture definition and training |
| `test.py` / `test2.py` | Model evaluation and inference on test images |
| `model.json` | Saved model architecture |
| `model.weights.h5` | Saved trained model weights |
| `Dataset/` | Categorized image dataset for training |

**Key concepts:** Conv2D, MaxPooling, Flatten, Dense, Dropout, model checkpointing

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/CodeWithArjunan/<your-dl-repo>.git
cd <your-dl-repo>
```

### Install dependencies

```bash
pip install tensorflow keras opencv-python numpy pandas matplotlib seaborn pytesseract jupyter dlib
```

### Run individual projects

```bash
# Drowsiness Detection
cd Drowsiness_Detection && python main.py

# First Neural Network — Train
cd First_Neural_Network && python Training_Model.py

# First Neural Network — Test
cd First_Neural_Network && python test.py

# Hand Gesture Recognition — Train
cd Hand_Gesture_Reg && python train.py

# Hand Gesture Recognition — Test
cd Hand_Gesture_Reg && python test.py

# Object Detection
cd Object_detection && python object_dect.py

# Licence Plate Recognition
cd Licence_plate_Recognition && python main.py

# Vehicle Detection
cd Vechicle_detection && python main.py

# Label Reading (OCR)
cd Lable_Reading && python opticalCharacterRecognition.py

# Image Classifier — Train
cd img_classifier && python train.py

# Image Classifier — Test
cd img_classifier && python test.py

# Email Spam Detection (Jupyter Notebook)
cd Email_Spam_detect && jupyter notebook Spam.ipynb
```

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Language** | Python 3.x |
| **Deep Learning** | TensorFlow, Keras |
| **Computer Vision** | OpenCV, dlib, Haar Cascades |
| **Pre-trained Models** | MobileNet SSD (Caffe format) |
| **OCR** | Pytesseract |
| **Data & Visualization** | NumPy, Pandas, Matplotlib, Seaborn |
| **Notebook** | Jupyter Notebook |
| **Model Formats** | `.h5`, `.json`, `.weights.h5`, `.caffemodel` |

---

## 📈 Learning Roadmap

- [x] First Neural Network (Dense / Perceptron layers)
- [x] CNN — Hand Gesture Recognition
- [x] CNN — Custom Image Classifier
- [x] CNN — Road Sign Recognition
- [x] Computer Vision — Drowsiness Detection (EAR + dlib)
- [x] Computer Vision — Vehicle Detection (Haar Cascade)
- [x] Computer Vision — Object Detection (MobileNet SSD)
- [x] Computer Vision — Licence Plate Recognition
- [x] OCR — Label / Text Reading from Images
- [x] NLP — Email Spam Detection
- [ ] Recurrent Neural Networks (RNN / LSTM)
- [ ] Transfer Learning (VGG16, ResNet50, EfficientNet)
- [ ] Generative AI (GANs / Diffusion Models)
- [ ] Model Deployment (FastAPI / Flask / Streamlit)
- [ ] Edge AI & TensorFlow Lite

---

## 🤝 Connect with Me

[![GitHub](https://img.shields.io/badge/GitHub-CodeWithArjunan-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/CodeWithArjunan)

---

> ⭐ **Star this repo** if you find it useful or are following along the deep learning journey!
