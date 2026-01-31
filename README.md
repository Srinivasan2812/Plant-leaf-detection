# 🌿 Plant Leaf Disease Detection using CNN

This project implements a **Convolutional Neural Network (CNN)** to classify plant leaf diseases into three categories. Built with **TensorFlow** and **Keras**, the model is optimized for Google Colab and includes a real-time prediction feature where users can upload images to see immediate results.

## 📌 Project Overview

The goal is to automate the detection of crop diseases, which can significantly help farmers take early action. The notebook handles the entire pipeline:

1. **Data Extraction:** Unzipping datasets directly from Google Drive.
2. **Preprocessing:** Data augmentation using `ImageDataGenerator`.
3. **Modeling:** A custom 3-layer CNN architecture.
4. **Inference:** Interactive file upload for disease prediction.

---

📂 Dataset Information

The model is trained on the **Plant Leaf Disease Dataset**, specifically curated for identifying health status in crops. The data is structured to facilitate supervised learning through image classification.

### 1. Data Structure

The images are organized into a directory-based structure, which the `ImageDataGenerator` uses to automatically assign labels:

* **Train Set:** Used for the model to learn features and patterns.
* **Test/Validation Set:** Used to evaluate the model's accuracy on "unseen" data.

2. Classes & Categories

The dataset contains **3 classes**. Based on typical plant disease datasets, these usually represent:

* Healthy:Leaves with no visible signs of infection or nutrient deficiency.
* Rusty: Leaves showing spots, lesions, or wilting.
* Powdery:Leaves showing mosaic patterns or yellowing (chlorosis).

3. Data Augmentation

To prevent **overfitting** (where the model just memorizes the training photos), the following transformations are applied in the code:

* **Rescaling:** Normalizing pixel values to a  range.
* Shear Range (20%): Tilting the image to simulate different camera angles.
* Zoom Range (20%): Randomly zooming in to help the model focus on small spots.
* Horizontal Flip:Flipping the image to ensure the model recognizes the disease regardless of leaf orientation.


---

## 🏗️ Model Architecture

The neural network is a sequential stack designed to extract hierarchical features from leaf images:

* **Convolutional Layers:** 3 layers with increasing filters (32, 64, 128) using **ReLU** activation to detect edges, textures, and specific disease spots.
* **Pooling:** Max Pooling () follows each conv layer to reduce spatial dimensions and focus on the most important features.
* **Fully Connected:** A dense layer of 128 neurons followed by a **Softmax** output layer for 3-way classification.

---

## 🚀 Installation & Usage

### 1. Requirements

This project is designed to run on **Google Colab** to take advantage of free GPU acceleration. The following libraries are used:

* `tensorflow` / `keras`
* `numpy`
* `matplotlib` / `seaborn`
* `PIL` (Pillow)

### 2. Setup

1. Upload your dataset `Plant-leaf-disease detection.zip` to your **Google Drive** home directory.
2. Open the notebook in Google Colab.
3. Run the first cell to mount your Drive; follow the authentication prompt.
4. The script will automatically extract the zip file to the `/content/plant_dataset` temporary folder.

### 3. Training & Prediction

* Execute the training cells to train the model for 5 epochs.
* Scroll to the final cell. A **"Choose Files"** button will appear.
* Upload an image of a plant leaf from your local computer to see the predicted class and confidence percentage.

---

## 📈 Performance Monitoring

The training process uses `categorical_crossentropy` as the loss function and tracks `accuracy`. You can visualize the training history (accuracy and loss curves) using the `history` object generated during the `model.fit()` process.

---
