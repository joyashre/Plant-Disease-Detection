# 🌱 Plant Disease Detection 🌿


![crop test](https://github.com/user-attachments/assets/35fcf1db-e079-48ea-a00d-e72b99766220)

Welcome to the Plant Disease Detection project! This repository contains a Python script that uses deep learning to identify plant diseases from leaf images using the PlantVillage dataset. Powered by MobileNetV2, this project simplifies plant disease classification with a beginner-friendly approach, including training, validation, and test splits, plus visualizations like confusion matrices and sample predictions. 🚀

## 📖 Project Overview
This project combines and simplifies code from two sources:

[GitHub Notebook](https://github.com/joyashre/Plant-Disease-Detection/blob/main/Crop_test.ipynb)

[Google Colab Notebook](https://colab.research.google.com/drive/1SgNmfDhtGkP2yH1zhodkn-JlznMEfKzq?usp=sharing)

It uses transfer learning with MobileNetV2 to classify plant leaf images into disease categories (e.g., Tomato_Healthy, Potato_Early_Blight). The dataset is split into:

> Training: 70% 🧠
> Validation: 15% ✅
> Test: 15% 🧪

The script evaluates the model on the test dataset, computes per-image confidence scores, generates a confusion matrix, and displays a sample prediction with the image.

## ✨ Features

> 📷 Image Classification: Detects plant diseases from leaf images.

> 📊 Dataset Splits: 70% train, 15% validation, 15% test.

> 🔍 Evaluation: Test accuracy, per-image confidence scores, and confusion matrix.

> 📈 Visualizations: Training accuracy/loss plots, confusion matrix heatmap, and sample image prediction.

> 💾 Model Saving: Saves the trained model for reuse.

> 🛠 Beginner-Friendly: Clear code with detailed comments.


## 🛠 Prerequisites

Python 3.8+ 🐍

Libraries: TensorFlow, NumPy, Matplotlib, Seaborn, Scikit-learn

PlantVillage dataset 📂


## Download the PlantVillage Dataset:

Get it from Kaggle.

Place it in a folder (e.g., PlantVillage) with subfolders for each class (e.g., Tomato_Healthy, Potato_Early_Blight).



## 📂 Dataset Structure

The dataset should be organized as:

PlantVillage/

├── Tomato_Healthy/

│   ├── image1.jpg

│   ├── image2.jpg

│   └── ...

├── Potato_Early_Blight/

│   ├── image1.jpg

│   └── ...

└── [other classes]/


> What Happens:

📚 Loads and splits the dataset (70% train, 15% validation, 15% test).

🧠 Trains the MobileNetV2 model for 10 epochs.

📊 Evaluates on validation and test sets, printing accuracy and loss.

🔎 Tests the model on the test dataset, showing per-image predictions and confidence.

📈 Saves plots:

Training accuracy/loss: training_results.png

Sample test image prediction: sample_prediction.png


> 💾 Saves the model as plant_disease_model.h5.



## 📈 Outputs

Test Accuracy: Overall accuracy on the test dataset (e.g., 90.5%).

Confidence Scores: Per-image prediction confidence (e.g., "Image 1: True: Tomato_Healthy, Predicted: Tomato_Healthy, Confidence: 95.32%").

Confusion Matrix: Heatmap showing classification performance across classes.

Sample Prediction: Displays one test image with true and predicted labels.

## 🐛 Troubleshooting

Dataset Path Error 📍: Update dataset_path in the script to match your folder location.

Dependency Issues 🛠: Ensure all libraries are installed (pip install tensorflow numpy matplotlib seaborn scikit-learn).

Low Accuracy 📉: Try:

Increasing epochs (e.g., 20).

Unfreezing MobileNetV2 layers:base_model.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])


Memory Issues 💾: Reduce batch_size to 16 in the script.

## 🌟 Extending the Project

📈 Add precision, recall, and F1-score using sklearn.metrics.classification_report.

📊 Plot confidence score distribution for test images.

🌐 Deploy as a web app with Flask or Streamlit.

🧠 Try other models like EfficientNetB0 for better accuracy.


## 🙌 Acknowledgments

Thanks to the PlantVillage dataset creators and TensorFlow community. 🌍


Happy coding, and let’s keep our plants healthy! 🌿🚀

