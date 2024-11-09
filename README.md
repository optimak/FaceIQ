# FaceIQ
## UTKFace Multi-Task Age, Gender, and Ethnicity Prediction

Welcome to the **UTKFace Multi-Task Prediction** project! This repository showcases a deep learning pipeline for predicting **age, gender, and ethnicity** using the popular **UTKFace dataset**. This project leverages **transfer learning** with a pre-trained VGG16 model to achieve high accuracy across all three tasks, making it a robust solution for facial analysis.

## 🔥 Key Features
- **Multi-Task Learning**: Simultaneous training for age regression, gender classification, and ethnicity classification.
- **Transfer Learning with VGG16**: Fine-tuned VGG16 as a feature extractor for efficient, high-quality predictions.
- **Early Stopping and Learning Rate Scheduling**: Intelligent training strategy to prevent overfitting and improve model performance.
- **Data Augmentation**: Resizing, scaling, and transforming images for improved model generalization.

## 🛠️ Model Architecture
The model includes:
- **Convolutional Layers** from VGG16 for feature extraction.
- Custom **Fully Connected Layers** for each task:
  - **Age Prediction**: Regression model for predicting continuous age.
  - **Gender Prediction**: Binary classification model with softmax output.
  - **Ethnicity Prediction**: Multi-class classification for five ethnicities.

## 📊 Evaluation Metrics
- **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)** for age regression.
- **Accuracy** for both gender and ethnicity classification tasks.

## 📈 Training Strategy
Utilized **EarlyStopping** and **ReduceLROnPlateau** callbacks to optimize model training, helping the network converge faster while avoiding overfitting.

## 🎨 Visualizations
Training metrics visualized for each task, showcasing **loss, accuracy,** and **error rates** over each epoch.

## 🚀 Getting Started
Clone the repo and set up the environment to start exploring the model’s predictions!

## 🧑‍💻 Libraries & Frameworks
- **TensorFlow** and **Keras** for deep learning.
- **PIL** for image processing.
- **Sklearn** for data preparation and evaluation.

## 💡 Future Work
- Extend the model to predict additional facial attributes.
- Experiment with different architectures (e.g., ResNet, EfficientNet) for improved accuracy.
