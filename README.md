# Intelligent-Plant-Recognition-and-Disease-Diagnosis-
# 🌿 Plant Leaf Disease Detection using Deep Learning and Machine Learning

## 📘 Overview

This project focuses on the early and accurate detection of plant leaf diseases using a hybrid approach that combines deep learning and classical machine learning models. In response to growing food security challenges caused by diseases affecting major crops, the system leverages advanced image processing, denoising, and classification techniques to identify diseases in leaves of **maize**, **mango**, and **tomato** plants.

A convolutional neural network (MobileNet), Multi-Layer Perceptron (MLP), Random Forest, and XGBoost models were trained and evaluated on the **PlantVillage dataset**. The models were compared based on accuracy, efficiency, and performance, with MobileNet emerging as the most accurate and deployment-friendly model.

---

## 🔍 Problem Statement

Agriculture remains the backbone of many economies, yet it is highly susceptible to losses caused by plant diseases. Timely and precise detection is critical for controlling disease spread, improving yield, and reducing food insecurity. Traditional disease identification relies heavily on human expertise and visual inspection, which is inefficient, inconsistent, and infeasible at scale.

This project aims to:
- Reduce dependency on manual detection.
- Enable faster and more reliable diagnosis through image-based automated systems.
- Provide a real-time disease detection solution deployable even on low-power devices like smartphones or IoT systems.

---

## 🎯 Objectives

- ✅ Develop an intelligent image classification system to detect and classify plant leaf diseases.
- ✅ Apply **MobileNet CNN** architecture for lightweight and fast inference.
- ✅ Train and evaluate **XGBoost**, **Random Forest**, and **MLP classifiers** for comparative analysis.
- ✅ Perform **image denoising and data augmentation** to improve training dataset quality.
- ✅ Use the **PlantVillage dataset** and enhance it via augmentation techniques.
- ✅ Determine the most effective model based on precision, recall, F1-score, and confusion matrix.

---

## 🧠 Technologies & Tools Used

| Category                | Technology                                |
|------------------------|-------------------------------------------|
| Programming Language   | Python                                    |
| Deep Learning          | TensorFlow, Keras, MobileNet              |
| Machine Learning       | Scikit-learn, XGBoost, Random Forest, MLP |
| Image Processing       | OpenCV, PIL                               |
| Cloud/Hardware         | Google Colab, Local GPU                   |
| Dataset                | PlantVillage (Maize, Mango, Tomato)       |

---

## 🧪 Dataset Details

- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Classes**: Healthy, Diseased (multi-label classification)
- **Images**: 
  - Maize: 2000
  - Tomato: 2000
  - Mango: 437
- **Preprocessing**:
  - Image resizing (90x90 px)
  - Denoising
  - RGB conversion
  - Data Augmentation (rotation, flipping, contrast, blur)

---

## ⚙️ Methodology

### 🔧 Image Preprocessing
- Denoising using CNN-based filters
- Normalization of pixel values (scaled to 0–1)
- Augmentation: flip, rotate, zoom, blur

### 📊 Dataset Split
- Training: 80%
- Testing: 20%
- Labels: `0 = Diseased`, `1 = Healthy`

---

### 📈 Models Implemented

#### 1. **MobileNet (CNN)**
- 28-layer pre-trained CNN on ImageNet
- 3 added layers: GlobalAveragePooling2D, Dropout (20%), Dense (Sigmoid)
- Optimized for mobile/IoT deployment
- Accuracy: **100%**

#### 2. **Multi-Layer Perceptron (MLP)**
- 3 hidden layers (512, 256, 128 neurons)
- Activation: ReLU, Output: Sigmoid
- Early stopping for regularization
- Accuracy: **97%**

#### 3. **Random Forest**
- 100 decision trees
- Handles missing data well
- Accuracy: **96%**

#### 4. **XGBoost**
- Boosted tree ensemble
- High performance and generalization
- Accuracy: **97.25%**

---

## 📊 Evaluation Metrics

| Model        | Accuracy | Precision | Recall | F1 Score |
|--------------|----------|-----------|--------|----------|
| MobileNet    | 100%     | 100%      | 100%   | 100%     |
| MLP          | 97%      | 97%       | 97%    | 97%      |
| Random Forest| 96%      | 96%       | 96%    | 96%      |
| XGBoost      | 97.25%   | 97%       | 97%    | 97%      |

### Confusion Matrix Summary
- **MobileNet**: No false positives/negatives
- **Others**: Minor misclassifications, still high performance

---

## 🔄 Workflow Diagram

```mermaid
flowchart TD
    A[Load Dataset] --> B[Image Preprocessing]
    B --> C[Label Extraction]
    C --> D[Data Splitting (80/20)]
    D --> E[Feature Scaling]
    E --> F[Model Training]
    F --> G1[MobileNet]
    F --> G2[XGBoost]
    F --> G3[Random Forest]
    F --> G4[MLP]
    G1 --> H[Evaluation]
    G2 --> H
    G3 --> H
    G4 --> H
✅ Conclusion
This project demonstrates the effective use of machine learning and deep learning in automating plant leaf disease detection. Among the models compared, MobileNet CNN stands out as the most reliable and practical solution due to its:

High accuracy,

Lightweight design,

Suitability for real-time and mobile deployment.

By leveraging image denoising and dataset augmentation, the detection system enhances robustness even on smaller datasets. This work contributes to building smart farming systems capable of operating in resource-constrained environments, directly benefiting farmers and agricultural stakeholders.

📚 References
Prabira Kumar Sethy et al. - Image Processing for Rice Plant Disease Diagnosis

Laura Falaschetti et al. - CNN-Based Image Detector for Plant Disease

Husin et al. - Automated Plant Recognition

Saleem et al. - Deep Learning for Tomato Wilt Detection

Sharma et al. - CNN Segmentation for Plant Disease

Al Bashish et al. - K-Means and NN for Leaf Disease

Brahimi et al. - Saliency Map Visualization in CNN

📌 Future Work
Expand the dataset with more crop types and real-field data.

Implement mobile-based app for farmers to use in the field.

Integrate real-time treatment suggestions via AI.

Add disease progression monitoring using time-series images.

🚀 How to Run
bash
Copy
Edit
# Clone the repo
git clone https://github.com/yourusername/plant-leaf-disease-detection.git
cd plant-leaf-disease-detection

# Install requirements
pip install -r requirements.txt

# Run training (Jupyter or script)
python train_model.py
🤝 Contributing
Feel free to fork this repository and submit a pull request to improve the dataset, models, or deployment.

📝 License
This project is licensed under the MIT License.

yaml
Copy
Edit

---

Let me know if you want to:
- Customize the tone (technical, academic, casual).
- Generate the `requirements.txt` file.
- Add screenshots or model diagrams.

Ready to help finalize it!
