# Diabetes Classifier AI Model

This project is a machine learning model that classifies patients as diabetic or non-diabetic based on several health-related features. The model utilizes a neural network built with TensorFlow to predict diabetes from a dataset of medical indicators. The dataset used is the PIMA Indian Diabetes Dataset, available on GitHub.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Visualization](#visualization)
8. [Future Enhancements](#future-enhancements)


## Project Overview
The Diabetes Classifier AI Model is built using Python, TensorFlow, and scikit-learn. The goal of this project is to predict whether a patient has diabetes based on several health-related features. This project was developed to gain insights into the data using data visualization, understand model development processes using machine learning frameworks, and address data imbalance issues to improve the classifier's prediction performance.

The key objectives of this project are:
- Build a model capable of accurately predicting diabetes using medical data.
- Handle data imbalance to improve classification performance.
- Visualize key aspects of the dataset to understand relationships between features.

## Dataset
- **Source**: The dataset used in this project is the PIMA Indian Diabetes Dataset. This dataset is publicly available and can be accessed via the link below.
  - [Dataset Link](https://raw.githubusercontent.com/rishimj/diabetes-classification-model/main/diabetes.csv?token=GHSAT0AAAAAACJPHUMCSMXLGEL25HGHTMJCZJ62W2Q)
  
- **Features**:
  - `Pregnancies`: Number of pregnancies.
  - `Glucose`: Plasma glucose concentration over 2 hours in an oral glucose tolerance test.
  - `BloodPressure`: Diastolic blood pressure (mm Hg).
  - `SkinThickness`: Triceps skinfold thickness (mm).
  - `Insulin`: 2-hour serum insulin (mu U/ml).
  - `BMI`: Body Mass Index.
  - `DiabetesPedigreeFunction`: Diabetes pedigree function (a function that scores likelihood of diabetes based on family history).
  - `Age`: Age of the patient (years).
  
- **Target Variable**:
  - `Outcome`: 1 if the patient has diabetes, 0 otherwise.

### Data Preprocessing
- **Handling Missing Values**: Rows with missing values were removed to ensure clean data.
- **Data Normalization**: The feature columns were standardized using **StandardScaler** to improve model convergence.
- **Balancing the Dataset**: The dataset had an imbalance between diabetic and non-diabetic samples. We used **Random Over Sampling** to increase the representation of diabetic cases to match the number of non-diabetic cases.

## Model Architecture

The model is a **Sequential Neural Network** consisting of:
1. **Input Layer**: Accepts the standardized feature vectors.
2. **Hidden Layers**:
   - Two fully connected layers (Dense layers) with **16 neurons** each and **ReLU** activation functions.
   - These layers help capture non-linear relationships between features.
3. **Output Layer**:
   - A single neuron with a **sigmoid** activation function to classify each input as either `0` (non-diabetic) or `1` (diabetic).

### Hyperparameters
- **Learning Rate**: 0.001
- **Batch Size**: 16
- **Epochs**: 20
- **Loss Function**: Binary Crossentropy, used for measuring the performance in binary classification.
- **Optimizer**: Adam, chosen for its adaptive learning rate and efficiency in training.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rishimj/diabetes-classification-model.git
   cd diabetes-classification-model

## Results
The trained model achieved the following metrics:

- **Training Accuracy**: 76.3%
- **Validation Accuracy**: 73.5%
- **Test Accuracy**: 77.0%

These results demonstrate the model’s capability to classify patients as diabetic or non-diabetic with reasonable accuracy.

## Visualizations
Several visualizations were performed to understand the dataset and model behavior:

- **Feature Histograms**: Generated histograms for each feature, comparing diabetic and non-diabetic distributions.
  - Example: A histogram for **Glucose** shows distinct distributions between diabetic and non-diabetic samples.
- **Training Metrics**: The model's training and validation accuracy were plotted over 20 epochs to ensure that the model converged and did not overfit.

## Future Enhancements

- **Hyperparameter Tuning**: Further tune hyperparameters like learning rate, batch size, and the number of hidden units to improve accuracy and reduce overfitting.
- **Alternative Sampling Techniques**: Experiment with other data balancing techniques such as **SMOTE** to further reduce overfitting risks associated with random oversampling.
- **Feature Engineering**: Explore additional features or interaction terms that could help the model better predict diabetes.
- **Testing with Other Models**: Implement other classification models like **Random Forest**, **Support Vector Machines (SVM)**, or **Gradient Boosting** to compare their performance against the neural network.
- **Cross-Validation**: Introduce **k-fold cross-validation** to evaluate the robustness and reliability of the model.
