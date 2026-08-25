# Churn and Salary Prediction using ANN 

A complete **Customer Churn and Salary Prediction** project built as part of my learning journey into **Artificial Neural Networks (ANN), Deep Learning, and Machine Learning deployment**.

This is a relatively small and simple dataset project. The primary objective of this project was not to achieve the highest possible accuracy or build a production-ready prediction system. Instead, my main goal was to understand how Artificial Neural Networks (ANNs) work in practice — from data preprocessing and feature encoding to building networks, training them, evaluating them, making predictions, and finally deploying them as a web application.

This project helped me bridge the gap between the theoretical concepts I was learning about ANN and their practical implementation in a complete end-to-end project.

---

## 📌 Project Overview

The Streamlit application contains two prediction tasks trained on customer data:

- **Customer Churn Prediction:** a binary classification model that predicts whether a customer is likely to leave.
- **Salary Prediction:** a regression model that estimates a customer's annual salary.

Users can switch between the two tasks from the application sidebar and run inference using the same customer profile information.

### Customer Churn Prediction

Customer churn occurs when a customer stops using a company's products or services.

The objective of this project is to predict whether a customer is likely to churn based on different customer attributes such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card status
- Active Member status
- Estimated Salary

The problem is treated as a **binary classification problem**:

```text
0 → Customer will not churn
1 → Customer will churn
```

The trained ANN produces a probability between `0` and `1`. A threshold of `0.5` is used to make the final classification.

### Salary Prediction

The salary task is treated as a **regression problem**. The trained ANN predicts a continuous annual salary value based on customer demographics, financial information, activity, and churn status.

---

# 🎯 Project Goal

My goal with this project was not simply to train neural networks.

I wanted to understand the complete workflow:

```text
Raw Data
   ↓
Data Preprocessing
   ↓
Encoding
   ↓
Feature Scaling
   ↓
Train/Test Split
   ↓
Artificial Neural Network
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Prediction
   ↓
Streamlit Application with Churn and Salary Modes
   ↓
Cloud Deployment
```

This project therefore represents an important milestone in my journey of learning **ANN and Deep Learning**.

---

# 🧠 My ANN Learning Journey

While learning Artificial Neural Networks, I came across many concepts that initially seemed difficult:

- Neurons
- Weights
- Biases
- Activation Functions
- Forward Propagation
- Backpropagation
- Loss Functions
- Optimizers
- Epochs
- Batches
- Validation
- Overfitting
- Early Stopping

Instead of learning these concepts only theoretically, I wanted to apply them to a real-world problem.

This Churn and Salary Prediction project became my way of connecting the theory with practical implementation.

---

# 📊 Dataset Features

| Feature | Description |
|---|---|
| `CreditScore` | Customer's credit score |
| `Geography` | Customer's country/region |
| `Gender` | Customer's gender |
| `Age` | Customer's age |
| `Tenure` | Number of years the customer has been with the company |
| `Balance` | Customer's account balance |
| `NumOfProducts` | Number of products used by the customer |
| `HasCrCard` | Whether the customer has a credit card |
| `IsActiveMember` | Whether the customer is an active member |
| `EstimatedSalary` | Estimated customer salary |
| `Exited` | Target variable indicating churn |

---

# 🔄 Data Preprocessing

Before feeding the data into the ANN, I performed several preprocessing steps.

## Label Encoding

Categorical gender values were converted into numerical values.

```text
Female → 0
Male   → 1
```

using `LabelEncoder`.

## One-Hot Encoding

Geography contains multiple categories:

```text
France
Germany
Spain
```

I used One-Hot Encoding:

```text
France  → [1, 0, 0]
Germany → [0, 1, 0]
Spain   → [0, 0, 1]
```

This avoids introducing an artificial numerical relationship between the categories.

## Feature Scaling

The numerical features have very different ranges.

```text
Age          → 18–92
CreditScore  → 300–850
Balance      → 0–200000+
Salary       → 0–200000+
```

Therefore, I used `StandardScaler` to standardize the numerical features before feeding them into the neural network.

---

# 🧠 Artificial Neural Network

The model was built using **TensorFlow/Keras**.

The general architecture is:

```text
Input Layer
     ↓
Dense Layer + ReLU
     ↓
Dense Layer + ReLU
     ↓
Output Layer + Sigmoid
```

The hidden layers use **ReLU** activation.

The output layer uses **Sigmoid**, because this is a binary classification problem.

The sigmoid function produces a value between `0` and `1`, which can be interpreted as the predicted probability of churn.

---

# ⚡ Activation Functions

### ReLU

```text
ReLU(x) = max(0, x)
```

It is used in the hidden layers to introduce non-linearity into the neural network.

### Sigmoid

The sigmoid function converts the final output into a value between `0` and `1`.

For example:

```text
Prediction = 0.12
```

indicates a relatively low predicted probability of churn.

While:

```text
Prediction = 0.89
```

indicates a relatively high predicted probability of churn.

---

# 📉 Loss Function

Because this is a binary classification problem, I used:

```python
loss="binary_crossentropy"
```

Binary Cross-Entropy measures the difference between the actual target and the predicted probability.

The basic learning process is:

```text
Input
 ↓
Prediction
 ↓
Calculate Loss
 ↓
Backpropagation
 ↓
Update Weights
 ↓
Repeat
```

---

# ⚙️ Optimizer

An optimizer is responsible for updating the neural network's weights during training.

Through repeated iterations, the optimizer tries to minimize the loss and improve the model's predictions.

---

# 🔁 Epochs and Batches

During training, the dataset is processed over multiple **epochs**.

One epoch represents one complete pass through the training dataset.

The dataset can also be divided into smaller **batches**, allowing the neural network to process the data in manageable groups.

This helped me understand what happens behind:

```python
model.fit()
```

rather than treating model training as a black box.

---

# 🛑 Early Stopping

I also experimented with **Early Stopping** to reduce overfitting.

The basic idea is:

```text
Training Loss
      ↓
    Decreasing

Validation Loss
      ↓
    Improving
      ↓
Stops improving
      ↓
Training stops
```

This prevents the model from unnecessarily continuing training once validation performance stops improving.

---

# 💾 Model Saving

After training, I saved the trained classification and regression ANNs using the Keras format:

```text
model.keras
```

I also saved the preprocessing objects so that new input data could be transformed exactly the same way as the training data.

```text
model.keras
regression_model.keras
lebel_encoder_gender.pkl
onehot_encoder_geo.pkl
scalar.pkl
regression_label_encoder_gender.pkl
regression_one_hot_encoder.pkl
regression_scalar.pkl
```

This is extremely important because the preprocessing during prediction must match the preprocessing used during training.

---

# 🔮 Prediction Pipelines

### Churn classification

For a new customer, the churn mode follows this process:

```text
Customer Input
      ↓
Gender Encoding
      ↓
Geography One-Hot Encoding
      ↓
Feature Scaling
      ↓
ANN Model
      ↓
Churn Probability
      ↓
Classification
```

For example:

```text
Prediction = 0.84
```

With a threshold of `0.5`:

```text
0.84 >= 0.5
```

Therefore:

```text
Customer will churn
```

### Salary regression

The salary mode follows a similar preprocessing pipeline, then returns a continuous value:

```text
Customer Input
   ↓
Gender Encoding
   ↓
Geography One-Hot Encoding
   ↓
Feature Scaling
   ↓
ANN Regression Model
   ↓
Estimated Annual Salary
```

---

# 🌐 Streamlit Web Application

After training the model, I wanted to make it usable through a proper interface instead of running predictions only from a Jupyter Notebook.

I built a web application using **Streamlit**.

The application allows users to select either **Churn Prediction** or **Salary Prediction**, then enter:

- Geography
- Gender
- Age
- Credit Score
- Balance
- Estimated Salary
- Tenure
- Number of Products
- Credit Card status
- Active Member status

Then the user can click the inference button to get either a churn probability and risk assessment or an estimated annual salary.

---

# ☁️ Deployment

I also deployed the application using **Streamlit Community Cloud**.

This introduced another important part of my learning journey: deployment and dependency management.

🌐 [View the deployed Churn and Salary Prediction App](https://churnai-6fmszu7jzgkpfbtbovfbmm.streamlit.app/)
---

# 🛠️ Technologies Used

## Programming Language

- Python

## Machine Learning / Deep Learning

- TensorFlow
- Keras
- Scikit-learn

## Data Processing

- Pandas
- NumPy

## Web Application

- Streamlit

## Model Persistence

- Keras `.keras`
- Pickle `.pkl`

## Development & Deployment

- Jupyter Notebook
- VS Code
- Git
- GitHub
- Streamlit Community Cloud

---

# 📁 Project Structure

```text
Churn.ai/
│
├── app.py
├── model.keras
├── regression_model.keras
│
├── lebel_encoder_gender.pkl
├── onehot_encoder_geo.pkl
├── scalar.pkl
├── regression_label_encoder_gender.pkl
├── regression_one_hot_encoder.pkl
├── regression_scalar.pkl
│
├── requirements.txt
│
└── README.md
```

---

# 📦 Requirements

```text
streamlit
tensorflow
pandas
numpy
scikit-learn
```

Install them locally using:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application Locally

Clone the repository:

```bash
git clone <your-repository-url>
```

Move into the project:

```bash
cd Churn.ai
```

Create/activate your virtual environment if required.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

---

# 📈 What I Learned

This project taught me much more than just how to train an ANN.

## Machine Learning

- Binary classification
- Regression
- Train/test splitting
- Feature preprocessing
- Feature scaling
- Model evaluation
- Prediction probabilities
- Continuous value prediction

## Artificial Neural Networks

- Neural network architecture
- Dense layers
- Weights and biases
- ReLU
- Sigmoid
- Forward propagation
- Backpropagation
- Binary Cross-Entropy
- Optimizers
- Epochs
- Batches
- Validation
- Early stopping
- Overfitting

## Python & Data Science

- Pandas
- NumPy
- Scikit-learn
- DataFrames
- Encoders
- Sparse matrices
- Model preprocessing

## Deployment

- Streamlit
- GitHub
- `requirements.txt`
- Python environments
- Dependency management
- Model serialization
- Cloud deployment

---

# ❤️ My Learning Journey

This project represents an important step in my journey of learning Artificial Neural Networks and applying them to both classification and regression problems.

When I started, concepts such as:

```text
Neurons
Weights
Biases
Activation Functions
Backpropagation
Loss
Optimizers
```

were mostly theoretical concepts.

Through this project, I started connecting those concepts to an actual working system.

I went from:

```text
Learning ANN
     ↓
Understanding the theory
     ↓
Preparing real-world data
     ↓
Building the ANN
     ↓
Training the model
     ↓
Evaluating the model
     ↓
Saving the model
     ↓
Building a Streamlit UI
     ↓
Deploying the application
```

This project therefore isn't just a Customer Churn Prediction project for me.

It now also includes salary regression, making it a broader practical step toward understanding Deep Learning and building end-to-end AI applications.

---

## ⭐ Final Thoughts

> **This project is a representation of my learning journey with Artificial Neural Networks. It started with understanding the basic theory of neural networks and gradually evolved into building an end-to-end application for both churn classification and salary regression. Every error, debugging session, preprocessing problem, model experiment, and deployment issue became part of the learning process. This project is not the destination of my AI journey — it is one of the first steps.**
