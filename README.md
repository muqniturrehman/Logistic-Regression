# 🌧️ Logistic Regression for Rain Prediction

## 📚 Project Overview

This project implements **Logistic Regression from scratch** using NumPy to **predict the occurrence of rain** based on environmental features such as temperature, humidity, wind speed, pressure, and cloud cover.

The pipeline includes:

- Data loading and preprocessing
- Min-max feature scaling
- Target variable encoding
- Gradient Descent optimization
- Model training and evaluation
- Accuracy calculation

---

## 🗃️ Dataset

- The dataset is expected to be in a file named `data.csv`.
- Input features:
  - `Temperature`
  - `Humidity`
  - `Wind_Speed`
  - `Cloud_Cover`
  - `Pressure`
- Target variable:
  - `Rain` (categorical: "rain" or "no rain")

---

## 🧪 Steps and Components

### 1. 🔍 Importing Libraries

```python
import numpy as np
import pandas as pd
```

### 2. 📥 Load Dataset

```python
data = pd.read_csv("data.csv")
```

### 3. 🧼 Data Preprocessing

#### ➤ Missing Values
Check for missing data using `data.isna().sum()`.

#### ➤ Feature Scaling
Use **Min-Max Scaling** on:
- Temperature
- Humidity
- Wind Speed
- Cloud Cover
- Pressure

#### ➤ Target Encoding
Encode `Rain` as:
- `1` for `"rain"`
- `0` for `"no rain"`

---

### 4. 🔀 Train-Test Split

Custom `train_test_split(data, ratio)` function used to split the data randomly into training and testing sets.

---

### 5. 🧠 Logistic Regression (from Scratch)

#### ➤ Sigmoid Function
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

#### ➤ Cost Function
Binary Cross-Entropy Loss is used:
```python
def compute_cost(X, y, w):
    h = sigmoid(X @ w)
    return - (y @ np.log(h) + (1 - y) @ np.log(1 - h)) / len(y)
```

#### ➤ Gradient Function
```python
def compute_gradient(X, y, w): 
    h = sigmoid(X @ w)
    return (1 / X.shape[0]) * X.T @ (h - y)
```

#### ➤ Gradient Descent
```python
def gradient_descent(X, y, w_in, cost_fn, grad_fn, alpha, num_iters): 
    w = w_in.copy()
    for i in range(num_iters):
        w -= alpha * grad_fn(X, y, w)
    return w
```

- Learning rate: `alpha = 0.1`
- Iterations: `10,000`

---

### 6. 🤖 Predictions

```python
def predict(X, w): 
    return (sigmoid(X @ w) >= 0.5).astype(int)
```

---

### 7. 🎯 Accuracy Calculation

```python
def compute_accuracy(y_true, y_pred):
    return (np.sum(y_true == y_pred) / len(y_true)) * 100
```

---

## ✅ Output

After training, the script:
- Prints model predictions on the test set
- Computes and displays **accuracy** of the model

Example:
```
Predictions: [0. 1. 0. ...]
Model Accuracy: 87.50%
```

---

## 📂 File Structure

```
├── data.csv
├── logistic_regression.py
└── README.md
```

---

## ⚙️ How to Run

1. Place your dataset file as `data.csv` in the project root.
2. Run the script using:

```bash
python logistic_regression.py
```

Make sure all required libraries (`numpy`, `pandas`) are installed.

---

## 🧠 Key Concepts

- Logistic Regression is a linear classifier that models the probability of a binary outcome.
- The sigmoid function maps predictions to probabilities between 0 and 1.
- Binary Cross Entropy is used to measure the cost.
- Gradient Descent iteratively updates weights to minimize the cost.

---

## 👨‍💻 Author

**Muqnit Ur Rehman**  
**Roll No:** BDSF23M039

---

## 📝 License

This project is open-source and free to use under the MIT License.

