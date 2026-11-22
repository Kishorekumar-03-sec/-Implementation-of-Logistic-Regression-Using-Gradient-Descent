# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries and load the placement dataset using Pandas.
2. Remove unnecessary columns and convert categorical data into numerical values.
3. Initialize the model parameters and apply the sigmoid function and gradient descent to train the Logistic Regression model.
4. Predict the placement status of students and calculate the training accuracy.

## Program:


## Program to implement the the Logistic Regression Using Gradient Descent.
## Developed by: Kishorekumar S
## RegisterNumber:  212224040162
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Placement_Data.csv')

print(data.head())

data = data.drop('sl_no', axis=1)
data = data.drop('salary', axis=1)

print(data.head())

data["gender"] = data["gender"].astype('category')
data["ssc_b"] = data["ssc_b"].astype('category')
data["hsc_b"] = data["hsc_b"].astype('category')
data["degree_t"] = data["degree_t"].astype('category')
data["workex"] = data["workex"].astype('category')
data["specialisation"] = data["specialisation"].astype('category')
data["status"] = data["status"].astype('category')
data["hsc_s"] = data["hsc_s"].astype('category')

print(data.dtypes)

data["gender"] = data["gender"].cat.codes
data["ssc_b"] = data["ssc_b"].cat.codes
data["hsc_b"] = data["hsc_b"].cat.codes
data["degree_t"] = data["degree_t"].cat.codes
data["workex"] = data["workex"].cat.codes
data["specialisation"] = data["specialisation"].cat.codes
data["status"] = data["status"].cat.codes
data["hsc_s"] = data["hsc_s"].cat.codes

print(data.head())

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

print(x.shape)
print(y.shape)

theta = np.random.randn(x.shape[1])

print(theta)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10))

def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient

        if (i + 1) % 100 == 0:
            current_loss = loss(theta, X, y)
            print(f"Iteration {i+1}, Loss: {current_loss:.4f}")

    return theta

theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)

print(theta)

def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

y_pred = predict(theta, x)
accuracy = np.mean(y_pred.flatten() == y)

print("Training Accuracy:", accuracy)
print(y_pred[:20])

xnew1 = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
xnew2 = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])

y_prednew1 = predict(theta, xnew1)
y_prednew2 = predict(theta, xnew2)

print(y_prednew1[0])
print(y_prednew2[0])

```

## Output:
<img width="722" height="278" alt="image" src="https://github.com/user-attachments/assets/e0591bd4-4511-4313-b005-4aff9fdfcdc7" />
<img width="741" height="280" alt="image" src="https://github.com/user-attachments/assets/aba45a38-98b2-4501-812d-97fc1c60689c" />
<img width="742" height="302" alt="image" src="https://github.com/user-attachments/assets/d0f8042e-8048-434a-9165-af5c254f0414" />
<img width="732" height="282" alt="image" src="https://github.com/user-attachments/assets/26d1455d-acd5-4f0f-b509-c29329f4739b" />
<img width="735" height="440" alt="image" src="https://github.com/user-attachments/assets/cdbe9107-3608-4866-8faa-b25bea423666" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

