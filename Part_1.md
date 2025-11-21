# taller-challenge

## 1. Weaknesses
The model fails completely on the minority class (label = 1) and might be because of the imbalance class
- Precision = 0
- Recall = 0
- F1 = 0

The model never predicts the class 1.
Accuracy is misleading because predicting all zeros gives ~74% accuracy.
Poor handling of categorical variables
LabelEncoder is not a good idea because we don't have an order. One hot encoding may work better.

## 2. Where Does the Model Fail Most?
The model fails heavily on return (class 1) predictions
Completely misses actual returns: recall = 0.
Potencial reasons:
    - Logistic regression is too simple for the problem you want to solve.
    - Imbalance problem
    - Features do not capture the interaction between classes

## 3. Is Accuracy the Right Metric?
No because of the imbalance dataset. A better metric might be F1 or recall itself can help to identify the problems in the model.