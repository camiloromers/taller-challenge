# taller-challenge

## 1. Dataset Problems
**Issue 1**: The main problem is the class imbalace. Despite many people suggest to use SMOTE or similar techniques to balance the dataset, that is not a good idea because it changes the distribution of the data. It's better to keep it as it is.
**Issue 2**: Try another categorical encoding because with LabelEncoding the order is impossed.
**Issue 3**: Try a better model that can capture model non lineal relationship in the dataset and perhaps make it more interpretable like tree-based models.

## 2. Feature Engineering
A good set of new features might be
- `price * discount_applied` captures effective purchase price
- `tenure / age` loyalty tendency
- `previous_returns_ratio` strong return predictor

## 3. Different Algortithm
`RandomForest` can be a good choice to start modeling the data differently with a simple hyperparameters tunning using a grid.

## Results:
```bash
Best Random Forest
              precision    recall  f1-score   support

           0       0.84      0.42      0.56      1495
           1       0.31      0.76      0.44       505

    accuracy                           0.51      2000
   macro avg       0.57      0.59      0.50      2000
weighted avg       0.70      0.51      0.53      2000

Accuracy: 0.5055
```
Even though the accuracy is lower, the model is predicting class 1. What we expected from this experiment is:

- Recall for class 1 will increase
- Precision for class 1 will increase
- Consequently, the F1-score for class 1 will increase
