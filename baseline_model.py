"""
Baseline Model - Simple Logistic Regression
Use this as your starting point
"""
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
train = pd.read_csv('ecommerce_returns_train.csv')
test = pd.read_csv('ecommerce_returns_test.csv')

def preprocess(df):
    """Simple preprocessing pipeline"""
    df_processed = df.copy()
    
    # Encode categorical: product_category
    le_category = LabelEncoder()
    df_processed['product_category_encoded'] = le_category.fit_transform(
        df_processed['product_category']
    )
    
    # Handle missing sizes (Fashion items only have sizes)
    if df_processed['size_purchased'].notna().any():
        most_common_size = df_processed['size_purchased'].mode()[0]
        df_processed['size_purchased'].fillna(most_common_size, inplace=True)
        
        le_size = LabelEncoder()
        df_processed['size_encoded'] = le_size.fit_transform(
            df_processed['size_purchased']
        )
    
    # Feature selection
    feature_cols = [
        'customer_age', 'customer_tenure_days', 'product_category_encoded',
        'product_price', 'days_since_last_purchase', 'previous_returns',
        'product_rating', 'size_encoded', 'discount_applied'
    ]
    
    X = df_processed[feature_cols]
    y = df_processed['is_return']
    
    return X, y

# Prepare data
X_train, y_train = preprocess(train)
X_test, y_test = preprocess(test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train baseline model
baseline_model = LogisticRegression(random_state=42, max_iter=1000)
baseline_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = baseline_model.predict(X_test_scaled)

# Basic evaluation
print("Baseline Model Performance")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save artifacts
joblib.dump(baseline_model, 'baseline_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n" + "=" * 50)
print("YOUR TASK: Evaluate thoroughly and improve this baseline")
print("=" * 50)


def compute_financial_impact(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    
    impact = 0
    for yt, yp in zip(y_true, y_pred):
        if yp == 1 and yt == 1:   # TP
            impact += 15
        elif yp == 1 and yt == 0: # FP
            impact -= 3
        elif yp == 0 and yt == 1: # FN
            impact -= 18    
    return impact

impact = compute_financial_impact(y_test, y_pred)

thresholds = np.linspace(0.05, 0.95, 10)
results = []

for t in thresholds:
    impact = compute_financial_impact(y_test, y_pred, threshold=t)
    results.append((t, impact))

df = pd.DataFrame(results, columns=['threshold', 'total_impact'])
df.sort_values('total_impact', ascending=False).head()

## Modeling part with new tree-based model

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score,  make_scorer, f1_score, recall_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV


def feat_eng(df, fit_encoders=False, encoders=None):
    df = df.copy()

    if fit_encoders:
        encoders = {}

        le_cat = LabelEncoder()
        df["product_category_encoded"] = le_cat.fit_transform(df["product_category"])
        encoders["cat"] = le_cat

        le_size = LabelEncoder()
        df["size_purchased"].fillna(df["size_purchased"].mode()[0], inplace=True)
        df["size_encoded"] = le_size.fit_transform(df["size_purchased"])
        encoders["size"] = le_size
    else:
        df["product_category_encoded"] = encoders["cat"].transform(df["product_category"])
        df["size_purchased"].fillna(df["size_purchased"].mode()[0], inplace=True)
        df["size_encoded"] = encoders["size"].transform(df["size_purchased"])

    # New features
    df["effective_price"] = df["product_price"] * (1 - df["discount_applied"])
    df["age_tenure_ratio"] = df["customer_tenure_days"] / (df["customer_age"] + 1)
    df["previous_returns_ratio"] = df["previous_returns"] / (df["customer_tenure_days"] + 1)

    feature_cols = [
        "customer_age","customer_tenure_days","product_category_encoded",
        "product_price","effective_price","age_tenure_ratio",
        "days_since_last_purchase","previous_returns","previous_returns_ratio",
        "product_rating","size_encoded","discount_applied"
    ]

    X = df[feature_cols]
    y = df["is_return"]

    return X, y, encoders


# Fit encoders
X_train, y_train, encoders = feat_eng(train, fit_encoders=True)
X_test, y_test, _ = feat_eng(test, fit_encoders=False, encoders=encoders)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# rf = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=12,
#     class_weight="balanced",
#     random_state=42
# )
# rf.fit(X_train, y_train)
# pred_rf = rf.predict(X_test)

# print(classification_report(y_test, pred_rf))
# print("Accuracy:", accuracy_score(y_test, pred_rf))

param_dist = {
    "n_estimators": [150, 300, 500, 800],
    "max_depth": [6, 8, 10, 12, 15, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
    "class_weight": ["balanced"]  # do not change this
}

scoring = {
    "recall_class_1": make_scorer(recall_score, pos_label=1),
    "f1_class_1": make_scorer(f1_score, pos_label=1),
}

rf_base = RandomForestClassifier(random_state=42)

rf_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=40,
    scoring=scoring,
    refit="f1_class_1",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_
pred_rf_tuned = best_rf.predict(X_test)

print("Best Random Forest")
print(classification_report(y_test, pred_rf_tuned))
print("Accuracy:", accuracy_score(y_test, pred_rf_tuned))
