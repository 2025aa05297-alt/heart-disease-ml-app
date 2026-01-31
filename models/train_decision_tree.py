import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
from models.common_preprocessing import *

# Load data
df = load_dataset("data/heart.csv")
X, y = split_features_target(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_data(X, y)

# Scaling (kept for pipeline consistency)
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

# Regularized Decision Tree (ANTI-OVERFITTING)
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=10,
    min_samples_split=20,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
print("Decision Tree Metrics (Regularized)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("MCC:", matthews_corrcoef(y_test, y_pred))

# Save model
joblib.dump(model, "saved_models/decision_tree.pkl")
