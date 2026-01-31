import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *
from models.common_preprocessing import *

df = load_dataset("data/heart.csv")
X, y = split_features_target(df)

X_train, X_test, y_train, y_test = train_test_data(X, y)

X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

model = GaussianNB()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("Naive Bayes Metrics")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("MCC:", matthews_corrcoef(y_test, y_pred))

joblib.dump(model, "saved_models/naive_bayes.pkl")
