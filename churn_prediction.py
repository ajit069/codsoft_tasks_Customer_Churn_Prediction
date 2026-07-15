import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
df = pd.read_csv("Churn_Modelling.csv")

# Remove unnecessary columns
df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["Geography"] = le.fit_transform(df["Geography"])

# Features and Target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- Logistic Regression ----------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

print("========== Logistic Regression ==========")
print("Accuracy:", accuracy_score(y_test, pred_lr))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred_lr))
print("\nClassification Report:\n", classification_report(y_test, pred_lr))

# ---------------- Random Forest ----------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print("\n========== Random Forest ==========")
print("Accuracy:", accuracy_score(y_test, pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred_rf))
print("\nClassification Report:\n", classification_report(y_test, pred_rf))

# ---------------- Gradient Boosting ----------------
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
pred_gb = gb.predict(X_test)

print("\n========== Gradient Boosting ==========")
print("Accuracy:", accuracy_score(y_test, pred_gb))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred_gb))
print("\nClassification Report:\n", classification_report(y_test, pred_gb))

# ---------------- Compare Models ----------------
accuracies = {
    "Logistic Regression": accuracy_score(y_test, pred_lr),
    "Random Forest": accuracy_score(y_test, pred_rf),
    "Gradient Boosting": accuracy_score(y_test, pred_gb)
}

print("\n========== Model Comparison ==========")
for model, acc in accuracies.items():
    print(f"{model}: {acc:.4f}")

# Plot Accuracy Comparison
plt.figure(figsize=(8,5))
plt.bar(accuracies.keys(), accuracies.values())
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()