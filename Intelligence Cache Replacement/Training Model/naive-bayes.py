import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# --- Load dataset ---
data = pd.read_csv("cache_dataset.csv")
print("Dataset shape:", data.shape)
print("\nFirst few rows:\n", data.head())

# --- Encode non-numeric columns ---
le = LabelEncoder()
if 'access_type' in data.columns:
    data['access_type'] = le.fit_transform(data['access_type'].astype(str))

# --- Select valid numeric columns ---
# Exclude non-feature or irrelevant fields if present
allowed_cols = ['last_access_time', 'access_count', 'recency_rank', 'access_type', 'cache_item', 'label']
data = data[[col for col in allowed_cols if col in data.columns]]

# --- Split features and target ---
X = data.drop(columns=['label'])
y = data['label']

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train Naive Bayes model ---
model = GaussianNB()
model.fit(X_train, y_train)

# --- Predictions ---
y_pred = model.predict(X_test)

# --- Evaluation ---
print("\n" + "="*50)
print("NAIVE BAYES MODEL PERFORMANCE")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Save model ---
joblib.dump(model, "Models/naive_bayes_cache.pkl")
print("\nModel saved as: Models/naive_bayes_cache.pkl")

# --- Reload and test on few samples ---
loaded_model = joblib.load("Models/naive_bayes_cache.pkl")
print("\nSample predictions:", loaded_model.predict(X_test[:5]))
