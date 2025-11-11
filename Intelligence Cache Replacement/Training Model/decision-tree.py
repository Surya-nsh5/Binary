import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# --- Load dataset ---
df = pd.read_csv('cache_dataset.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# --- Encode categorical data ---
le = LabelEncoder()
df['access_type_encoded'] = le.fit_transform(df['access_type'])

# --- Feature and label selection ---
X = df[['recency_rank', 'access_count', 'last_access_time', 'access_type_encoded']]
y = df['label']

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Train Decision Tree ---
dt_model = DecisionTreeClassifier(
    max_depth=4, min_samples_split=20, min_samples_leaf=10, random_state=50
)
dt_model.fit(X_train, y_train)

# --- Predictions ---
y_pred = dt_model.predict(X_test)

# --- Evaluation ---
print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# --- Save the trained Decision Tree model ---
joblib.dump(dt_model, "Models/decision_tree_cache.pkl")
print("\nModel saved as: Models/decision_tree_cache.pkl")

# # --- Reload and test prediction for verification ---
# loaded_model = joblib.load("Models/decision_tree_cache.pkl")
# sample_pred = loaded_model.predict(X_test[:5])
# print("\nSample Predictions from Reloaded Model:", sample_pred)
