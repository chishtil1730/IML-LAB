import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
# Replace 'your_file.csv' with the actual path to your file
df = pd.read_csv('../framingham_cleaned.csv')

# 2. Data Preprocessing
# Handling missing values (common in health datasets)
# We will drop rows with missing values, or you could use df.fillna()
df = df.dropna()

# 3. Define Features (X) and Target (y)
X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

# 4. Split the data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Feature Scaling
# Logistic Regression performs better when numerical features are on the same scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Initialize and Train the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 7. Make Predictions
y_pred = model.predict(X_test_scaled)

# 8. Calculate Metrics
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
}

# --- Visualizations ---

# Plot 1: Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot 2: Performance Metrics Bar Chart
plt.figure(figsize=(8, 6))
# Updated line to avoid the FutureWarning
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), hue=list(metrics.keys()), palette='magma', legend=False)
plt.ylim(0, 1)
plt.title('Logistic Regression Performance Metrics')
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.ylabel('Score')
plt.show()

# Print detailed report
print("Model Performance Summary:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")