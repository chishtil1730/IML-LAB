import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv('../framingham_cleaned.csv')

# 2. Data Preprocessing
df = df.dropna()

# 3. Define Features (X) and Target (y)
X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Initialize and Train the Linear Regression Model
lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)

# 7. Make Predictions and Convert to Binary (Threshold 0.5)
y_pred_cont = lin_model.predict(X_test_scaled)
y_pred_bin = (y_pred_cont >= 0.5).astype(int)

# 8. Calculate Metrics
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_bin),
    'Precision': precision_score(y_test, y_pred_bin),
    'Recall': recall_score(y_test, y_pred_bin),
    'F1-score': f1_score(y_test, y_pred_bin)
}

# --- Visualizations ---

# Plot 1: Confusion Matrix
plt.figure(figsize=(6, 5))
cm_l = confusion_matrix(y_test, y_pred_bin)
sns.heatmap(cm_l, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Confusion Matrix - Linear Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot 2: Performance Metrics Bar Chart
plt.figure(figsize=(8, 6))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), hue=list(metrics.keys()), palette='viridis', legend=False)
plt.ylim(0, 1)
plt.title('Performance Metrics - Linear Regression')
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.ylabel('Score')
plt.show()

# Print detailed report
print("Linear Regression Model Performance Summary:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")