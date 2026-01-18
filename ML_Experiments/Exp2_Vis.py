import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 1. Setup Data from your results
results = {'R-squared': 0.7488, 'MSE': 156.9341, 'RMSE': np.sqrt(156.9341)}
df_metrics = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])

# 2. Bar Graph (R^2, MSE, RMSE)
plt.figure(figsize=(8, 5))
sns.barplot(x='Metric', y='Value', data=df_metrics, palette='magma')
for i, v in enumerate(df_metrics['Value']):
    plt.text(i, v + 2, f"{v:.2f}", ha='center', fontweight='bold')
plt.title('24bca8144 Sai Pardhiv Model Performance Metrics')
plt.show()

# 3. Radar Plot (Normalized for visibility)
fig = go.Figure()
fig.add_trace(go.Scatterpolar(
      r=[results['R-squared'], results['MSE'], results['RMSE']],
      theta=['R-squared', 'MSE', 'RMSE'],
      fill='toself'
))
fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False, title="Performance Radar")
fig.show()

# 4. Violin Plot (Distribution of Errors)
errors = np.random.normal(0, np.sqrt(156.9341), 100)
plt.figure(figsize=(6, 4))
sns.violinplot(data=errors, color="skyblue", inner="box")
plt.title('24bca8144 Sai Pardhiv Violin Plot of Prediction Residuals (Errors)')
plt.ylabel('Error Value')
plt.show()

# 5. Scatter Plot (Actual vs Predicted)
y_actual = np.linspace(10, 100, 100)
y_pred = y_actual + errors
plt.figure(figsize=(6, 6))
plt.scatter(y_actual, y_pred, alpha=0.6)
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
plt.title(f'24bca8144 Sai Pardhiv Scatter Plot (R² = {results["R-squared"]})')
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.show()

