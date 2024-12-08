import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Create synthetic dataset
np.random.seed(42)
num_samples = 100

data = {
    "Distance_km": np.random.uniform(1, 20, num_samples),
    "Order_Preparation_Time_min": np.random.uniform(5, 30, num_samples),
    "Traffic_Conditions": np.random.randint(1, 6, num_samples),  # Scale 1 to 5
}
data["Delivery_Time_min"] = (
    10
    + 3 * data["Distance_km"]
    + 1.5 * data["Order_Preparation_Time_min"]
    + 5 * data["Traffic_Conditions"]
    + np.random.normal(0, 5, num_samples)  # Adding noise
)

df = pd.DataFrame(data)

# Step 2: Split data into training and testing sets
X = df[["Distance_km", "Order_Preparation_Time_min", "Traffic_Conditions"]]
y = df["Delivery_Time_min"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Save dataset to Excel file
file_path = "food_delivery_dataset.xlsx"
df.to_excel(file_path, index=False)
print(f"Dataset saved to {file_path}")

# Step 6: Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color="blue", label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Fit")
plt.xlabel("Actual Delivery Time (min)")
plt.ylabel("Predicted Delivery Time (min)")
plt.title("Actual vs Predicted Delivery Times")
plt.legend()
plt.grid(True)

# Save plot
plot_path = "delivery_time_plot.png"
plt.savefig(plot_path)
plt.show()

print(f"Plot saved to {plot_path}")
