import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import matplotlib.pyplot as plt

csv_path = "C:/Users/HerschelLeonAlexande/Desktop/Weather/weather.csv"
data = pd.read_csv(csv_path, header=0)

data.columns = data.columns.str.strip().str.lower()
print("Spalten in CSV:", list(data.columns))

required_cols = ['temperature', 'humidity', 'wind_speed', 'rainfall', 'condition']
for col in required_cols:
    if col not in data.columns:
        raise ValueError(f"Spalte '{col}' fehlt in der CSV-Datei!")

le = LabelEncoder()
data['condition_encoded'] = le.fit_transform(data['condition'])

X = data[['temperature', 'humidity', 'wind_speed', 'rainfall']]
y_temp = data['temperature']
y_hum = data['humidity']
y_rain = data['rainfall']
y_cond = data['condition_encoded']

X_train, X_test, y_train_temp, y_test_temp = train_test_split(X, y_temp, test_size=0.2, random_state=42)
_, _, y_train_hum, y_test_hum = train_test_split(X, y_hum, test_size=0.2, random_state=42)
_, _, y_train_rain, y_test_rain = train_test_split(X, y_rain, test_size=0.2, random_state=42)
X_train_cond, X_test_cond, y_train_cond, y_test_cond = train_test_split(X, y_cond, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_cond_scaled = scaler.transform(X_train_cond)
X_test_cond_scaled = scaler.transform(X_test_cond)

def train_regressor(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model_temp = train_regressor(X_train_scaled, y_train_temp)
model_hum = train_regressor(X_train_scaled, y_train_hum)
model_rain = train_regressor(X_train_scaled, y_train_rain)

model_cond = RandomForestClassifier(n_estimators=100, random_state=42)
model_cond.fit(X_train_cond_scaled, y_train_cond)

def print_regression_results(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} RMSE: {rmse:.2f}")

print_regression_results("Temperature", y_test_temp, model_temp.predict(X_test_scaled))
print_regression_results("Humidity", y_test_hum, model_hum.predict(X_test_scaled))
print_regression_results("Rainfall", y_test_rain, model_rain.predict(X_test_scaled))

y_pred_cond = model_cond.predict(X_test_cond_scaled)
print("\nCondition Accuracy:", accuracy_score(y_test_cond, y_pred_cond))
print(classification_report(y_test_cond, y_pred_cond, target_names=le.classes_))

importances = model_cond.feature_importances_
features = X.columns
plt.bar(features, importances, color='lightblue')
plt.title("Feature Importance (Condition Prediction)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

new_weather = pd.DataFrame({
    'temperature': [22],
    'humidity': [70],
    'wind_speed': [8],
    'rainfall': [0]
})

new_scaled = scaler.transform(new_weather)
pred_temp = model_temp.predict(new_scaled)[0]
pred_hum = model_hum.predict(new_scaled)[0]
pred_rain = model_rain.predict(new_scaled)[0]
pred_cond = le.inverse_transform(model_cond.predict(new_scaled))[0]

print("\nPredicted Weather for new input:")
print(f"Temperature: {pred_temp:.1f}degC")
print(f"Humidity: {pred_hum:.1f}%")
print(f"Rainfall: {pred_rain:.1f} mm")
print(f"Condition: {pred_cond}")
