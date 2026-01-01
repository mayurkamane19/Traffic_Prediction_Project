import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("../data/traffic_weather.csv")
df["date_time"] = pd.to_datetime(df["date_time"])
df = df.sort_values("date_time")

# Feature engineering
df["hour"] = df["date_time"].dt.hour
df["day"] = df["date_time"].dt.dayofweek
df["month"] = df["date_time"].dt.month
df["lag_1"] = df["traffic_volume"].shift(1)
df["lag_24"] = df["traffic_volume"].shift(1)

df.dropna(inplace=True)

features = [
    "temp","rain","snow","clouds",
    "hour","day","month","lag_1","lag_24"
]

X = df[features]
y = df["traffic_volume"]

# Train-test split (time based)
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Save model
with open("traffic_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save plot
plt.figure(figsize=(10,4))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Traffic Prediction")
plt.tight_layout()
plt.savefig("../app/static/plot.png")

print("✅ Model trained + plot saved")
