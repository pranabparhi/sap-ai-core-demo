import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle

# Load data...
data = pd.read_csv("Advertising.csv")

# Split data
train = data.iloc[:130]
test = data.iloc[130:150]

# Features
X_train = train[["TV","Radio","Newspaper"]]
y_train = train["Actual Sales"]

X_test = test[["TV","Radio","Newspaper"]]
y_test = test["Actual Sales"]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Save model
with open("model.pkl","wb") as f:
    pickle.dump(model,f)

# Graph: Actual vs Predicted
plt.plot(y_test.values,label="Actual")
plt.plot(predictions,label="Predicted")
plt.legend()
plt.title("Test Data: Actual vs Predicted Sales")
print("Actual:", list(y_test.values))
print("Predicted:", list(predictions))

from sklearn.metrics import r2_score, mean_squared_error
print("R2:", r2_score(y_test, predictions))
print("RMSE:", mean_squared_error(y_test, predictions, squared=False))

print("Training complete")
