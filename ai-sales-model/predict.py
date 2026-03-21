
import pickle
import pandas as pd

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load remaining test data
data = pd.read_csv("Advertising.csv")

# Take last 50 rows
test_data = data.tail(50)

X_test = test_data[['TV', 'Radio', 'Newspaper']]

# Predict
predictions = model.predict(X_test)

# Show results
print("Predictions for remaining 50 records:")
print(predictions)
