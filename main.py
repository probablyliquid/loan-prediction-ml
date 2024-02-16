import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataset = pd.read_csv("loan_data.csv").drop(columns=["Loan_ID"]).dropna()

# Separate features (X) and target variable (y)
X = dataset.drop(columns=["Loan_Status"])
y = dataset["Loan_Status"]

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standard Scaling for input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model definition and training
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(16, input_shape=(X_train.shape[1],), activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Input user details
print("\nLOAN PREDICTOR\n")

userName = input("Enter your name: ")
userGender = input("Enter your Gender (M for Male / F for Female): ").upper()
userGender = 1.0 if userGender == "M" else 2.0

married = input("Enter Y if married, N if not married: ").upper()
married = 1.0 if married == "Y" else 2.0

dependents = float(input("Enter number of dependents: "))
dependents = min(dependents, 3.0)

graduate = input("Do you have a graduate degree? (Y/N): ").upper()
graduate = 1.0 if graduate == "Y" else 2.0

selfEmployed = input("Are you self-employed? (Y/N): ").upper()
selfEmployed = 1.0 if selfEmployed == "Y" else 2.0

income = float(input("Enter your monthly income (ideally 0-7500): "))
coIncome = float(
    input(
        "Enter your co-applicant's monthly income (ideally 0-7500, 0 if not applicable): "
    )
)
amount = float(input("Enter the loan amount (in thousands, ideally from 10-150): "))
term = float(input("Enter the term of the loan (in months in multiples of 12): "))

creditHistory = input("Do you have a credit history? (Y/N): ").upper()
creditHistory = 1.0 if creditHistory == "Y" else 0.0

propertyArea = input(
    "Where is the location of your home? (R for Rural, U for Urban, S for Semiurban): "
).upper()
propertyArea = {"U": 1.0, "S": 2.0, "R": 3.0}.get(propertyArea, 0.0)

# Create a NumPy array with the collected data and scale it
user_data = np.array(
    [
        userGender,
        married,
        dependents,
        graduate,
        selfEmployed,
        income,
        coIncome,
        amount,
        term,
        creditHistory,
        propertyArea,
    ],
    dtype=float,
)
user_data_scaled = scaler.transform(user_data.reshape(1, -1))

# Displaying Prediction:
prediction = model.predict(user_data_scaled)

# The prediction is a 2D array, so we extract the first element
prediction = prediction[0]

if prediction >= 0.5:
    print(f"Dear {userName}, we are glad to tell you that your loan was APPROVED!")
else:
    print(f"Dear {userName}, we are sorry to tell you that your loan was NOT APPROVED.")
