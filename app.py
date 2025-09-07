# Step 1: Libraries import karna
import pandas as pd    # data read & handle karne ke liye
import numpy as np     # maths operations ke liye
from sklearn.model_selection import train_test_split   # data split karne ke liye
from sklearn.preprocessing import StandardScaler       # data normalize karne ke liye
from sklearn.linear_model import LogisticRegression    # ML model
from sklearn.metrics import accuracy_score             # accuracy check

# Step 2: Dataset load karna
data = pd.read_csv("diabetes.csv")   # apne project folder ka dataset
print("Dataset Loaded ✅")
print(data.head())   # pehle 5 rows dekhne ke liye

# Step 3: Features (X) aur Label (y) alag karna
X = data.drop("Outcome", axis=1)   # inputs (Age, Glucose, BMI, etc.)
y = data["Outcome"]                # output (0 = healthy, 1 = diabetic)

# Step 4: Train/Test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scaling (sab values ek level pe lana)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Model banana
model = LogisticRegression()
model.fit(X_train, y_train)   # training

# Step 7: Prediction karna
y_pred = model.predict(X_test)

# Step 8: Accuracy check karna
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Step 9: New patient prediction
# 8 values dene honge (same order me)
new_patient = np.array([[3, 120, 70, 25, 80, 28.5, 0.5, 40]])   # sample input

new_patient = scaler.transform(new_patient)                # normalize
prediction = model.predict(new_patient)

if prediction[0] == 1:
    print("Prediction: Diabetic 🩸")
else:
    print("Prediction: Healthy ✅")
