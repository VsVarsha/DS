import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data = {
    "TV": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    "Sale_Success": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 1 = Sale, 0 = No Sale
}

#Convert to DataFrame
df = pd.DataFrame(data)

#Split into features
X = df[["TV"]]
y = df["Sale_Success"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

#Prediction
unknown_input = [[95]]  # TV budget = $95
predicted_class = model.predict(unknown_input)

#Output Prediction
print(f"Predicted Sale Success for TV budget {unknown_input[0][0]}: {'Yes' if predicted_class[0] == 1 else 'No'}")
