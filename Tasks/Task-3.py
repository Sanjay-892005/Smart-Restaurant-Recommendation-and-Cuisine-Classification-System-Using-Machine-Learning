import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv(r"C:/Users/Sanjay/Desktop/Task-3/Dataset .csv")
df.rename(columns={
    'Restaurant Name': 'Restaurant_Name',
    'Cuisines': 'Cuisine',
    'Average Cost for two': 'Price_Range',
    'Aggregate rating': 'Rating',
    'Votes': 'Votes'
}, inplace=True)
df['Cuisine'].fillna('Unknown', inplace=True)
df['Price_Range'].fillna(df['Price_Range'].median(), inplace=True)
df['Rating'].fillna(df['Rating'].mean(), inplace=True)
df['Votes'].fillna(df['Votes'].median(), inplace=True)
label_encoder = LabelEncoder()
df['Cuisine_Label'] = label_encoder.fit_transform(df['Cuisine'])
X = df[['Price_Range', 'Rating', 'Votes']]
y = df['Cuisine_Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
sample_input = pd.DataFrame({
    'Price_Range': [800],
    'Rating': [4.2],
    'Votes': [500]
})

predicted_label = model.predict(sample_input)
predicted_cuisine = label_encoder.inverse_transform(predicted_label)

print("\nPredicted Cuisine:", predicted_cuisine[0])