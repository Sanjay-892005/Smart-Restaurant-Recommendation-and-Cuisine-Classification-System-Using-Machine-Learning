import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv(r"C:\Users\Sanjay\Desktop\Task-1\Dataset .csv")
target_column = "Aggregate rating"
X = df.drop(columns=[target_column])
y = df[target_column]
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
lr_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression Results")
print("Mean Squared Error:", mse_lr)
print("R2 Score:", r2_lr)
dt_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(random_state=42))
])

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("\nDecision Tree Results")
print("Mean Squared Error:", mse_dt)
print("R2 Score:", r2_dt)
results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree"],
    "MSE": [mse_lr, mse_dt],
    "R2 Score": [r2_lr, r2_dt]
})

print("\nModel Comparison")
print(results)
dt = dt_model.named_steps["model"]

feature_names = (
    list(numerical_cols) +
    list(
        dt_model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_cols)
    )
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": dt.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features")
print(importance_df.head(10))
print("\nConclusion:")
print("Decision Tree generally performs better for non-linear data.")
print("Important factors influencing ratings include location, votes, price range, and cuisines.")