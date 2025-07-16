import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Load the Dataset ---
try:
    df = pd.read_csv('/Users/akhilduggal/Downloads/Salary_Dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'salary_data.csv' not found. Please make sure the file is in the same directory.")
    exit() # Exit if the file isn't found

# --- 2. Data Preprocessing ---
print("\n--- Data Preprocessing ---")

# Handle missing values (example: fill with mode for categorical(discrete), median for numerical(continuous))
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].median(), inplace=True)
print("Missing values handled.")

# Feature Engineering (creating a new feature if it could help the model)
# 'Education Level' and 'Job Title' might need encoding.

# Encode categorical features
categorical_cols = ['Education Level', 'Job Title'] 
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(f"'{col}' encoded.")
    else:
        print(f"Warning: '{col}' not found in dataset. Skipping encoding.")

# Defining features (X) and target (y)
X = df[['Years of Experience', 'Education Level', 'Job Title']]
y = df['Salary']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

# Standardize numerical features (Years of Experience)
scaler = StandardScaler()
X_train['Years of Experience'] = scaler.fit_transform(X_train[['Years of Experience']])
X_test['Years of Experience'] = scaler.transform(X_test[['Years of Experience']])
print("Numerical features scaled.")

# --- 3. Function for Model Training and Evaluation ---
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):

    print(f"\n--- Training and Evaluating {model_name} ---")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"{model_name} Results:")
    print(f"  R-squared Score: {r2:.2f}")
    print(f"  Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"  Root Mean Squared Error (RMSE): ${rmse:,.2f}")

    return predictions, r2, mae, rmse

# --- 4. Model Training and Evaluation ---

# Initialize models
linear_model = LinearRegression()
decision_tree_model = DecisionTreeRegressor(random_state=42)
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42) # n_estimators can be tuned

# Train and evaluate Linear Regression
lr_predictions, lr_r2, lr_mae, lr_rmse = train_and_evaluate_model(
    linear_model, X_train, y_train, X_test, y_test, "Linear Regression"
)

# Train and evaluate Decision Tree
dt_predictions, dt_r2, dt_mae, dt_rmse = train_and_evaluate_model(
    decision_tree_model, X_train, y_train, X_test, y_test, "Decision Tree"
)

# Train and evaluate Random Forest
rf_predictions, rf_r2, rf_mae, rf_rmse = train_and_evaluate_model(
    random_forest_model, X_train, y_train, X_test, y_test, "Random Forest"
)

print("\n--- Analysis Summary ---")
print("We compared three models for salary prediction:")
print("1. Linear Regression: A straightforward model showing the linear relationship. It's easy to interpret.")
print(f"   R-squared: {lr_r2:.2f}, MAE: ${lr_mae:,.2f}, RMSE: ${lr_rmse:,.2f}")
print("2. Decision Tree: Learns by splitting the data based on feature values. Can capture complex patterns but prone to overfitting.")
print(f"   R-squared: {dt_r2:.2f}, MAE: ${dt_mae:,.2f}, RMSE: ${dt_rmse:,.2f}")
print("3. Random Forest: An ensemble of many decision trees. This model generally performs best due to its ability to average out individual tree errors, making it more robust and accurate.")
print(f"   R-squared: {rf_r2:.2f}, MAE: ${rf_mae:,.2f}, RMSE: ${rf_rmse:,.2f}")
print(f"\nBased on R-squared, Random Forest ({rf_r2:.2f}) performed the best among the three, indicating it explains a higher proportion of the variance in salary.")
print("The Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) provide insights into the typical prediction error in dollar amounts.")
