import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Setup paths
DATA_DIR = Path("telecom")
client = pd.read_csv(DATA_DIR / "Client.csv")
record = pd.read_csv(DATA_DIR / "Record.csv")
df = record.merge(client, on="Customer_ID", how="inner")

# Basic analysis of missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({"missing": missing, "pct": missing_pct}).sort_values("pct", ascending=False)
print("Top missing values:")
print(missing_df.head(20))

# Drop features with more than 50% missing if any
to_drop = missing_df[missing_pct > 50].index.tolist()
print(f"Dropping features with >50% missing: {to_drop}")
df = df.drop(columns=to_drop)

# Feature engineering (reproducing and extending)
df["upgrade_eligible"] = (df["eqpdays"] >= 270).astype(int)
df["bill_shock_risk"] = (df["vceovr_Mean"] >= 20.0).astype(int)
df["severe_usage_drop"] = (df["change_mou"] <= -100.0).astype(int)
df["eqp_tenure_ratio"] = df["eqpdays"] / (df["months"] * 30.0 + 1)
df["is_budget_phone"] = (df["hnd_price"] <= 50.0).astype(int)
df["contract_cliff_window"] = df["months"].between(9, 10).astype(int)

# Target / Features
X = df.drop(columns=["churn", "Customer_ID"])
y = df["churn"]

# Numeric vs Categorical
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Model
lgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1))
])

# Broader hyperparameter grid
lgb_param_dist = {
    'classifier__n_estimators': [200, 500, 1000],
    'classifier__learning_rate': [0.01, 0.03, 0.05, 0.1],
    'classifier__num_leaves': [31, 63, 127],
    'classifier__max_depth': [-1, 7, 10, 15],
    'classifier__feature_fraction': [0.7, 0.8, 0.9],
    'classifier__bagging_fraction': [0.7, 0.8, 0.9],
    'classifier__bagging_freq': [1, 5]
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Starting RandomizedSearchCV for LightGBM...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgb_search = RandomizedSearchCV(
    lgb_pipeline,
    param_distributions=lgb_param_dist,
    n_iter=20,  # Increased from 5
    scoring='accuracy',
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

lgb_search.fit(X_train, y_train)

print(f"Best Accuracy: {lgb_search.best_score_}")
print(f"Best Params: {lgb_search.best_params_}")

y_pred = lgb_search.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
