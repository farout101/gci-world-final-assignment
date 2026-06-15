import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

def create_features(df):
    # Existing features
    df["upgrade_eligible"] = (df["eqpdays"] >= 270).astype(int)
    df["bill_shock_risk"] = (df["vceovr_Mean"] >= 20.0).astype(int)
    df["severe_usage_drop"] = (df["change_mou"] <= -100.0).astype(int)
    
    # New features
    df["mou_per_rev"] = df["mou_Mean"] / (df["rev_Mean"] + 1)
    df["overage_ratio"] = df["ovrrev_Mean"] / (df["rev_Mean"] + 1)
    df["roam_ratio"] = df["roam_Mean"] / (df["rev_Mean"] + 1)
    df["drop_vce_ratio"] = df["drop_vce_Mean"] / (df["plcd_vce_Mean"] + 1)
    df["complete_vce_ratio"] = df["comp_vce_Mean"] / (df["plcd_vce_Mean"] + 1)
    
    # Trend features
    df["mou_change_ratio"] = df["change_mou"] / (df["avgmou"] + 1)
    df["rev_change_ratio"] = df["change_rev"] / (df["avgrev"] + 1)
    
    # Account features
    df["avg_rev_per_month"] = df["totrev"] / (df["months"] + 1)
    df["avg_mou_per_month"] = df["totmou"] / (df["months"] + 1)
    
    return df

def main():
    DATA_DIR = Path("telecom")
    client = pd.read_csv(DATA_DIR / "Client.csv")
    record = pd.read_csv(DATA_DIR / "Record.csv")
    df = record.merge(client, on="Customer_ID", how="inner")
    
    df = create_features(df)
    
    # Drop Customer_ID
    X = df.drop(columns=["churn", "Customer_ID"])
    y = df["churn"]
    
    # Numeric and Categorical columns
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
    
    # Baseline Model with LightGBM
    lgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ))
    ])
    
    print("Evaluating baseline model with new features...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(lgb_pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

if __name__ == "__main__":
    main()
