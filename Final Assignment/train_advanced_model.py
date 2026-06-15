import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
import joblib

warnings.filterwarnings("ignore")

def load_data():
    DATA_DIR = Path("telecom")
    client = pd.read_csv(DATA_DIR / "Client.csv")
    record = pd.read_csv(DATA_DIR / "Record.csv")
    df = record.merge(client, on="Customer_ID", how="inner")
    return df

def feature_engineering(df):
    # Ratios and Interactions
    df["mou_per_rev"] = df["mou_Mean"] / (df["rev_Mean"] + 1)
    df["overage_rev_ratio"] = df["ovrrev_Mean"] / (df["rev_Mean"] + 1)
    df["roam_rev_ratio"] = df["roam_Mean"] / (df["rev_Mean"] + 1)
    df["drop_vce_ratio"] = df["drop_vce_Mean"] / (df["plcd_vce_Mean"] + 1)
    df["complete_vce_ratio"] = df["comp_vce_Mean"] / (df["plcd_vce_Mean"] + 1)
    
    # Tenure based features
    df["mou_per_tenure"] = df["totmou"] / (df["months"] + 1)
    df["rev_per_tenure"] = df["totrev"] / (df["months"] + 1)
    df["calls_per_tenure"] = df["totcalls"] / (df["months"] + 1)
    
    # Equipment age features
    df["eqp_tenure_ratio"] = df["eqpdays"] / (df["months"] * 30 + 1)
    df["upgrade_eligible"] = (df["eqpdays"] >= 360).astype(int)
    
    # Change trend features
    df["mou_trend"] = df["change_mou"] / (df["avgmou"] + 1)
    df["rev_trend"] = df["change_rev"] / (df["avgrev"] + 1)
    
    # Bill shock risk
    df["bill_shock_risk"] = (df["ovrrev_Mean"] > 10).astype(int)
    
    # Data vs Voice
    df["data_voice_ratio"] = df["plcd_dat_Mean"] / (df["plcd_vce_Mean"] + 1)
    
    return df

def main():
    print("Loading data...")
    df = load_data()
    
    print("Performing feature engineering...")
    df = feature_engineering(df)
    
    # Target and Features
    X = df.drop(columns=["churn", "Customer_ID"])
    y = df["churn"]
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Preprocessing pipelines
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
    
    # Model configuration
    # Using more optimized parameters for LightGBM
    lgbm_params = {
        'n_estimators': 1000,
        'learning_rate': 0.02,
        'num_leaves': 63,
        'max_depth': 12,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(**lgbm_params))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    print("\nTop 20 Features:")
    # Get feature names from preprocessor
    ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
    feature_names = numeric_cols + cat_feature_names
    
    importances = model.named_steps['classifier'].feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print(feat_imp.head(20))
    
    # Save model
    MODEL_DIR = Path("Final Assignment/models")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "advanced_churn_model.pkl")
    print(f"Model saved to {MODEL_DIR / 'advanced_churn_model.pkl'}")

if __name__ == "__main__":
    main()
