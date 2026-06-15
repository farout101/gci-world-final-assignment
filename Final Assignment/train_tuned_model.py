import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    # Usage Ratios
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
    
    # Voice vs Data Usage
    df["data_usage_total"] = df["plcd_dat_Mean"] + df["comp_dat_Mean"]
    df["voice_usage_total"] = df["plcd_vce_Mean"] + df["comp_vce_Mean"]
    df["data_voice_usage_ratio"] = df["data_usage_total"] / (df["voice_usage_total"] + 1)
    
    # Customer Service contact
    df["custcare_per_mou"] = df["custcare_Mean"] / (df["mou_Mean"] + 1)
    
    return df

def main():
    print("Loading and Merging Data...")
    df = load_data()
    
    print("Feature Engineering...")
    df = feature_engineering(df)
    
    X = df.drop(columns=["churn", "Customer_ID"])
    y = df["churn"]
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
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
    
    # Hyperparameter Grid
    param_dist = {
        'classifier__n_estimators': [500, 1000],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__num_leaves': [31, 63, 127],
        'classifier__max_depth': [-1, 10, 20],
        'classifier__feature_fraction': [0.6, 0.8, 1.0],
        'classifier__bagging_fraction': [0.6, 0.8, 1.0],
        'classifier__reg_alpha': [0, 0.1, 1.0],
        'classifier__reg_lambda': [0, 0.1, 1.0]
    }
    
    lgbm_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1))
    ])
    
    print("Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Starting Randomized Search...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        lgbm_pipeline,
        param_distributions=param_dist,
        n_iter=10,
        scoring='accuracy',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    print(f"Best Parameters: {search.best_params_}")
    print(f"Best Train Accuracy: {search.best_score_:.4f}")
    
    best_model = search.best_estimator_
    
    print("Evaluating on Test Set...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] # type: ignore
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test ROC AUC: {test_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save Model
    MODEL_DIR = Path("Final Assignment/models")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "final_churn_model.pkl")
    print(f"Final model saved to {MODEL_DIR / 'final_churn_model.pkl'}")

if __name__ == "__main__":
    main()
