import numpy as np
import pandas as pd
import streamlit as st
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class ModelOptimizer:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def breast_cancer_objective(self, trial):
        model_name = trial.suggest_categorical('model', ['RandomForest', 'SVC', 'XGBoost'])

        if model_name == 'RandomForest':
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

        elif model_name == 'SVC':
            C = trial.suggest_float('C', 1e-3, 1e3, log=True)
            gamma = trial.suggest_float('gamma', 1e-4, 1e-1, log=True)
            clf = SVC(C=C, gamma=gamma)

        else:  # XGBoost
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            clf = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                use_label_encoder=False,
                eval_metric='logloss'
            )

        # Cross-validation
        score = cross_val_score(clf, self.X, self.y, n_jobs=-1, cv=3, scoring='accuracy')
        return score.mean()

    def iris_objective(self, trial):
        classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
        
        if classifier_name == "SVC":
            svc_c = trial.suggest_float("svc_c", 1e-5, 1e2, log=True)
            model = SVC(C=svc_c)
        else:  # RandomForest
            rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 100)
            rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
            model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth)
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        
        score = cross_val_score(pipeline, self.X, self.y, n_jobs=-1, cv=3).mean()
        return score

def main():
    st.title("🎯 Automated Model Selection Dashboard")
    
    # Dataset selection
    dataset_name = st.sidebar.selectbox(
        "Choose Dataset",
        ["Iris", "Breast Cancer"]
    )
    
    # Load selected dataset
    if dataset_name == "Iris":
        X, y = load_iris(return_X_y=True)
        optimizer = ModelOptimizer(X, y)
        objective = optimizer.iris_objective
    else:
        data = load_breast_cancer()
        X = data.data
        y = data.target
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        optimizer = ModelOptimizer(X, y)
        objective = optimizer.breast_cancer_objective

    # Create and run study
    study = optuna.create_study(direction="maximize")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Number of trials selection
    n_trials = st.sidebar.slider("Number of trials", 10, 100, 30)
    
    if st.button("Start Optimization"):
        for i in range(n_trials):
            study.optimize(objective, n_trials=1)
            progress = (i + 1) / n_trials
            progress_bar.progress(progress)
            status_text.text(f"Trial {i + 1}/{n_trials}")
        
        # Display results
        st.subheader("Best Model Parameters:")
        st.json(study.best_params)
        
        st.subheader("Best Cross-Validation Score:")
        st.metric("Accuracy", f"{study.best_value:.3f}")
        
        # Display optimization history
        st.subheader("Optimization History")
        history_df = pd.DataFrame(
            {"Value": [trial.value for trial in study.trials]}
        )
        st.line_chart(history_df)

if __name__ == "__main__":
    main()

