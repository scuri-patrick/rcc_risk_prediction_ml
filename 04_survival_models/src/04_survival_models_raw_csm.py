import sys

# Add the parent directory to the system path
sys.path.append("../")

import datetime
import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import optuna
import mlflow
import numpy as np
import pandas as pd
from azureml.core import Dataset, Workspace
from lifelines.statistics import logrank_test
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import (
    RFECV,
    SelectKBest,
    SequentialFeatureSelector,
    mutual_info_regression,
)
from sklearn.impute import IterativeImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sksurv.ensemble import (
    ComponentwiseGradientBoostingSurvivalAnalysis,
    ExtraSurvivalTrees,
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
)
from sksurv.functions import StepFunction
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis, IPCRidge
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.tree import SurvivalTree
from sklearn.base import clone
import argparse
import mltable
from uc2_functions import *
import argparse
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore', category=RuntimeWarning)

def load_config(path_config):
    with open(path_config, 'r') as file:
        config = json.load(file)
    models = []
    params = []
    for e in config:
        models.append(e['model'])
        params.append(e['params'])
    return models, params

def load_importances(path_importances, random_state):
    df_importances = pd.read_json(path_importances)
    features_top_t1 = df_importances[(df_importances['random_state'] == random_state) & (df_importances['model'] == "RandomSurvivalForest_selector_T1")]['top_features'].tolist()[0]
    features_top_t0 = df_importances[(df_importances['random_state'] == random_state) & (df_importances['model'] == "RandomSurvivalForest_selector_T0")]['top_features'].tolist()[0]
    return features_top_t1, features_top_t0
    
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Run survival analysis models with specified configuration.')
    parser.add_argument("--RANDOM_STATE", default=42, type=int, help='Random state seed for the simulation.')
    parser.add_argument("--EXPERIMENT_NAME", default="Default_Experiment", type=str, help='Name of the MLflow experiment.')
    parser.add_argument("--DATA_ID", default=None, type=str, help='Identifier for the dataset to use.')
    parser.add_argument("--PATH_IMPORTANCES", default=None, type=str, help='Path to features selected previously using a RSF on the same seed.')
    parser.add_argument("--N_MAX_FEATURES", default=None, type=int, help='Max number of features to use.')
    parser.add_argument("--PATH_CONFIG", default=None, type=str, help='Path to configuration file for pipeline and search space.')
    parser.add_argument("--DIR_MODEL_PKL", default='./models', type=str, help='Directory to save trained model pickles.')
    
    args = parser.parse_args()
    return args

def get_data(data_id):
    tbl = mltable.load(f'azureml:/{data_id}')
    df_ohe = tbl.to_pandas_dataframe()
    return df_ohe

def select_by_collinearity(df_ohe, features_top_t1, features_top_t0, method="spearman", threshold=0.6, top_x=20, log_to_mlflow=True):
    def get_highly_correlated_pairs(df, method, threshold):
        corr_matrix = df.corr(method=method)
        corr_pairs = corr_matrix.abs().stack().reset_index()
        corr_pairs.columns = ['feature1', 'feature2', 'correlation']
        corr_pairs = corr_pairs[corr_pairs['feature1'] != corr_pairs['feature2']]
        highly_correlated_pairs = corr_pairs[corr_pairs['correlation'] > threshold].sort_values(by='correlation', ascending=False)
        return highly_correlated_pairs[['feature1', 'feature2']].values.tolist()

    def filter_features(features, highly_correlated_pairs):
        features_to_drop = []
        for pair in highly_correlated_pairs:
            if pair[0] in features and pair[1] in features:
                if features.index(pair[0]) > features.index(pair[1]):
                    features_to_drop.append(pair[0])
                else:
                    features_to_drop.append(pair[1])
            else:
                raise Exception("Both highly_correlated_pairs out of features")
        return [element for element in features if element not in features_to_drop]

    highly_correlated_pairs_t1 = get_highly_correlated_pairs(df_ohe[features_top_t1], method=method, threshold=threshold)
    features_top_t1_filter = filter_features(features_top_t1, highly_correlated_pairs_t1)
    
    highly_correlated_pairs_t0 = get_highly_correlated_pairs(df_ohe[features_top_t0], method=method, threshold=threshold)
    features_top_t0_filter = filter_features(features_top_t0, highly_correlated_pairs_t0)
    
    def plot_correlation_matrix(features, title, filename):
        corr_matrix = df_ohe[features].corr(method=method)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        fig = plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True,
                    annot_kws={"size": 8}, xticklabels=1, yticklabels=1, mask=mask)
        plt.xticks(rotation=90, ha='center', fontsize=10)
        plt.yticks(fontsize=10)
        plt.title(title, fontsize=14)
        plt.tight_layout()

        try:
            if log_to_mlflow:
                mlflow.log_figure(fig, filename)
                print(f"Figure {filename} logged to mlflow successfully.")
            else:
                plt.show()
        except Exception as e:
            print(f"Error logging figure {filename} to mlflow: {e}")
        
        plt.close()

    features_top_t1_filter = features_top_t1_filter[:top_x]
    features_top_t0_filter = features_top_t0_filter[:top_x]

    # Plot and log the correlation heatmap for final selected features for T1
    plot_correlation_matrix(features_top_t1_filter, f"Correlation Matrix of Final Features (T1)", "correlation_matrix_final_t1.png")
    
    # Plot and log the correlation heatmap for final selected features for T0
    plot_correlation_matrix(features_top_t0_filter, f"Correlation Matrix of Final Features (T0)", "correlation_matrix_final_t0.png")

    return features_top_t1_filter, features_top_t0_filter

def train_test_split_impute(df_ohe, random_state):
    """
    Perform train-test split.
    Impute data (fit-transform on train, transform on test).
    """
    #____________________________________________________________
    # Drop na on target columns
    not_features = ["P_1_id", "death", "csm", "ocm", "ttdeath"]
    df_ohe = df_ohe.dropna(subset=["ttdeath", "death"])
    #____________________________________________________________
    # Train test split
    # List features
    features_all = sorted(set(df_ohe.columns.tolist()) - set(not_features))
    # Train test split
    # Define features and target
    X = df_ohe[features_all]
    y = np.array(
        [(event, time) for event, time in zip(df_ohe["death"], df_ohe["ttdeath"])],
        dtype=[("event", bool), ("time", float)],
    )
    ids = df_ohe["P_1_id"]
    mlflow.log_param(
        "death_perc_5yrs",
        pd.Series(y["event"]).value_counts(sort=True, normalize=True)[True],
    )
    # Split data and IDs into training and testing sets
    (
        X_train_missing,
        X_test_missing,
        y_train,
        y_test,
        ids_train,
        ids_test,
    ) = train_test_split(
        X,
        y,
        ids,
        test_size=0.2,
        stratify=y["event"],
        random_state=random_state,
    )
    del X, y, ids
    #____________________________________________________________
    # Imputation
    # Fit and trasform on train
    X_train = X_train_missing.copy()
    imputer = IterativeImputer(
        max_iter=25, initial_strategy="median", random_state=random_state
    )
    imputer = imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_train = pd.DataFrame(X_train, columns=X_train_missing.columns)
    # Assert
    assert set(X_train.columns) == set(X_train_missing.columns)
    del X_train_missing
    X_test = X_test_missing.copy()
    X_test = imputer.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=X_test_missing.columns)
    # Assert
    assert set(X_test.columns) == set(X_test_missing.columns)
    del X_test_missing
    return X_train, X_test, y_train, y_test

def model_mapper(name, random_state=None):
    """
    Maps a model name to its class instance with the optional random_state parameter.
    
    Args:
    - name (str): The name of the model class.
    - random_state (int, optional): A seed for the random number generator for models that support it.
    
    Returns:
    - An instance of the specified model class.
    """
    if name == "ComponentwiseGradientBoostingSurvivalAnalysis":
        model = ComponentwiseGradientBoostingSurvivalAnalysis(random_state=random_state)
    elif name == "GradientBoostingSurvivalAnalysis":
        model =  GradientBoostingSurvivalAnalysis(random_state=random_state)
    elif name == "RandomSurvivalForest":
        model =  RandomSurvivalForest(random_state=random_state)
    elif name == "ExtraSurvivalTrees":
        model =  ExtraSurvivalTrees(random_state=random_state)
    elif name == "CoxnetSurvivalAnalysis":
        model =  CoxnetSurvivalAnalysis(fit_baseline_model=True)
    elif name == "CoxPHSurvivalAnalysis":
        model =  CoxPHSurvivalAnalysis()
    elif name == "SurvivalTree":
        model =  SurvivalTree(random_state=random_state)
    else:
        raise ValueError(f"Model name '{name}' is not recognized.")
    return model

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="overflow encountered in exp", module="sksurv.linear_model.coxph")
    args = parse_args()
    df_ohe = get_data(data_id=args.DATA_ID)
    mlflow.set_experiment(args.EXPERIMENT_NAME)
    mlflow.start_run(run_name=str(args.RANDOM_STATE))
    features_top_t1, features_top_t0 = load_importances(args.PATH_IMPORTANCES, int(args.RANDOM_STATE))
    mlflow.log_param("n_features_top_t1_unfiltered", len(features_top_t1))
    mlflow.log_param("n_features_top_t0_unfiltered", len(features_top_t0))
    features_top_t1, features_top_t0 = select_by_collinearity(df_ohe=df_ohe,
                                                               features_top_t1=features_top_t1,
                                                               features_top_t0=features_top_t0,
                                                               method="spearman",
                                                               threshold=0.6,
                                                               top_x=20,
                                                              log_to_mlflow=True)
    if len(features_top_t1) > args.N_MAX_FEATURES:
        features_top_t1 = features_top_t1[:args.N_MAX_FEATURES]
    if len(features_top_t0) > args.N_MAX_FEATURES:
        features_top_t0 = features_top_t0[:args.N_MAX_FEATURES]
    mlflow.log_param("n_features_top_t1_collinearity_limited", len(features_top_t1))
    mlflow.log_param("n_features_top_t0_collinearity_limited", len(features_top_t0))
    X_train, X_test, y_train, y_test = train_test_split_impute(df_ohe=df_ohe, random_state=args.RANDOM_STATE)
    models, params = load_config(args.PATH_CONFIG)
    model_instances = [model_mapper(x, args.RANDOM_STATE) for x in models]
    param_grids = params
    assert len(model_instances) == len(param_grids), "The number of model instances and parameter grids must match."
    for model_instance, param_grid in tqdm(zip(model_instances, param_grids)):
        pipeline_skurv(model=model_instance,
                       param_grid=param_grid,
                       k_min=4,
                       X_train=X_train[features_top_t1],
                       y_train=y_train,
                       X_test=X_test[features_top_t1],
                       y_test=y_test,
                       random_state=args.RANDOM_STATE,
                       n_folds=10,
                       tau=60,
                       dir_models=args.DIR_MODEL_PKL,
                       dataset_name="raw",
                       timepoint='T1')
    for model_instance, param_grid in tqdm(zip(model_instances, param_grids)):
        pipeline_skurv(model=model_instance,
                       param_grid=param_grid,
                       k_min=4,
                       X_train=X_train[features_top_t0],
                       y_train=y_train,
                       X_test=X_test[features_top_t0],
                       y_test=y_test,
                       random_state=args.RANDOM_STATE,
                       n_folds=10,
                       tau=60,
                       dir_models=args.DIR_MODEL_PKL,
                       dataset_name="raw",
                       timepoint='T0')
    mlflow.end_run()

if __name__ == "__main__":
    main()
