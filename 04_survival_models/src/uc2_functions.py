import datetime
import json
import os
import warnings
from collections import Counter
import re

from scipy import stats
import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from azureml.core import Dataset, Workspace
from lifelines.statistics import logrank_test
from scipy.stats import wilcoxon
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import (RFECV, SelectKBest,
                                       SequentialFeatureSelector,
                                       mutual_info_regression)
from sklearn.impute import IterativeImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.model_selection import (GridSearchCV, KFold, ParameterGrid,
                                     RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sksurv.ensemble import (ComponentwiseGradientBoostingSurvivalAnalysis,
                             ExtraSurvivalTrees, GradientBoostingSurvivalAnalysis,
                             RandomSurvivalForest)
from sksurv.functions import StepFunction
from sksurv.linear_model import (CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis,
                                 IPCRidge)
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                            cumulative_dynamic_auc, integrated_brier_score)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.tree import SurvivalTree
import seaborn as sns
import joblib
from sklearn.base import clone
import tempfile
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from tqdm import tqdm
import warnings
from matplotlib.ticker import MultipleLocator
from pandas.api.types import CategoricalDtype
warnings.filterwarnings('ignore', category=RuntimeWarning)

class IrrelevantFeatures:
    def __init__(self, columns: list):
        self.columns = columns

    def __t2(self):
        l = [x for x in self.columns if x.startswith("FUP")]
        print(len(l), "columns at t2")
        return l

    def __id(self):
        l = [x for x in self.columns if "_id" in x]
        print(len(l), "columns with id")
        return l

    def __dates(self):
        suffixes_to_remove = ["Date", "DateStart", "DateEnd", "birth"]
        l = [
            item
            for item in self.columns
            if any(item.endswith(suffix) for suffix in suffixes_to_remove)
        ]
        print(len(l), "columns with dates")
        return l

    def __units(self):
        l = [item for item in self.columns if item.endswith("Um")]
        print(len(l), "columns with units of measure")
        return l

    def __notes(self):
        l = [item for item in self.columns if item.endswith("note")]
        print(len(l), "columns with clinical notes")
        return l

    def spotall(self):
        l = self.__t2() + self.__id() + self.__dates() + self.__units() + self.__notes()
        return l


class UnivariateFeatureSelector:
    def __init__(self, df, target, p_threshold=0.05, verbose=True):
        self.df = df
        self.target = target
        self.p_threshold = p_threshold
        self.verbose = verbose

    def get_sample_one_two(self, c):
        df0 = self.df[self.df[self.target] == 0]
        df1 = self.df[self.df[self.target] == 1]
        sample1 = df0[c].dropna()
        sample2 = df1[c].dropna()
        return sample1.tolist(), sample2.tolist()

    def create_contingency_table_for_chi2(self, c):
        sample1, sample2 = self.get_sample_one_two(c)
        if sample1 and sample2:
            df = pd.DataFrame(
                {
                    c: sample1 + sample2,
                    self.target: [0] * len(sample1) + [1] * len(sample2),
                }
            )
            return pd.crosstab(df[c], df[self.target])
        else:
            if self.verbose:
                print(
                    f"Skipping Chi-square test for column {c} due to insufficient data."
                )
            return pd.DataFrame()

    def compute_chisquare_test(self, contingency):
        if not contingency.empty:
            return stats.chi2_contingency(contingency)
        else:
            return np.nan, np.nan

    def compute_stat_importance_categorical(self, c):
        contingency_df = self.create_contingency_table_for_chi2(c)
        result = self.compute_chisquare_test(contingency_df)
        rejected = result[1] < self.p_threshold if result[1] is not np.nan else False
        return result[1], rejected

    def check_if_distribution_is_normal(self, sample):
        try:
            if len(sample) > 1:
                k2, p = stats.normaltest(sample)
                return p >= 1e-3
            else:
                if self.verbose:
                    print(
                        f"Distribution is not normal. Cannot use t-test for column {c}."
                    )
                return np.nan, False
        except ValueError:
            if self.verbose:
                print(
                    f"Not enough data to determine if distribution is normal for column {c}."
                )
            return np.nan, False

    def perform_t_test(self, sample1, sample2):
        if len(sample1) > 1 and len(sample2) > 1:
            v1, v2 = np.var(sample1), np.var(sample2)
            n = v1 if v1 > v2 else v2
            d = v1 if v1 < v2 else v2 + np.finfo(float).eps
            r = n / d
            equal_var = r <= 4
            s, p = stats.ttest_ind(a=sample1, b=sample2, equal_var=equal_var)
            return s, p
        else:
            return np.nan, np.nan

    def check_if_distributions_are_similar(self, sample1, sample2):
        if sample1 and sample2:  # Add this check
            return stats.ks_2samp(sample1, sample2).pvalue > self.p_threshold
        else:
            if self.verbose:
                print(
                    f"Cannot perform Kolmogorov-Smirnov test due to insufficient data."
                )
            return False

    def compute_stat_importance_t_test(self, c):
        sample1, sample2 = self.get_sample_one_two(c)
        if self.check_if_distribution_is_normal(
            sample1
        ) and self.check_if_distribution_is_normal(sample2):
            result = self.perform_t_test(sample1, sample2)
            rejected = (
                result[1] < self.p_threshold if result[1] is not np.nan else False
            )
            return result[1], rejected
        else:
            if self.verbose:
                print(f"Distribution is not normal. Cannot use t-test for column {c}.")
            return np.nan, None

    def compute_wilcoxon_ranksum_test(self, sample1, sample2):
        if len(sample1) > 1 and len(sample2) > 1:
            return stats.ranksums(sample1, sample2)
        else:
            return np.nan, np.nan

    def compute_stat_importance_numerical(self, c):
        sample1, sample2 = self.get_sample_one_two(c)
        if self.check_if_distributions_are_similar(sample1, sample2):
            result = self.compute_wilcoxon_ranksum_test(sample1, sample2)
            rejected = (
                result[1] < self.p_threshold if result[1] is not np.nan else False
            )
            return result[1], rejected
        else:
            if self.verbose:
                print(
                    f"Distribution shapes are not similar. Cannot use Wilcoxon rank sum test for column {c}."
                )
            return np.nan, None


def count_columns_by_dtype(df, return_lists=False):
    """
    Function to count the number of columns of each data type in a pandas DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame whose column types are to be counted.
    return_lists (bool): Whether to return lists of columns by their data types.

    Returns:
    tuple of lists of column names (strings) if return_lists is True, else None.
    Prints count of each data type.
    """

    unique_dtypes = df.dtypes.apply(lambda x: x.name).unique()
    dtypes_list = unique_dtypes.tolist()

    bool_cols, float_cols, ordinal_cols, non_ordinal_cols, other_cols = (
        [],
        [],
        [],
        [],
        [],
    )

    for dtype in dtypes_list:
        if dtype == "category":
            cat_cols = df.select_dtypes(include=[dtype])
            col_names = cat_cols.columns.tolist()
            ordinal_cols = [c for c in col_names if cat_cols[c].cat.ordered]
            non_ordinal_cols = [c for c in col_names if not cat_cols[c].cat.ordered]
            if return_lists is False:
                print(f"ordinal category: {len(ordinal_cols)}")
                print(f"non ordinal category: {len(non_ordinal_cols)}")

        elif dtype == "boolean":
            bool_cols = df.select_dtypes(include=[dtype]).columns.tolist()
            if return_lists is False:
                print(f"{dtype}: {len(bool_cols)}")

        elif dtype == "float64":
            float_cols = df.select_dtypes(include=[dtype]).columns.tolist()
            if return_lists is False:
                print(f"{dtype}: {len(float_cols)}")

        else:
            other_cols_df = df.select_dtypes(include=[dtype])
            other_cols.extend(other_cols_df.columns.tolist())
            if return_lists is False:
                print(f"{dtype}: {other_cols_df.shape[1]}")

    if return_lists:
        return bool_cols, float_cols, ordinal_cols, non_ordinal_cols, other_cols


class DataFrameCaster:
    """
    DataFrameCaster class casts pandas dataframe columns to appropriate datatypes.

    Attributes:
    ----------
    df : DataFrame
        The DataFrame to be casted

    Methods:
    -------
    infer_and_cast():
        Casts each column of the DataFrame to the appropriate datatype.
    """

    def __init__(self, df):
        """
        Constructs the necessary attributes for the DataFrameCaster object.

        Parameters:
        ----------
        df : DataFrame
            The DataFrame to be casted
        """
        self.df = df

    def infer_and_cast(self):
        """
        Casts each column of the DataFrame to an appropriate datatype.

        For each column, if it consists of '0' and '1' (both numeric and string),
        it is casted to boolean. If the column contains numerical values encoded as
        strings (excluding np.nan), it is casted to float. Otherwise, if a column has
        less than 10 unique values, it is considered as a categorical column.

        Returns:
        -------
        DataFrame
            The DataFrame after casting the columns
        """
        for col in self.df.columns:
            # If the column is already boolean, convert it to 'boolean' if it's not already
            if self.df[col].dtype == "bool":
                self.df[col] = self.df[col].astype("boolean")
                continue

            unique_values = self.df[col].dropna().unique()
            unique_value_set = set(unique_values)  # drop NaNs first

            # Check for boolean values
            boolean_sets = [
                {0, 1},
                {0.0, 1.0},
                {"0", "1"},
                {"Si", "No"},
                {"si", "no"},
                {"SI", "NO"},
                {"Yes", "No"},
                {"yes", "no"},
                {"YES", "NO"},
            ]
            if (
                unique_value_set in boolean_sets
                or set(list(unique_values)[::-1]) in boolean_sets
            ):
                if (
                    self.df[col].dtype == "object"
                ):  # Check if the column is of string type
                    self.df[col] = np.where(
                        self.df[col].str.lower() == "no",
                        False,
                        np.where(
                            self.df[col].str.lower() == "si",
                            True,
                            np.where(
                                self.df[col].str.lower() == "yes",
                                True,
                                np.where(
                                    self.df[col] == "0",
                                    False,
                                    np.where(
                                        self.df[col] == "1",
                                        True,
                                        np.where(
                                            self.df[col] == 0,
                                            False,
                                            np.where(self.df[col] == 1, True, np.nan),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    )  # Replace any other value with NaN
                else:
                    self.df[col] = np.where(
                        self.df[col] == "0",
                        False,
                        np.where(
                            self.df[col] == "1",
                            True,
                            np.where(
                                self.df[col] == 0,
                                False,
                                np.where(self.df[col] == 1, True, np.nan),
                            ),
                        ),
                    )  # Replace any other value with NaN
                self.df[col] = self.df[col].astype("boolean")
                continue

            elif len(unique_values) < 10:
                if np.issubdtype(unique_values.dtype, np.number):
                    self.df[col] = self.df[col].astype(
                        CategoricalDtype(
                            categories=np.sort(unique_values), ordered=True
                        )
                    )
                else:
                    try:
                        # Check if the string values can be converted to numeric (float)
                        numeric_values = pd.to_numeric(unique_values, errors="coerce")
                        if not pd.Series(numeric_values).isna().any():
                            # If conversion successful, convert column to numeric
                            self.df[col] = self.df[col].astype(float)
                            # Check if the float values can be converted to boolean
                            if set(self.df[col].dropna().unique()) == {0.0, 1.0}:
                                self.df[col] = np.where(
                                    self.df[col] == 0.0,
                                    False,
                                    np.where(self.df[col] == 1.0, True, self.df[col]),
                                )
                                self.df[col] = self.df[col].astype(
                                    "boolean"
                                )  # use "boolean" instead of bool
                        else:
                            self.df[col] = self.df[col].astype("category")
                    except ValueError:
                        self.df[col] = self.df[col].astype("category")
            else:
                try:
                    # Attempt to convert the column to float
                    self.df[col] = self.df[col].astype(float)
                except ValueError:
                    self.df[col] = self.df[col].astype("category")

        return self.df


def cast_category_to_object(df):
    """
    This function will cast columns of type 'category' as 'object' in a pandas dataframe.

    :param df: input pandas DataFrame
    :return: DataFrame with 'category' columns casted to 'object'
    """
    for col in df.columns:
        if df[col].dtype.name == "category":
            df[col] = df[col].astype("object")
    return df


def identify_near_zero_variance(
    df, prevalence_threshold=0.99, unique_ratio_threshold=0.01
):
    """
    This function uses two criteria to define near zero variance:
    1 The prevalence of the most common value in a column is greater than prevalence_threshold (default is 95%).
    In other words, the most common value accounts for more than 95% of all observations.
    2 The ratio of unique values to the total number of samples is less than unique_ratio_threshold (default is 10%)
    In other words, less than 10% of the column's values are unique.
    """
    near_zero_variance_cols = []
    for col in df.columns:
        if (
            df[col].dtype.kind in "biufc"
        ):  # Check if column dtype is in {byte, int, uint, float, complex}
            most_common_val_pct = (
                df[col].value_counts(normalize=True).values[0]
            )  # Percentage of most common value
            unique_ratio = (
                df[col].nunique() / df[col].count()
            )  # Ratio of unique values to total number of samples

            if (
                most_common_val_pct > prevalence_threshold
                and unique_ratio < unique_ratio_threshold
            ):
                near_zero_variance_cols.append(col)
    return near_zero_variance_cols


def check_separation(df, target="death"):
    perfect_separators = []
    almost_perfect_separators = []

    for column in df.columns:
        # Skip the target column
        if column == target:
            continue

        if df[column].dtype.name == "category" or df[column].dtype.name == "boolean":
            # Calculate the number of unique targets for each value of the predictor
            unique_targets = df.groupby(column)[target].nunique()

            # If the number of unique targets is always 1, then this is a perfect separator
            if unique_targets.max() == 1:
                perfect_separators.append(column)
            # If the number of unique targets is mostly 1, then this is an almost perfect separator
            elif (
                unique_targets.mean() <= 1.1
            ):  # the threshold 1.1 is just an example, you can adjust it to your needs
                almost_perfect_separators.append(column)

        elif df[column].dtype.name == "float64":
            # Calculate the correlation with the target
            correlation = df[[column, target]].corr().iloc[0, 1]

            # If the correlation is perfect, then this is a perfect separator
            if abs(correlation) == 1:
                perfect_separators.append(column)
            # If the correlation is very high, then this is an almost perfect separator
            elif (
                abs(correlation) > 0.9
            ):  # the threshold 0.9 is just an example, you can adjust it to your needs
                almost_perfect_separators.append(column)

    return perfect_separators, almost_perfect_separators


def find_least_significative(df, col_0, col_1):
    test_0 = df[df["col_name"] == col_0]["signific"].mean()
    test_1 = df[df["col_name"] == col_1]["signific"].mean()
    if test_0 > test_1:
        return col_0
    elif test_0 < test_1:
        return col_1
    else:
        return None


def one_hot_encoding(df, cols):
    # Copy the DataFrame to avoid modifying the original one
    df_copy = df.copy()

    # Create dummy DataFrames
    dummies = [
        pd.get_dummies(df_copy[col], prefix=col, drop_first=True).astype("boolean")
        for col in cols
    ]

    # Concatenate original DataFrame with dummy DataFrames
    df_ohe = pd.concat([df_copy] + dummies, axis=1)

    # Drop original columns
    df_ohe = df_ohe.drop(columns=cols)

    return df_ohe


def plot_kaplanmeier(
    df,
    event_col,
    time_col,
    tau_months,
    ylim=None,
    ydef=0.01,
    col_groupby=None,
    ax=None,
    title="Kaplan Maier survival curve",
    legend_title=None,
):
    """
    Plot Kaplan-Meier survival curves.

    Parameters:
    - df (DataFrame): The input data frame containing survival data.
    - event_col (str): Column name in the data frame representing the event (1 if the event occurred, 0 otherwise).
    - time_col (str): Column name in the data frame representing the survival time.
    - tau_months (int): Maximum time for x-axis (in months).
    - ylim (tuple, optional): Tuple specifying the y-axis limits. Defaults to None, in which case the limits are computed based on data.
    - ydef (float, optional): The decremental value to generate y-ticks. Defaults to 0.01.
    - col_groupby (str, optional): Column name to group by for plotting multiple survival curves. If not specified, a single curve is plotted.
    - ax (matplotlib axis object, optional): The axis on which to plot the survival curve. If None, a new figure and axis are created.
    - title (str, optional): Title of the plot. Defaults to "Kaplan Maier survival curve".
    - legend_title (str, optional): Title for the legend. Defaults to None, in which case the col_groupby value is used if provided.

    Returns:
    None. The function plots the Kaplan-Meier survival curves directly.

    Notes:
    - The function uses `kaplan_meier_estimator` to compute the survival probabilities and confidence intervals.
    - If `col_groupby` is provided, the function will plot multiple survival curves (one for each group) and compute logrank tests between each pair of groups.
    - The plotted survival curves include shaded areas representing the confidence intervals.

    Example:
    plot_kaplanmeier(df=my_dataframe, event_col='event', time_col='duration', tau_months=60, col_groupby='treatment_group')
    """
    if col_groupby:
        df = df.dropna(subset=[col_groupby])
        df = df.sort_values(col_groupby)
    if ax is None:
        fig, ax = plt.subplots()
    if col_groupby is None:
        time, survival_prob, conf_int = kaplan_meier_estimator(
            df[event_col].astype(bool), df[time_col], conf_type="log-log"
        )
        time = np.insert(time, 0, 0)
        survival_prob = np.insert(survival_prob, 0, 1)
        conf_int = (np.insert(conf_int[0], 0, 1), np.insert(conf_int[1], 0, 1))
        ax.step(time, survival_prob, where="post")
        ax.fill_between(
            time, conf_int[0], conf_int[1], alpha=0.2, step="post", color="grey"
        )
        ax.set_title(title)
    if col_groupby is not None:
        survival_data = {}
        survival_prob_list = []
        for value in df[col_groupby].unique():
            mask = df[col_groupby] == value
            time_cell, survival_prob_cell, conf_int = kaplan_meier_estimator(
                df[event_col].astype(bool)[mask],
                df[time_col][mask],
                conf_type="log-log",
            )
            time_cell = np.insert(time_cell, 0, 0)
            survival_prob_cell = np.insert(survival_prob_cell, 0, 1)
            conf_int = (np.insert(conf_int[0], 0, 1), np.insert(conf_int[1], 0, 1))
            ax.step(
                time_cell,
                survival_prob_cell,
                where="post",
                label=f"{value} (n = {mask.sum()})",
            )
            ax.fill_between(
                time_cell,
                conf_int[0],
                conf_int[1],
                alpha=0.2,
                step="post",
                color="grey",
            )
            survival_data[value] = (
                df[event_col].astype(bool)[mask],
                df[time_col][mask],
            )
            survival_prob_list.append(survival_prob_cell)
        survival_prob_list = [
            item for sublist in survival_prob_list for item in sublist
        ]
        unique_groups = list(df[col_groupby].unique())
        for i in range(len(unique_groups)):
            for j in range(i + 1, len(unique_groups)):
                group1, group2 = unique_groups[i], unique_groups[j]
                results = logrank_test(
                    survival_data[group1][0],
                    survival_data[group2][0],
                    survival_data[group1][1],
                    survival_data[group2][1],
                )
                print(
                    f"Logrank Test between {group1} and {group2}: p-value = {results.p_value:.4f}"
                )
        ax.set_title(title)
        legend = ax.legend(loc="best")
        if legend:
            legend.set_title(legend_title)
    if not ylim:
        if col_groupby is None:
            survival_prob_list = [survival_prob]
        ylim = round(round(np.min(survival_prob_list), 2) - 0.02, 2)
    ax.set_xlim(0, tau_months)
    ax.set_ylabel("est. probability of survival $\hat{S}(t)$")
    ax.set_xlabel("time $t$ (months)")
    yticks = [1.00]
    current_value = 1.00
    while current_value - ydef > ylim:
        current_value -= ydef
        yticks.append(round(current_value, 2))
    yticks = sorted(yticks)
    ax.set_yticks(yticks)
    ax.set_ylim(ylim, 1 + (ydef / 5))

    
def get_highly_correlated_pairs(df, method="spearman", threshold=0.8):
    """Spot highly correlated pairs of features in a pandas DataFrame."""
    corr_matrix = df.corr(method=method).abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find pairs of feature columns with correlation greater than threshold
    to_drop_pairs = [
        (col, row)
        for col in upper.columns
        for row in upper.index
        if upper.loc[row, col] > threshold
    ]

    return to_drop_pairs


def optimize_rsf(
    X_tune,
    y_tune,
    grid,
    n_trials,
    n_folds,
    model_dir,
    model_filename,
    random_state,
):
    """
    Bayesian grid search for a RandomSurvivalForest using Optuna.
    This function performs hyperparameter optimization and saves the best hyperparameters.

    Parameters:
    - X_tune: Feature set for tuning
    - y_tune: Target set for tuning
    - grid: Search grid for hyperparameters
    - n_trials: Number of optimization trials
    - n_folds: Number of cross-validation folds
    - model_dir: Directory where to save the model
    - model_filename: Filename for saving the model
    - random_state: Random state for reproducibility
    """

    # Create directories if not existing
    os.makedirs(model_dir, exist_ok=True)

    # Full path for model saving
    model_path = os.path.join(model_dir, model_filename)

    # Objective function for Optuna
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", grid["n_estimators"][0], grid["n_estimators"][1]
            ),
            "max_depth": trial.suggest_int(
                "max_depth", grid["max_depth"][0], grid["max_depth"][1]
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                grid["min_samples_split"][0],
                grid["min_samples_split"][1],
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                grid["min_samples_leaf"][0],
                grid["min_samples_leaf"][1],
            ),
            "max_features": trial.suggest_int(
                "max_features",
                grid["max_features"][0],
                grid["max_features"][1],
            ),
            "warm_start": False,
        }
        # Stratified K-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        scores = []
        for train_index, val_index in skf.split(X_tune, y_tune["event"]):
            X_train, X_val = X_tune.iloc[train_index], X_tune.iloc[val_index]
            y_train, y_val = y_tune[train_index], y_tune[val_index]

            rsf = RandomSurvivalForest(**params, random_state=random_state)
            rsf.fit(X_train, y_train)
            predicted_risk = rsf.predict(X_val)
            result_censored = concordance_index_censored(
                y_val["event"], y_val["time"], predicted_risk
            )
            scores.append(result_censored[0])

        return -np.mean(scores)  # Negate to maximize

    # Progress bar callback for Optuna
    def callback(study, trial):
        pbar.update(1)

    model_path = os.path.join(model_dir, model_filename)
    # Check if model file already exists
    if os.path.isfile(model_path):
        mlflow.log_param("model_path", model_path)
        print(f"Older version of the model found in {model_path}")
        return

    try:
        # Disable logging for Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Optuna optimization
        study = optuna.create_study()
        start_time = datetime.datetime.now()

        with tqdm(total=n_trials, desc="Optimizing RandomSurvivalForest") as pbar:
            study.optimize(objective, n_trials=n_trials, callbacks=[callback])

        end_time = datetime.datetime.now()

        # Save the best hyperparameters in a JSON file
        best_params_dict = {
            "random_state": random_state,
            "best_params": study.best_params,
            "best_performance": study.best_value,
            "training_steps": n_trials,
            "training_start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(model_path, "w") as f:
            json.dump(best_params_dict, f, indent=4)
        mlflow.log_param("model_path", model_path)
        print(f"Best hyperparameters saved in {model_path}")

        # Display the best hyperparameters and performance
        print(f"Best Hyperparameters: {study.best_params}")
        print(f"Best Performance (Negative Concordance Index): {study.best_value}")

    except Exception as e:
        print(f"An error occurred: {e}")


def replace_longest_match(value, dict_legend):
    """Replaces the beginning of a string based on the longest matching key in a dictionary.
    Useful for variables expanded through one-hot encoding."""

    # Sort the keys of the dictionary in descending order of their length
    sorted_keys = sorted(dict_legend.keys(), key=len, reverse=True)

    for key in sorted_keys:
        if value.startswith(key):
            # Extract the suffix
            suffix = value[len(key) + 1 :]

            # Format the replacement with the suffix, separated by a space if the suffix is not empty
            replacement = dict_legend.get(key, key)
            return f"{replacement} {suffix}".strip() if suffix else replacement

    # Return the original value if no match is found
    return value


def plot_feature_importance(df, n_features, size: tuple = (5, 8)):
    """Plot an histogram given a pandas DataFrame with feature importances"""
    # Sort DataFrame by 'importances_mean' for better visualization
    df_plot = df.sort_values(by="importances_mean", ascending=False).head(n_features)
    # Create a figure and a set of subplots (ax)
    # Increased height to 10 units
    fig, ax = plt.subplots(figsize=size)
    # Create horizontal bar chart
    y_pos = np.arange(len(df_plot["feature_definition"]))
    bars = ax.barh(
        y_pos,
        df_plot["importances_mean"],
        align="center",
        color="skyblue",
        edgecolor="blue",
        alpha=0.7,
    )
    # Add error bars separately using errorbar function with elinewidth
    for bar, err in zip(bars, df_plot["importances_std"]):
        ax.errorbar(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            xerr=err,
            color="black",
            elinewidth=0.5,
            capsize=3,
        )
    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance with Standard Deviation")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["feature_definition"])
    # Make the chart visually more appealing
    ax.invert_yaxis()  # reverse the y-axis to have the top features at the top
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
    plt.show()


def plot_multiple_kaplanmeier(df_risk, ylim, ydef):
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))

    plot_kaplanmeier(
        df=df_risk,
        event_col="death",
        time_col="ttdeath",
        tau_months=60,
        ylim=ylim,
        ydef=ydef,
        col_groupby="risk_group_grant",
        ax=axes[0],
        title="Prognostic GRANT",
    )

    plot_kaplanmeier(
        df=df_risk,
        event_col="death",
        time_col="ttdeath",
        tau_months=60,
        ylim=ylim,
        ydef=ydef,
        col_groupby="risk_group_cox_baseline",
        ax=axes[1],
        title="Cox baseline",
    )

    plot_kaplanmeier(
        df=df_risk,
        event_col="death",
        time_col="ttdeath",
        tau_months=60,
        ylim=ylim,
        ydef=ydef,
        col_groupby="risk_group_cox_t1",
        ax=axes[2],
        title="Cox T1",
    )

    plot_kaplanmeier(
        df=df_risk,
        event_col="death",
        time_col="ttdeath",
        tau_months=60,
        ylim=ylim,
        ydef=ydef,
        col_groupby="risk_group_cox_t0",
        ax=axes[3],
        title="Cox T0",
    )

    fig.suptitle("Kaplan Maier survival curves", fontsize=16)
    plt.tight_layout()
    plt.show()

    
def find_problematic_values(values, n_models):
    """Check if each unique value appears exactly n_models times"""
    value_counts = Counter(values)
    problematic_values = [value for value, count in value_counts.items() if count != n_models]
    return problematic_values


def delete_run(experiment, client, random_state_delete):
    for run in tqdm(experiment.get_runs(include_children=True)):
        data = client.get_run(run.id).data
        if "model_path" in data.params and data.params["model_path"]:
            if data.params["random_state"] == str(random_state_delete):
                run_info = client.get_run(run.id).info
                if run_info.status == "RUNNING":
                    client.set_terminated(run.id)
                print("Deleting", run.id)
                client.delete_run(run.id)
        else:
            continue
            
def find_least_parent_run_id(group):
    """
    Function to find the parent_run_id with the least rows if there are multiple
    """
    parent_run_counts = group['parent_run_id'].value_counts()
    if len(parent_run_counts) > 1:
        return parent_run_counts.idxmin()
    return None

def delete_run_with_parent(experiment, client, least_parent_run_ids_dict, verbose=False):
    gen = experiment.get_runs(include_children=True)
    for run in tqdm(gen):
        data = client.get_run(run.id).data
        # Check if child run
        if "mlflow.parentRunId" in data.tags:
            # Get model path
            artifacts = mlflow.tracking.MlflowClient().list_artifacts(run.id)
            counter = 0
            for artifact in artifacts:
                if artifact.path.endswith(".pkl"):
                    model_path = artifact.path
                    counter += 1
            if counter == 0:
                if "model_path" in data.params and data.params["model_path"]:
                    model_path = data.params["model_path"]
            if counter > 1:
                if verbose is True:
                    print(run.id, "No model path")
            if model_path:
                random_state = data.params.get("random_state")
                parent_run_id = data.tags.get("mlflow.parentRunId")
                if random_state in least_parent_run_ids_dict:
                    print(f"random_state {random_state} found in least_parent_run_ids_dict")
                    if parent_run_id == least_parent_run_ids_dict[random_state]:
                        print(f"Match found for random_state {random_state} and parent_run_id {parent_run_id}")
                        run_info = client.get_run(run.id).info
                        if run_info.status == "RUNNING":
                            client.set_terminated(run.id)
                        if verbose:
                            print("Deleting", run.id)
                        client.delete_run(run.id)
            else:
                continue
            
def delete_all_runs(experiment, client):
    for run in tqdm(experiment.get_runs(include_children=True)):
        data = client.get_run(run.id).data
        run_info = client.get_run(run.id).info
        if run_info.status == "RUNNING":
            client.set_terminated(run.id)
        print("Deleting", run.id)
        client.delete_run(run.id)


def filter_filenames(dataset, filenames, timepoint):
    """
    Filters a list of filenames to include only those that match the pattern:
    'raw_best_model_rsf_' followed by a date in the format yyyyMMdd-HHmmss.
    """
    if timepoint == "T1":
        pattern = "^{}\_best\_model\_rsf\_\d+\.json$".format(re.escape(dataset))
    elif timepoint == "t0":
        pattern = "^{}\_best\_model\_rsf\_t0\_\d+\.json$".format(re.escape(dataset))
    else:
        raise ValueError("Invalid timepoint specified.")
    return [filename for filename in filenames if re.match(pattern, filename)]


def count_occurrences(list_of_lists, len_most_common=10):
    # Limit len of nested lists
    list_of_lists = [x[:len_most_common] for x in list_of_lists]

    # Count occurrences of each feature
    feature_count = Counter(feature for sublist in list_of_lists for feature in sublist)

    # Get the top N most frequent features
    top_n_features = feature_count.most_common(len_most_common)

    # Initialize dictionaries to count rankings
    first_rank_count = {feature: 0 for feature, _ in top_n_features}
    top_5_rank_count = {feature: 0 for feature, _ in top_n_features}

    # Count the number of times each feature is ranked first and in the top 3
    for sublist in list_of_lists:
        for rank, feature in enumerate(sublist):
            if feature in first_rank_count and rank == 0:
                first_rank_count[feature] += 1
            if feature in top_5_rank_count and rank < 5:
                top_5_rank_count[feature] += 1

    # Prepare data for DataFrame
    data = []
    for feature, count in top_n_features:
        data.append(
            {
                "Feature": feature,
                "Freq. top {}".format(len_most_common): count,
                "Freq. top 5": top_5_rank_count[feature],
                "Freq. first rank": first_rank_count[feature],
            }
        )

    # Create DataFrame
    return pd.DataFrame(data)


def split_string(s):
    # Pattern to match the required groups
    pattern = r"(.+?)_(\d+)_(\d{8}-\d{6})"
    match = re.match(pattern, s)
    if match:
        return match.groups()
    else:
        return None

class SelectKFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Select the top k features based on their presumed ordering by importance. It assumes that the input features (X) are already sorted by descending importance
    """
    def __init__(self, k=8):  # Default value can be adjusted as needed.
        self.k = k
        self.selected_features_ = None

    def fit(self, X, y=None):
        # Assuming X is already sorted by feature importance in descending order
        # Select the top-k features. No need for SelectKBest if features are pre-ranked.
        self.selected_features_ = np.arange(self.k)
        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise ValueError("The fit method must be called before transform.")
        # Check if X is a DataFrame and use .iloc if true
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_features_]
        else:
            # Fallback for numpy arrays in case X is not a DataFrame
            return X[:, self.selected_features_]

def tune_sksurv_model(model, param_grid, k_min, k_step_size, X_tune, y_tune, random_state: int, n_folds=10):
    """Tune sksurv models with feature selection, hyperparameter tuning, stratification based on the 'event', and return the best model"""
    # Ensure param_grid includes a range for 'select__k'
    if 'select__k' not in param_grid:
        select_k_range = range(k_min, X_tune.shape[1] + 1, k_step_size)
        mlflow.log_param("select_k_range", str(list(select_k_range)))
        param_grid['select__k'] = select_k_range
    # Initialize variables to track the best model and its performance
    best_score = -1
    best_model = None

    # Stratified K-Fold Cross-Validation based on 'event' indicator
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    for train_index, val_index in skf.split(X_tune, y_tune["event"]):
        # Splitting the data into training and validation sets based on the indices
        X_train, X_val = X_tune.iloc[train_index], X_tune.iloc[val_index]
        y_train, y_val = y_tune[train_index], y_tune[val_index]

        # Creating a pipeline with feature selection and the model
        model_pipeline = Pipeline([
            ("select", SelectKFeaturesTransformer()),
            ("model", model)
        ])

        # Initialize GridSearchCV without CV strategy since we are manually splitting the folds
        grid_search = GridSearchCV(model_pipeline, param_grid, n_jobs=-1)

        # Fit the model on the current training fold
        grid_search.fit(X_train, y_train)

        # Evaluate the model on the validation set
        best_model_fold = grid_search.best_estimator_
        predicted_risk = best_model_fold.predict(X_val)
        result_censored = concordance_index_censored(
            y_val["event"], y_val["time"], predicted_risk
        )
        score = result_censored[0]

        # Update the best model if the current one is better
        if score > best_score:
            best_score = score
            best_model = best_model_fold

    # Before returning, fit the best_model (pipeline) to the entire tuning dataset
    if best_model is not None:
        best_model.fit(X_tune, y_tune)
    
    # Return the best model based on the highest concordance index
    return best_model, best_score


def validate_sksurv_model(
    model,
    y_train,
    X_test,
    y_test,
    tau,
):
    """
    Validate sksurv models using concordance_index_censored, concordance_index_ipcw,
    integrated_brier_score, mean_cumulative_dynamic_auc, and plot time-dependent AUC.

    Parameters:
    - model: Trained sksurv model.
    - y_train: Training survival data.
    - X_test: Test features.
    - y_test: Test survival data.
    - tau: Time point for IPCW concordance index and for AUC

    Returns:
    - result_censored: Concordance index (censored).
    - result_ipcw: Concordance index (IPCW).
    - score_brier: Integrated Brier score.
    - mean_auc: Mean time-dependent AUC.
    - fig: Matplotlib figure object of the AUC plot.
    """
    # Predict risk scores
    try:
        predicted_risk = model.predict(X_test)
    except:
        predicted_risk = model.predict(X_test.values)
    
    # Evaluate Concordance Index (Censored)
    result_censored = concordance_index_censored(
        event_indicator=y_test["event"],
        event_time=y_test["time"],
        estimate=predicted_risk
    )[0]
    
    # Evaluate Concordance Index (IPCW)
    result_ipcw = concordance_index_ipcw(
        survival_train=y_train,
        survival_test=y_test,
        estimate=predicted_risk,
        tau=tau,
    )[0]
    
    # Compute Integrated Brier Score (IBS)
    # Determine time points between the 10th and 90th percentile
    all_times = np.concatenate([y_train["time"], y_test["time"]])
    lower, upper = np.percentile(all_times, [10, 90]) 
    brier_times = np.arange(lower, upper)
    
    # Predict survival probabilities at the specified time points
    try:
        surv_prob = np.row_stack([
            fn(brier_times) for fn in model.predict_survival_function(X_test)
        ])
    except:
        surv_prob = np.row_stack([
            fn(brier_times) for fn in model.predict_survival_function(X_test.values)
        ])
    
    # Calculate Integrated Brier Score
    score_brier = integrated_brier_score(
        survival_train=y_train,
        survival_test=y_test,
        estimate=surv_prob,
        times=brier_times
    )
    
    # Compute Time-dependent AUC
    mean_auc = np.nan
    adjustment = 0
    while np.isnan(mean_auc):
        print(adjustment)
        # Define validation time points with adjustment
        va_times = np.linspace(1 + adjustment, tau - 1, int(tau/10))
        va_times = np.round(va_times, 0)
        try:
            auc, mean_auc = cumulative_dynamic_auc(
                survival_train=y_train,
                survival_test=y_test,
                estimate=predicted_risk,
                times=va_times
            )
        except Exception as e:
            print(f"Exception occurred: {e}")
            adjustment += 1
            continue  # Skip to the next iteration
        if np.isnan(mean_auc):
            adjustment += 1
        # Prevent infinite loop by setting a maximum adjustment
        if adjustment > 10:
            raise ValueError("Unable to compute mean_auc after multiple adjustments.")
    
    # Plot Time-dependent AUC
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(va_times, auc, marker="o", label="AUC at time t")
    ax.axhline(mean_auc, linestyle="--", color="red", label=f"Mean AUC = {mean_auc:.3f}")
    ax.set_xlabel("Time $t$ (months)")
    ax.set_ylabel("Time-dependent AUC")
    ax.grid(True)
    ax.legend()
    
    plt.close(fig)  # Prevent display in some environments
    
    return result_censored, result_ipcw, score_brier, mean_auc, fig


def pipeline_skurv(model,
               param_grid,
               k_min,
               X_train,
               y_train,
               X_test,
               y_test,
               random_state,
               n_folds,
               tau,
               dir_models,
               dataset_name,
               timepoint):
    """Training pipeline for sksurv models with mlflow logging"""
    model_name = model.__class__.__name__
    assert model_name and timepoint
    mlflow.start_run(run_name=model_name + "_" + timepoint, nested=True)
    mlflow.log_param("random_state", random_state)

    # Log the tuning grid
    mlflow.log_params(param_grid)

    best_model, best_score = tune_sksurv_model(model=model,
                                   param_grid=param_grid,
                                   k_min=k_min,
                                   k_step_size=3,
                                   X_tune=X_train,
                                   y_tune=y_train,
                                   random_state=random_state,
                                   n_folds=n_folds)
    # Log best performance obtained during tuning
    mlflow.log_param("best_performance", -best_score)
    # Log artifact
    os.makedirs(dir_models, exist_ok=True)
    model_path = os.path.join(dir_models, '{}_{}_{}_{}.pkl'.format(dataset_name,
                                                                model_name,
                                                                timepoint,
                                                                random_state))
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)
    # Log the selected features (column names might be needed)
    selected_indices = best_model['select'].selected_features_
    # Map the selected indices to feature names
    selected_features = [X_train.columns.values[idx] for idx in selected_indices]
    mlflow.log_param("n_features_in", len(selected_features))
    # Create a dictionary for the JSON object
    features_info = {
        "model": model_name,
        "timepoint": timepoint,
        "feature_names_in": selected_features
    }
    # Convert the dictionary to a JSON string
    features_json_str = json.dumps(features_info, indent=2)
    # Log the features as an artifact with a specific filename
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Define the full path with the desired filename
        file_path = os.path.join(tmp_dir, 'feature_names_in.json')
        # Write the JSON string to the file
        with open(file_path, 'w') as f:
            f.write(features_json_str)

        # Log the file as an artifact
        mlflow.log_artifact(file_path)
    # Log best model hyperparameters
    params_dict = dict(best_model['model'].get_params())
    params_arr = [(key, str(value)) for key, value in params_dict.items()]
    mlflow.log_params(dict(params_arr))
    # Validate
    result_censored, result_ipcw, score_brier, mean_auc, fig = validate_sksurv_model(model=best_model,
                          y_train=y_train,
                          X_test=X_test,
                          y_test=y_test,
                          tau=tau)
    mlflow.log_metric("concordance_index_censored", result_censored)
    mlflow.log_metric("concordance_index_ipcw", result_ipcw)
    mlflow.log_metric("integrated_brier_score", score_brier)
    mlflow.log_metric("mean_cumulative_dynamic_auc", mean_auc)
    mlflow.log_figure(fig, "time_dependent_auc.png")
    mlflow.end_run()
    return None


def bootstrap_mccv_results(data, alpha=0.05, n_bootstraps=10000):
    """
    Perform bootstrapping on MCCV results to compute confidence intervals.

    Parameters:
    - data: array-like, performance metrics from MCCV iterations.
    - alpha: significance level. Default is 0.05 for a 95% confidence interval.
    - n_bootstraps: number of bootstrap samples. Default is 10,000.

    Returns:
    - mean: Mean of the original data.
    - ci: Confidence interval as a tuple (lower_bound, upper_bound).
    - boot_means: Array of bootstrapped means.
    """
    n = len(data)
    boot_means = []
    for _ in range(n_bootstraps):
        # Resample with replacement from the MCCV results
        sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    lower_bound = np.percentile(boot_means, (alpha / 2) * 100)
    upper_bound = np.percentile(boot_means, (1 - alpha / 2) * 100)
    mean = np.mean(data)
    ci = (lower_bound, upper_bound)
    return mean, ci, boot_means


class ModelComparer:
    def __init__(self, model1_metrics, model2_metrics, alpha=0.05, higher_is_better=True):
        """
        Initialize the comparer with model performances and significance level.

        Parameters:
        - model1_metrics: array-like, performance metrics for model 1 obtained from simulations.
        - model2_metrics: array-like, performance metrics for model 2 obtained from simulations.
        - alpha: float, significance level for confidence intervals. Default is 0.05.
        - higher_is_better: bool, whether a higher metric is better (default) or not.
        """
        self.model1_metrics = np.array(model1_metrics)
        self.model2_metrics = np.array(model2_metrics)
        self.alpha = alpha
        self.higher_is_better = higher_is_better
        self.model1_ci = None
        self.model2_ci = None
        self.median_diff = None
        self.ci_diff = None
        self.wilcoxon_stat = None
        self.wilcoxon_p_value = None

    def compute_confidence_intervals(self):
        """
        Compute empirical confidence intervals for each model's performance metrics.
        """
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        self.model1_ci = (np.percentile(self.model1_metrics, lower_percentile),
                          np.percentile(self.model1_metrics, upper_percentile))
        self.model2_ci = (np.percentile(self.model2_metrics, lower_percentile),
                          np.percentile(self.model2_metrics, upper_percentile))

    def compute_median_difference(self):
        """
        Compute the median difference and its confidence interval.
        """
        differences = self.model2_metrics - self.model1_metrics
        self.median_diff = np.median(differences)
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        self.ci_diff = (np.percentile(differences, lower_percentile),
                        np.percentile(differences, upper_percentile))

    def wilcoxon_signed_rank_test(self):
        """
        Perform the Wilcoxon signed-rank test on the paired data.

        Returns:
        - statistic: float, the test statistic.
        - p_value: float, the p-value of the test.
        """
        stat, p_value = wilcoxon(self.model1_metrics, self.model2_metrics, alternative='two-sided')
        self.wilcoxon_stat = stat
        self.wilcoxon_p_value = p_value
        return stat, p_value

    def print_results(self, model1_name, model2_name):
        """
        Print the results, including confidence intervals and statistical test outcomes.
        """
        if self.model1_ci is None or self.model2_ci is None:
            self.compute_confidence_intervals()
        if self.median_diff is None or self.ci_diff is None:
            self.compute_median_difference()

        print(f"{model1_name} Performance Metrics:")
        print(f"Mean: {np.mean(self.model1_metrics):.5f}")
        print(f"Median: {np.median(self.model1_metrics):.5f}")
        print(f"{(1 - self.alpha) * 100}% Confidence Interval: {self.model1_ci}\n")

        print(f"{model2_name} Performance Metrics:")
        print(f"Mean: {np.mean(self.model2_metrics):.5f}")
        print(f"Median: {np.median(self.model2_metrics):.5f}")
        print(f"{(1 - self.alpha) * 100}% Confidence Interval: {self.model2_ci}\n")

        print(f"Median difference in performance: {self.median_diff:.5f}")
        print(f"{(1 - self.alpha) * 100}% Confidence Interval of the differences: {self.ci_diff}\n")

        # Wilcoxon signed-rank test
        stat, p_value = self.wilcoxon_signed_rank_test()
        print(f"Wilcoxon signed-rank test statistic: {stat:.5f}, p-value: {p_value:.5f}")

        # Assessing significance based on confidence interval of differences
        if self.ci_diff[0] > 0:
            better_model = model2_name if self.higher_is_better else model1_name
            print(f"\n{better_model} is significantly better than the other (based on CI of differences).")
        elif self.ci_diff[1] < 0:
            better_model = model1_name if self.higher_is_better else model2_name
            print(f"\n{better_model} is significantly better than the other (based on CI of differences).")
        else:
            print("\nNo significant difference between the models based on the confidence interval of differences.")

        # Assessing significance based on Wilcoxon test
        if p_value < self.alpha:
            if np.median(self.model2_metrics) > np.median(self.model1_metrics):
                better_model = model2_name if self.higher_is_better else model1_name
            else:
                better_model = model1_name if self.higher_is_better else model2_name
            print(f"{better_model} is significantly better than the other (based on Wilcoxon signed-rank test).")
        else:
            print("No significant difference between the models based on the Wilcoxon signed-rank test.")

    def plot_performance_distributions(self, model1_name, model2_name):
        """
        Plot the distributions of performance metrics for both models.
        """
        plt.figure(figsize=(12, 6))
        plt.hist(self.model1_metrics, bins=20, alpha=0.5, label=model1_name, density=True)
        plt.hist(self.model2_metrics, bins=20, alpha=0.5, label=model2_name, density=True)
        plt.title('Performance Metrics Distribution')
        plt.xlabel('Performance Metric')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

    def plot_boxplots(self, model1_name, model2_name):
        """
        Plot boxplots of performance metrics for both models.
        """
        data = [self.model1_metrics, self.model2_metrics]
        plt.figure(figsize=(8, 6))
        plt.boxplot(data, labels=[model1_name, model2_name])
        plt.title('Performance Metrics Boxplots')
        plt.ylabel('Performance Metric')
        plt.show()

            
def collect_simulations(experiment, client, dir_artifacts, verbose=False):
    os.makedirs(dir_artifacts, exist_ok=True)
    l = []
    gen = experiment.get_runs(include_children=True)
    for run in tqdm(gen):
        # Access the run in MLflow
        data = mlflow.get_run(run.id).data
        # Check if child run
        if "mlflow.parentRunId" in data.tags:
            # Get model path
            artifacts = mlflow.tracking.MlflowClient().list_artifacts(run.id)
            counter = 0
            for artifact in artifacts:
                if artifact.path.endswith(".pkl"):
                    model_path = artifact.path
                    counter += 1
            if counter == 0:
                if "model_path" in data.params and data.params["model_path"]:
                    model_path = data.params["model_path"]
            if counter > 1:
                if verbose is True:
                    print(run.id, "No model path")
            try:
                l.append(
                    {
                        "model": data.tags["mlflow.runName"],
                        "model_path": model_path,
                        "random_state": data.params["random_state"],
                        "parent_run_id": data.tags['mlflow.parentRunId'],
                        "n_features_in": data.params["n_features_in"],
                        "feature_names_in": data.params["feature_names_in"] if "feature_names_in" in data.params else "",
                        "best_performance_tuning": data.params["best_performance"] if "best_performance" in data.params else "",
                        "concordance_index_censored": data.metrics[
                            "concordance_index_censored"
                        ],
                        "concordance_index_ipcw": data.metrics["concordance_index_ipcw"],
                        "integrated_brier_score": data.metrics["integrated_brier_score"],
                        "mean_cumulative_dynamic_auc": data.metrics[
                            "mean_cumulative_dynamic_auc"
                        ],
                    }
                )
            except:
                parent_run_id = data.tags.get('mlflow.parentRunId')
                parent_run = mlflow.get_run(parent_run_id)
                parent_run_name = parent_run.data.tags.get('mlflow.runName')  # Custom tag for run name
                if verbose is True:
                    print(f"Parent run name for model with missing info: {parent_run_name}")
            # Dowload feature importance
            if "RandomSurvivalForest_selector_" in data.tags["mlflow.runName"]:
                try:
                    client.download_artifacts(
                        run_id=run.id, path="df_importance_rsf", dst_path=dir_artifacts
                    )
                    os.rename(
                        os.path.join(dir_artifacts, "df_importance_rsf"),
                        os.path.join(
                            dir_artifacts,
                            os.path.basename("df_importance_rsf" +
                                             "_" +
                                             data.params["model_path"].split(".json")[0].split("_")[-2] +
                                             "_" +
                                             data.params["model_path"].split(".json")[0].split("_")[-1] +
                                             ".json"),
                        ),
                    )
                except:
                    if verbose is True:
                        print(f"Parent run name for model with missing artifact: {parent_run_name}")
            else:
                continue
    print(len(l))
    df_metrics = pd.DataFrame(l)
    return df_metrics

def compare_random_states(df1, df2, name1, name2):
    list1 = sorted(df1['random_state'].unique().tolist())
    list2 = sorted(df2['random_state'].unique().tolist())
    condition = list1 == list2
    
    if not condition:
        print(f"Comparison between {name1} and {name2}:")
        print("Unique in df1 but not in df2:", set(list1) - set(list2))
        print("Unique in df2 but not in df1:", set(list2) - set(list1))
        print()
    
    return condition

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_metric_boxplot(
    df_metrics, 
    metric, 
    time_groups, 
    color_groups, 
    hue_column=None, 
    ax=None
):
    """
    Plot a boxplot for a specified metric with specified groupings from the time_groups dictionary,
    using custom colors and including the metric name in the plot title.

    Parameters:
        df_metrics (DataFrame): DataFrame containing metrics data.
        metric (str): Name of the metric to plot.
        time_groups (dict): Dictionary with grouping names as keys and lists of models as values.
        color_groups (dict): Dictionary with grouping names as keys and hex colors as values.
        hue_column (str): Column name to use for hue.
        ax (matplotlib.axes.Axes): Axes object to plot on. If None, a new figure and axes are created.

    Returns:
        matplotlib.axes.Axes: The matplotlib Axes object with the boxplot.
    """
    # Ensure we're working with a copy to avoid SettingWithCopyWarning
    df_metrics = df_metrics.copy()
    
    # Create a new column 'Group' in df_metrics based on the time_groups dictionary
    group_map = {model: group for group, models in time_groups.items() for model in models}
    df_metrics['Group'] = df_metrics['model'].apply(lambda x: group_map.get(x, 'Other'))
    
    # Melt the dataframe to long format for seaborn
    id_vars = ['model', 'Group']
    if hue_column:
        id_vars.append(hue_column)
    df_metrics_long = df_metrics.melt(
        id_vars=id_vars, 
        value_vars=[metric], 
        var_name='Metric type', 
        value_name='Metric value'
    )
    
    # Order the DataFrame based on the group order specified in the dictionary
    order = [model for group in time_groups.values() for model in group]  # Flatten the list of models maintaining the group order
    df_metrics_long['order'] = df_metrics_long['model'].apply(lambda x: order.index(x) if x in order else len(order))
    df_metrics_long.sort_values('order', inplace=True)
    
    # Update color_groups to include missing keys with default colors
    if hue_column:
        unique_hue_values = df_metrics[hue_column].unique()
        for hue_value in unique_hue_values:
            if hue_value not in color_groups:
                color_groups[hue_value] = None  # Let seaborn assign default colors
    
    # If no Axes is provided, create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create the boxplot
    boxplot = sns.boxplot(
        data=df_metrics_long,
        x='model',
        y='Metric value',
        hue=hue_column if hue_column else 'Group',
        palette=color_groups if hue_column is None else None,  # Use default colors if hue_column is provided
        dodge=bool(hue_column),  # Only dodge if there's a hue column
        ax=ax
    )
    
    ax.set_xlabel('')
    ax.set_ylabel('Value over MCCV Simulations')
    ax.tick_params(axis='x', rotation=90)  # Rotate model names to vertical
    
    # Handle legend
    if hue_column:
        ax.legend(title=hue_column, loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)
    else:
        ax.legend(title='Group', loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)
    
    # Adjust y-axis limits and ticks based on the metric
    if metric in ['concordance_index_censored', 'concordance_index_ipcw', 'mean_cumulative_dynamic_auc']:
        ax.set_ylim([0.5, 1])
        ax.set_yticks(np.arange(0.5, 1.05, 0.05))  # Set yticks from 0.5 to 1 with a step of 0.05
    else:
        ax.set_ylim([0, 0.05])
        ax.set_yticks(np.arange(0, 0.06, 0.005))
    
    # Set the title of the plot to be the metric name
    ax.set_title(metric)
    
    return ax
    
def plot_features_violin(df_metrics, time_groups, color_groups, hue_column=None, figsize=(12, 6)):
    """
    Plot a violin plot for the number of features used by different models.

    Parameters:
        df_metrics (DataFrame): DataFrame containing metrics data.
        time_groups (dict): Dictionary with grouping names as keys and lists of models as values.
        color_groups (dict): Dictionary with grouping names as keys and hex colors as values.
        hue_column (str): Column name to use for hue.
        figsize (tuple): Figure size in inches.

    Returns:
        None
    """
    # Create a new column 'Group' in df_metrics based on the time_groups dictionary
    group_map = {model: group for group, models in time_groups.items() for model in models}
    df_metrics = df_metrics.copy()
    df_metrics['Group'] = df_metrics['model'].apply(lambda x: group_map.get(x, 'Other'))

    # Order the DataFrame based on the group order specified in the dictionary
    ordered_models = [model for group in time_groups.values() for model in group]  # Flatten the list of models maintaining the group order
    df_metrics['order'] = pd.Categorical(df_metrics['model'], categories=ordered_models, ordered=True)
    df_metrics.sort_values('order', inplace=True)

    # Update color_groups to include missing keys with default colors
    if hue_column:
        unique_hue_values = df_metrics[hue_column].unique()
        for hue_value in unique_hue_values:
            if hue_value not in color_groups:
                color_groups[hue_value] = None  # Let seaborn assign default colors

    plt.figure(figsize=figsize)
    violinplot = sns.violinplot(
        data=df_metrics, 
        x='model', 
        y='n_features_in', 
        hue=hue_column if hue_column else 'Group',
        palette=color_groups if hue_column is None else None,  # Use default colors if hue_column is provided
        dodge=bool(hue_column)
    )
    plt.xlabel('')
    plt.ylabel('Number of Features')
    plt.tick_params(axis='x', rotation=90)  # Rotate model names to vertical

    # Set y-axis ticks starting from 1 and every 3 units thereafter
    max_features = df_metrics['n_features_in'].max()
    y_ticks = list(range(1, max_features + 2, 3))
    plt.gca().set_yticks(y_ticks)

    plt.legend(title='Group', loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)
    plt.title('Number of Features Distribution')

    # Set y-axis limit to the maximum number of features
    plt.ylim(1, max_features + 1)

    plt.tight_layout()
    plt.show()
    
    
def print_metrics(result_censored, result_ipcw, score_brier, mean_auc):
  """
  Prints evaluation metrics for right-censored time-to-event data with alignment and rounding.

  Args:
      result_censored: Concordance index for right-censored data.
      result_ipcw: Concordance index based on inverse probability weights.
      score_brier: Integrated Brier Score (IBS).
      mean_auc: Cumulative/dynamic AUC.
  """
  max_length = max(len(metric) for metric in [
      "Concordance index for right-censored data",
      "Concordance index for right-censored data based on inverse probability of censoring weights",
      "Integrated Brier Score (IBS)",
      "Cumulative/dynamic AUC for right-censored time-to-event data",
  ])

  print(f"Concordance index for right-censored data{' ' * (max_length - len('Concordance index for right-censored data'))} {format(np.round(result_censored, 4), '.4f')}")
  print(f"Concordance index for right-censored data based on inverse probability of censoring weights{' ' * (max_length - len('Concordance index for right-censored data based on inverse probability of censoring weights'))} {format(np.round(result_ipcw, 4), '.4f')}")
  print(f"Integrated Brier Score (IBS){' ' * (max_length - len('Integrated Brier Score (IBS)'))} {format(np.round(score_brier, 4), '.4f')}")
  print(f"Cumulative/dynamic AUC for right-censored time-to-event data{' ' * (max_length - len('Cumulative/dynamic AUC for right-censored time-to-event data'))} {format(np.round(mean_auc, 4), '.4f')}")
    
