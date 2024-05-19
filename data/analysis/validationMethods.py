import os
import contextlib
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from econml.dml import NonParamDML, LinearDML
from econml.dr import DRLearner
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, RidgeCV, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

import zepid
from zepid.causal.doublyrobust import TMLE

import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.vectors import StrVector
import logging


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


pandas2ri.activate()

utils = importr('utils')
utils = rpy2.robjects.packages.importr('utils')
utils.chooseCRANmirror(ind=1) # select the first mirror in the list
packnames = ('dbarts', 'grf', 'marginaleffects')

names_to_install = [x for x in packnames if not rpy2.robjects.packages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))


def matchit(outcome: str, treatment: str, data: pd.DataFrame, method: str = 'full', distance: str = 'glm', replace: bool = False) -> float:   
    """
    Perform propensity score matching using the MatchIt package in R.

    Parameters:
    outcome (str): The name of the outcome variable.
    treatment (str): The name of the treatment variable.
    data (pandas.DataFrame): The input data containing the outcome, treatment, and covariates.
    method (str, optional): The matching method to use. Defaults to 'nearest'.
    distance (str, optional): The distance metric to use. Defaults to 'glm'.
    replace (bool, optional): Whether to replace treated units in the control group. Defaults to False.

    Returns:
    float: The average treatment effect (ATE) estimated by the matching.

    """
    covariates = [col for col in data.columns if col not in [outcome, treatment]]
    if replace:
        replace = 'TRUE'
    else:
        replace = 'FALSE'
    data.to_csv('data.csv',index=False)
    matching_formula= treatment + ' ~ ' + ' + '.join(covariates)
    fitting_formula= outcome + ' ~ ' + treatment + ' * (' + ' + '.join(covariates) + ')'

    r_code_string = f"""
    library(MatchIt)
    library(marginaleffects)

    data <- read.csv('data.csv')

    control_data <- data[data[['{treatment}']] == 0, ]
    treatment_data <- data[data[['{treatment}']] == 1, ]

    m.out <- matchit(
        {matching_formula},
        data = data,
        method = "full",
        estimand = "ATE",
        distance = "glm",
        replace = {replace}
    )
    matched_data <- match.data(m.out)

    fit <- lm(as.formula({fitting_formula}), data = matched_data, weights = weights)
    ate_fit <- avg_comparisons(fit,
                variables = '{treatment}',
                vcov = ~subclass,
                newdata = subset(matched_data, X == 1),
                wts = "weights")
    ate <- ate_fit$estimate
    ate_std <- ate_fit$std.error
    """
    psnn = SignatureTranslatedAnonymousPackage(r_code_string, "powerpack")
    ate = psnn.ate
    ate_std = psnn.ate_std
    lower_conf, upper_conf = (ate_std * -1.96, ate_std * +1.96)
    return ate, lower_conf, upper_conf


def bart(outcome: str, treatment: str, data: pd.DataFrame) -> float:
    """
    Perform Bayesian Additive Regression Trees (BART) for causal estimation.

    Parameters:
    outcome (str): The name of the outcome variable.
    treatment (str): The name of the treatment variable.
    data (pd.DataFrame): The input data containing the outcome, treatment, and covariates.

    Returns:
    ate_est (float): The causal estimand results.
    """
    utils = importr('utils')
    dbarts = importr('dbarts')
    cate_est = pd.DataFrame()

    covariates = [col for col in data.columns if col not in [outcome, treatment]]

    # Split the data into control and treatment groups
    control_data = data[data[treatment] == 0]
    treatment_data = data[data[treatment] == 1]

    # Extract the covariates and outcomes for the control and treatment groups
    X_control = control_data[covariates].values
    Y_control = control_data[outcome].values

    X_treatment = treatment_data[covariates].values
    Y_treatment = treatment_data[outcome].values

    # Fit BART models for the control and treatment groups
    bart_model_control = dbarts.bart(X_control, Y_control, X_treatment, keeptrees=True, verbose=False)
    bart_model_treatment = dbarts.bart(X_treatment, Y_treatment, X_control, keeptrees=True, verbose=False)

    # Compute the predicted outcomes for the control and treatment groups
    predicted_outcomes_control = np.concatenate((Y_control, np.array(bart_model_control[7])))
    predicted_outcomes_treatment = np.concatenate((np.array(bart_model_treatment[7]), Y_treatment))

    # Compute the treatment effect
    treatment_effect = predicted_outcomes_treatment - predicted_outcomes_control

    # Store the treatment effect in the CATE DataFrame
    cate_est['CATE'] = treatment_effect
    cate_est['avg.CATE'] = cate_est.mean(axis=1)
    cate_est['std.CATE'] = cate_est.std(axis=1) # This doesn't really work, need to adjust the uncertainty 
    ate= cate_est['avg.CATE'].mean(axis=0)

    # lower_conf, upper_conf = (ate_std * -1.96, ate_std * +1.96)
    lower_conf, upper_conf = (None, None) # Need to fix
    return ate, lower_conf, upper_conf


def causalforest(outcome: str, treatment: str, data: pd.DataFrame, n_splits: int = 5) -> float:
    """
    Perform causal forest for estimating the Causal Average Treatment Effect (CATE).

    Parameters:
    outcome (str): The name of the outcome variable.
    treatment (str): The name of the treatment variable.
    data (pd.DataFrame): The input data containing the outcome, treatment, and covariates.
    n_splits (int, optional): The number of splits for cross-validation. Defaults to 5.

    Returns:
    float: The average CATE estimated by causal forest.
    """    
    grf = importr('grf')

    # Initialize StratifiedKFold and generate splits
    skf = StratifiedKFold(n_splits=n_splits)
    gen_skf = skf.split(data, data[treatment])

    # Initialize DataFrame for CATE estimates and determine covariates
    cate_est = pd.DataFrame()
    covariates = [col for col in data.columns if col not in [outcome, treatment]]

    # Loop over each split
    for train_idx, est_idx in gen_skf:
        # Split the data into training and estimation sets
        df_train = data.iloc[train_idx]
        df_est = data.iloc[est_idx]

        # Define the outcome, treatment, and covariates for the training set
        Ycrf = df_train[outcome]
        Tcrf = df_train[treatment]
        X = df_train[covariates]

        # Define the covariates for the estimation set
        Xtest = df_est[covariates]

        # Fit the causal forest model and predict the CATE
        crf = grf.causal_forest(X, Ycrf, Tcrf)
        tauhat = grf.predict_causal_forest(crf, Xtest)

        # Convert the CATE estimates to a DataFrame and append to the overall CATE estimates
        t_hat_crf = np.array(tauhat[0])
        cate_est_i = pd.DataFrame(t_hat_crf, index=df_est.index, columns=['CATE'])
        cate_est = pd.concat([cate_est, cate_est_i], axis=1)

    # Compute the average and standard deviation of the CATE estimates
    cate_est['avg.CATE'] = cate_est.mean(axis=1)
    cate_est['std.CATE'] = cate_est.std(axis=1)

    # Return the average of the average CATE estimates
    ate = cate_est['avg.CATE'].mean()
    ate_std = cate_est['std.CATE']
    lower_conf, upper_conf = (ate_std * -1.96, ate_std * +1.96)
    return ate, lower_conf, upper_conf


def metalearner(outcome: str, treatment: str, data: pd.DataFrame, est: str = 'T', method: str = 'linear') -> float:
    """
    Perform meta-learner estimation for estimating the Average Treatment Effect (ATE).

    Parameters:
    outcome (str): The name of the outcome variable.
    treatment (str): The name of the treatment variable.
    data (pd.DataFrame): The input data containing the outcome, treatment, and covariates.
    est (str, optional): The type of meta-learner to use. Defaults to 'T'.
    method (str, optional): The base learner method to use. Defaults to 'linear'.

    Returns:
    point (float): The estimated ATE by the meta-learner.
    """
    # Choose models
    if method=='linear':
        models = RidgeCV()
        propensity_model = LogisticRegressionCV()
    if method=='GBR':
        models = GradientBoostingRegressor()
        propensity_model = GradientBoostingClassifier() 
    
    # Choose metalearner
    if est=='T':
        learner = TLearner(models=models)
        learner.fit(data[outcome], data[treatment], X=data.drop(columns=[outcome,treatment]), inference='bootstrap') 
    elif est=='S':
        learner = SLearner(overall_model=models)
        learner.fit(data[outcome], data[treatment], X=data.drop(columns=[outcome,treatment]), inference='bootstrap') 
    elif est=='X':
        learner = XLearner(models=models, propensity_model=propensity_model)
        learner.fit(data[outcome], data[treatment], X=data.drop(columns=[outcome,treatment]), inference='bootstrap') 

    results = learner.ate_inference(T0=0, T1=1,  X=data.drop(columns=[outcome, treatment]))
    ate_point = results.mean_point
    lower_conf, upper_conf = results.conf_int_mean()
    return ate_point, (lower_conf, upper_conf)


def dml(outcome: str, treatment: str, data: pd.DataFrame, method: str = 'GBR') -> float:
    """
    Perform Double Machine Learning (DML) for estimating the Average Treatment Effect (ATE).

    Parameters:
    outcome (str): The name of the outcome variable.
    treatment (str): The name of the treatment variable.
    data (pd.DataFrame): The input data containing the outcome, treatment, and covariates.
    method (str, optional): The base learner method to use. Defaults to 'GBR'.

    Returns:
    float: The estimated ATE by DML.
    """
    if method == 'GBR':
        est = NonParamDML(
            model_y=GradientBoostingRegressor(),
            model_t=GradientBoostingClassifier(),
            model_final=GradientBoostingRegressor(),
            discrete_treatment=True
        )
        est.fit(
            data[outcome],
            data[treatment],
            X=data.drop(columns=[outcome, treatment]),
            W=data.drop(columns=[outcome, treatment]),
            inference='bootstrap'
        )
    if method == 'linear':
        est = LinearDML(discrete_treatment=True)
        est.fit(
            data[outcome],
            data[treatment],
            X=data.drop(columns=[outcome, treatment]),
            W=data.drop(columns=[outcome, treatment]),
            inference='bootstrap'
        )
    results = est.ate_inference(T0=0, T1=1,  X=data.drop(columns=[outcome, treatment]))
    ate_point = results.mean_point
    lower_conf, upper_conf = results.conf_int_mean()
    return ate_point, (lower_conf, upper_conf)


def doubleRobust(outcome: str, treatment: str, data: pd.DataFrame) -> float:
    """
    Perform Double Robust (DR) estimation for estimating the Average Treatment Effect (ATE).

    Parameters:
    outcome (str): The name of the outcome variable.
    treatment (str): The name of the treatment variable.
    data (pd.DataFrame): The input data containing the outcome, treatment, and covariates.

    Returns:
    point (float): The estimated ATE by DR.
    """
    est = DRLearner()
    est.fit(
        data[outcome],
        data[treatment],
        X = data.drop(columns=[outcome, treatment]),
        W = None,
        #W=data.drop(columns=[outcome, treatment])
        inference='bootstrap'
    )
    results = est.ate_inference(T0=0, T1=1,  X=data.drop(columns=[outcome, treatment]))
    ate = results.mean_point
    lower_conf, upper_conf = results.conf_int_mean()
    return ate, lower_conf, upper_conf


def tmle(outcome: str, treatment: str, data: pd.DataFrame) -> float:
    """
    Perform Targeted Maximum Likelihood Estimation (TMLE) for estimating the Average Treatment Effect (ATE).

    Parameters:
    outcome (str): The name of the outcome variable.
    treatment (str): The name of the treatment variable.
    data (pd.DataFrame): The input data containing the outcome, treatment, and covariates.

    Returns:
    average_treatment_effect (float): The estimated ATE by TMLE.
    """
    tml = TMLE(data, exposure=treatment, outcome=outcome)
    cols = data.drop(columns=[outcome, treatment]).columns
    s = str(cols[0])
    for j in range(1, len(cols)):
        s = s + ' + ' + str(cols[j])
    tml.exposure_model(s)
    tml.outcome_model(s)
    tml.fit()
    ate = tml.average_treatment_effect
    lower_conf, upper_conf = tml.average_treatment_effect_ci # 95th percentile
    return ate, lower_conf, upper_conf


def diff_means(outcome: str, treatment: str, data: pd.DataFrame) -> float:
    """
    Calculate the difference in means between the treatment and control groups.

    Parameters:
    outcome (str): The name of the outcome variable.
    treatment (str): The name of the treatment variable.
    data (pd.DataFrame): The input data containing the outcome, treatment, and covariates.

    Returns:
    float: The difference in means.
    """
    ate = data.loc[data[treatment] == 1, outcome].mean() - data.loc[data[treatment] == 0, outcome].mean()
    lower_conf = None
    upper_conf = None
    return ate, lower_conf, upper_conf


def estimate_ate(outcome: str, treatment: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the Average Treatment Effect (ATE) using various methods.

    Parameters:
    outcome (str): The name of the outcome variable.
    treatment (str): The name of the treatment variable.
    data (pd.DataFrame): The input data containing the outcome, treatment, and covariates.

    Returns:
    pd.DataFrame: A DataFrame containing the estimated ATEs for different methods.
    """
    results = []
    column_headers = ['ate', 'lower_conf', 'upper_conf']
    method_names = pd.DataFrame({'method': [
        # 'Diff. of Mean',
        # 'Gradient Boosting Trees DML',
        'Linear DML',
        # 'Doubly Robust (Linear)',
        'Linear T Learner',
        'Linear S Learner',
        'Linear X Learner',
        'GBT T Learner',
        'GBT S Learner',
        'GBT X Learner',
        #'Causal BART',
        #'Causal Forest',
        'Prop. Score Matching',
        'TMLE'
    ]})
    #with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    #results.append(diff_means(outcome, treatment, data))
    # Set up logging

    # Replace the print statement with logging
    #logging.info(f"Fitting model: {method_names.iloc[0, 0]}")
    #results.append(dml(outcome, treatment, data, method='GBR'))
    logging.info(f"Fitting model: {method_names.iloc[1, 0]}")
    results.append(dml(outcome, treatment, data, method='linear'))
    #logging.info(f"Fitting model: {method_names.iloc[2, 0]}")
    #results.append(doubleRobust(outcome, treatment, data))
    logging.info(f"Fitting model: {method_names.iloc[3, 0]}")
    results.append(metalearner(outcome, treatment, data, est='T', method='linear'))
    logging.info(f"Fitting model: {method_names.iloc[4, 0]}")
    results.append(metalearner(outcome, treatment, data, est='S', method='linear'))
    logging.info(f"Fitting model: {method_names.iloc[5, 0]}")
    results.append(metalearner(outcome, treatment, data, est='X', method='linear'))
    logging.info(f"Fitting model: {method_names.iloc[6, 0]}")
    results.append(metalearner(outcome, treatment, data, est='T', method='GBR'))
    logging.info(f"Fitting model: {method_names.iloc[7, 0]}")
    results.append(metalearner(outcome, treatment, data, est='S', method='GBR'))
    logging.info(f"Fitting model: {method_names.iloc[8, 0]}")
    results.append(metalearner(outcome, treatment, data, est='X', method='GBR'))
    # logging.info(f"Fitting model: {method_names.iloc[0, 0]}")
    # results.append(bart(outcome, treatment, data))
    # logging.info(f"Fitting model: {method_names.iloc[0, 0]}")
    # results.append(causalforest(outcome, treatment, data))
    logging.info(f"Fitting model: {method_names.iloc[9, 0]}")
    results.append(matchit(outcome, treatment, data, method='full'))
    logging.info(f"Fitting model: {method_names.iloc[10, 0]}")
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        results.append(tmle(outcome, treatment, data))
    results_df = pd.DataFrame(results, columns=column_headers)
    return pd.concat([method_names, results_df], axis=1)


def bootstrap_estimate_ate(outcome: str, treatment: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the Average Treatment Effect (ATE) using various methods.

    Parameters:
    outcome (str): The name of the outcome variable.
    treatment (str): The name of the treatment variable.
    data (pd.DataFrame): The input data containing the outcome, treatment, and covariates.

    Returns:
    pd.DataFrame: A DataFrame containing the estimated ATEs for different methods.
    """
    results = []
    column_headers = ['ate', 'lower_conf', 'upper_conf']
    method_names = pd.DataFrame({'method': [
        'Diff. of Mean',
        #'Gradient Boosting Trees DML',
        #'Linear DML',
        #'Doubly Robust (Linear)',
        #'Linear T Learner',
        #'Linear S Learner',
        #'Linear X Learner',
        #'Gradient Boosting Trees T Learner',
        #'Gradient Boosting Trees S Learner',
        #'Gradient Boosting Trees X Learner',
        'Causal BART',
        'Causal Forest',
        #'Propensity Score Matching',
        #'TMLE'
    ]})
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        results.append(diff_means(outcome, treatment, data))
        # results.append(dml(outcome, treatment, data, method='GBR'))
        # results.append(dml(outcome, treatment, data, method='linear'))
        # results.append(doubleRobust(outcome, treatment, data))
        # results.append(metalearner(outcome, treatment, data, est='T', method='linear'))
        # results.append(metalearner(outcome, treatment, data, est='S', method='linear'))
        # results.append(metalearner(outcome, treatment, data, est='X', method='linear'))
        # results.append(metalearner(outcome, treatment, data, est='T', method='GBR'))
        # results.append(metalearner(outcome, treatment, data, est='S', method='GBR'))
        # results.append(metalearner(outcome, treatment, data, est='X', method='GBR'))
        results.append(bart(outcome, treatment, data))
        results.append(causalforest(outcome, treatment, data))
        #results.append(matchit(outcome, treatment, data, method='full'))
        #results.append(tmle(outcome, treatment, data))
    results_df = pd.DataFrame(results, columns=column_headers)
    return pd.concat([method_names, results_df], axis=1)

def run_model_fits(outcome: str, treatment: str, data: pd.DataFrame, repeats: int, sample_frac: float, replace: bool) -> pd.DataFrame:
    """
    Perform bootstrap inference for estimating the Average Treatment Effect (ATE).

    Parameters:
    outcome (str): The name of the outcome variable.
    treatment (str): The name of the treatment variable.
    data (pd.DataFrame): The input data containing the outcome, treatment, and covariates.
    repeats (int, optional): The number of bootstrap repeats. Defaults to 10.

    Returns:
    pd.DataFrame: A DataFrame containing the estimated ATEs for each bootstrap iteration.
    """
    non_bootstrap_results_df = estimate_ate(outcome, treatment, data)
    bootstrap_results = []
    for _ in tqdm.tqdm(range(repeats), leave=False):
        data_ = data.sample(frac=sample_frac, replace=replace)
        ate_ = bootstrap_estimate_ate(outcome, treatment, data_.reset_index(drop=True))
        bootstrap_results.append(ate_)
    bootstrap_results_df = pd.concat(bootstrap_results, axis=0)
    return {
        'bootstrap_results': bootstrap_results_df,
        'nonbootstrap_results': non_bootstrap_results_df
    }