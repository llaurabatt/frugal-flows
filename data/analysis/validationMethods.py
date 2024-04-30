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

pandas2ri.activate()

utils = importr('utils')
utils = rpy2.robjects.packages.importr('utils')
utils.chooseCRANmirror(ind=1) # select the first mirror in the list
packnames = ('dbarts', 'grf')

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
    if replace:
        replace = 'TRUE'
    else:
        replace = 'FALSE'
    data.to_csv('data.csv',index=False)
    formula_cov = treatment+' ~ '
    i = 0
    for cov in data.columns:
        if cov!=outcome and cov!=treatment:
            if i!=0:
                formula_cov += '+' 
            formula_cov += str(cov)
            i += 1
    string = f"""
    library('MatchIt')
    data <- read.csv('data.csv')
    r <- matchit( {formula_cov},estimand="ATE", method = "{method}", data = data, replace = {replace})
    matrix <- r$match.matrix[,]
    names <- as.numeric(names(r$match.matrix[,]))
    mtch <- data[as.numeric(names(r$match.matrix[,])),]
    hh <- data[as.numeric(names(r$match.matrix[,])),'{outcome}']- data[as.numeric(r$match.matrix[,]),'{outcome}']
    
    data2 <- data
    data2${treatment} <- 1 - data2${treatment}
    r2 <- matchit( {formula_cov}, estimand="ATE", method = "{method}", data = data2, replace = {replace})
    matrix2 <- r2$match.matrix[,]
    names2 <- as.numeric(names(r2$match.matrix[,]))
    mtch2 <- data2[as.numeric(names(r2$match.matrix[,])),]
    hh2 <- data2[as.numeric(r2$match.matrix[,]),'{outcome}'] - data2[as.numeric(names(r2$match.matrix[,])),'{outcome}']
    """#%( formula_cov,method,replace,outcome,outcome, treatment, treatment, formula_cov,method,replace,outcome,outcome)
    
    psnn = SignatureTranslatedAnonymousPackage(string, "powerpack")
    match = psnn.mtch
    match2 = psnn.mtch2
    t_hat = pd.DataFrame(np.hstack((np.array(psnn.hh),np.array(psnn.hh2))),
                         index=list(psnn.names.astype(int))+list(psnn.names2.astype(int)),
                         columns=['CATE'])
    ate = np.mean(t_hat['CATE'])
    return ate


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
    cate_est['std.CATE'] = cate_est.std(axis=1) 
    ate_est = cate_est['avg.CATE'].mean(axis=0)
    return ate_est


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
    return cate_est['avg.CATE'].mean()

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
    if method=='linear':
        models = RidgeCV()
        propensity_model = LogisticRegressionCV()
    if method=='GBR':
        models = GradientBoostingRegressor()
        propensity_model = GradientBoostingClassifier() 
    if est=='T':
        T_learner = TLearner(models=models)
        T_learner.fit(data[outcome], data[treatment], X=data.drop(columns=[outcome,treatment]))
        point = T_learner.ate(X=data.drop(columns=[outcome,treatment]))
    elif est=='S':
        S_learner = SLearner( overall_model=models)
        S_learner.fit(data[outcome], data[treatment], X=data.drop(columns=[outcome,treatment]))
        point = S_learner.ate(X=data.drop(columns=[outcome,treatment]))
    elif est=='X':
        X_learner = XLearner(models=models,propensity_model=propensity_model)
        X_learner.fit(data[outcome], data[treatment], X=data.drop(columns=[outcome,treatment]))
        point = X_learner.ate(X=data.drop(columns=[outcome,treatment]))
    return point


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
            W=data.drop(columns=[outcome, treatment])
        )
        point = est.ate(data.drop(columns=[outcome, treatment]), T0=0, T1=1)
    if method == 'linear':
        est = LinearDML(discrete_treatment=True)
        est.fit(
            data[outcome],
            data[treatment],
            X=data.drop(columns=[outcome, treatment]),
            W=data.drop(columns=[outcome, treatment])
        )
        point = est.ate(data.drop(columns=[outcome, treatment]), T0=0, T1=1)
    return point


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
        W = None
        #W=data.drop(columns=[outcome, treatment])
    )
    point = est.ate(data.drop(columns=[outcome, treatment]), T0=0, T1=1)
    return point


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
    return tml.average_treatment_effect


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
    return data.loc[data[treatment] == 1, outcome].mean() - data.loc[data[treatment] == 0, outcome].mean()


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
    ate = {}
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        ate['Diff. of Mean'] = diff_means(outcome, treatment, data)
        ate['Gradient Boosting Trees DML'] = dml(outcome, treatment, data, method='GBR')
        ate['Linear DML'] = dml(outcome, treatment, data, method='linear')
        ate['Doubly Robust (Linear)'] = doubleRobust(outcome, treatment, data)
        ate['Linear T Learner'] = metalearner(outcome, treatment, data, est='T', method='linear')
        ate['Linear S Learner'] = metalearner(outcome, treatment, data, est='S', method='linear')
        ate['Linear X Learner'] = metalearner(outcome, treatment, data, est='X', method='linear')
        ate['Gradient Boosting Trees T Learner'] = metalearner(outcome, treatment, data, est='T', method='GBR')
        ate['Gradient Boosting Trees S Learner'] = metalearner(outcome, treatment, data, est='S', method='GBR')
        ate['Gradient Boosting Trees X Learner'] = metalearner(outcome, treatment, data, est='X', method='GBR')
        ate['Causal BART'] = bart(outcome, treatment, data)
        ate['Causal Forest'] = causalforest(outcome, treatment, data)
        # ate['Propensity Score Matching'] = matchit(outcome, treatment, data, method='full')
        ate['TMLE'] = tmle(outcome, treatment, data)
        return pd.DataFrame.from_dict(ate, orient='index').T


def bootstrap_ate_inference(outcome: str, treatment: str, data: pd.DataFrame, repeats: int, sample_frac: float, replace: bool) -> pd.DataFrame:
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
    results = []
    for _ in tqdm.tqdm(range(repeats), leave=False):
        data_ = data.sample(frac=sample_frac, replace=replace)
        ate_ = estimate_ate(outcome, treatment, data_.reset_index(drop=True))
        results.append(ate_)
    return pd.concat(results, axis=0)