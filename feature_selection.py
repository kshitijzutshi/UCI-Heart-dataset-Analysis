import argparse
import sys
import json
import shap
import numpy  as np
import pandas as pd 
import lightgbm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold,train_test_split, cross_val_score
from shaphypetune import BoostSearch, BoostBoruta, BoostRFE
from xgboost import XGBRegressor, plot_importance
from sklearn.inspection import permutation_importance
from sklearn.datasets   import load_boston
from sklearn.ensemble   import RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import sklearn
import seaborn as sns



def recursive_feature_evalutation(X, y, feature_names):
    report_body = {}
    model = sklearn.linear_model.LinearRegression()
    #Initializing RFE model
    rfe = RFE(model, 10)

    #Transforming data using RFE
    X_rfe = rfe.fit_transform(X,y)  

    #Fitting the data to model
    model.fit(X_rfe,y)
    ranks = pd.Series(data=rfe.ranking_, index=feature_names)
    ranks = pd.DataFrame(data=[ranks], index=["rfe"])
    
    temp = pd.Series(rfe.support_, index=feature_names)
    selected_features_rfe = temp[temp==True].index.values
    report_body["evaluation"] = ranks.to_dict()
    report_body["picked_features"] = list(selected_features_rfe)
    
    return report_body


def l1_regularisation(X, y, feature_names):
    report_body = {}
    reg = LassoCV()
    reg.fit(X, y)
    coef = pd.Series(reg.coef_, index=feature_names)
    report_ranks = pd.DataFrame(data=[coef], index=["lr_coeff"]).to_dict()
    report_body["evaluation"] = report_ranks
    report_body["picked_features"] = list(coef[coef != 0].index.values)
    return report_body


def backward_elimination(X, y, feature_names):
    feature_names = feature_names.copy()
    X_1 = sm.add_constant(X)
    # Fitting sm.OLS model
    model = sm.OLS(y, X_1).fit()
    p_values = model.pvalues
    
    report_body = {}
    
    pmax = 1
    while (len(feature_names) > 0):
        p = []
        X_1 = X[feature_names]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y, X_1).fit()
        p = pd.Series(model.pvalues.values[1:], index=feature_names)      
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax > 0.05):
            feature_names.remove(feature_with_p_max)
        else:
            break
            
    p_val_df = pd.DataFrame(data=[p_values], index=["p_value"])
    report_body["evaluation"] = p_val_df.to_dict()
    report_body["picked_features"] = feature_names
    return report_body
    
def pearson_correlation(X, y, feature_names):
    report_body = {}
    # Get F-Statistics and p-Values for each feature
    f_stats, p_values = sklearn.feature_selection.f_regression(X, y, center=False)
    f_stats = pd.Series(data=f_stats, index=X.columns.values)
    p_values = pd.Series(data=p_values, index=X.columns.values)
    df_stats = pd.DataFrame(data=[f_stats, p_values], index=["f_stat", "p_value"])
    # Get Pearson Correlations for each feature
    pearson_correlation = pd.concat([X, y], axis=1).corr(method="pearson")
    # Get Pearson Correlations only for Target
    cor_target = abs(pearson_correlation[args.target_name])
    relevant_features = cor_target[cor_target > 0.4]
    
    report_body["correlation_matrix"] = pearson_correlation.to_dict()
    report_body["evaluation"] = df_stats.to_dict()
    report_body["picked_features"] = list(relevant_features.index.values)
    return report_body


def extra_tree_importance(X, y, feature_names):
    report_body = {}
    model = ExtraTreesRegressor()
    model.fit(X, y.values)
    report_body["evaluation"] = pd.DataFrame(
        data=[pd.Series(model.feature_importances_, index=feature_names)],
        index=["extra_tree_importance"]
    ).to_dict()

    return report_body

def xgb(Xtrain, ytrain, feature_names):
    report_body = {}
    xgb = XGBRegressor(n_estimators=100)
    xgb.fit(Xtrain, ytrain)
    sorted_idx = xgb.feature_importances_.argsort()
    feature_importances = pd.Series(data=xgb.feature_importances_[sorted_idx], index=Xtrain.columns[sorted_idx])
    picked_features = list(feature_importances[feature_importances >= 0.05])
    report_body["evaluation"] = pd.DataFrame(
        data=[feature_importances],
        index=["xgb_importance"]
    ).to_dict()
    report_body["picked_features"] = picked_features
    
    return report_body


def permutation_based(Xtrain, ytrain, Xtest, ytest, feature_names):
    report_body = {}
    xgb = XGBRegressor(n_estimators=100)
    xgb.fit(Xtrain, ytrain)
    perm_importance = permutation_importance(xgb, Xtest, ytest).importances_mean
    sorted_idx = perm_importance.argsort()

    feature_importances = pd.Series(data=perm_importance, index=Xtrain.columns[sorted_idx])
    picked_features = list(feature_importances[feature_importances >= 0.05])
    
    report_body["evaluation"] = pd.DataFrame(
        data=[feature_importances],
        index=["permutation_based_importance"]
    ).to_dict()
    report_body["picked_features"] = picked_features
    
    return report_body


def shap_importance(Xtrain, ytrain, Xtest, ytest, feature_names):
    report_body = {}
    xgb = XGBRegressor(n_estimators=100)
    xgb.fit(Xtrain, ytrain)
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(Xtest)
    shap_values = np.abs(shap_values).mean(0)
    shap_values = pd.Series(data=shap_values, index=feature_names)
    
    report_body["evaluation"] = pd.DataFrame(
        data=[shap_values],
        index=["shap_importance"]
    ).to_dict()
    return report_body
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='Path to the dataset to process')
    parser.add_argument('--target_name', help="Name of the target variable")
    args = parser.parse_args()
    
    
    df = pd.read_csv(args.dataset_path)
    features = list(filter(lambda feature: feature != args.target_name, df.columns.values))
    X = df[features]
    y = df[[args.target_name]]
    
    report = {}
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
    
    report["pearson_correlation"] = pearson_correlation(X, y, features)
    report["rfe"] = recursive_feature_evalutation(X, y, features)
    report["l1_regularisation"] = l1_regularisation(X, y, features)
    report["backward_elimination"] = backward_elimination(X, y, features)
    report["extra_tree_importance"] = extra_tree_importance(X, y, features)
    report["xgb_importance"] = xgb(Xtrain, ytrain, features)
    report["permutation_based_importance"] = permutation_based(Xtrain, ytrain, Xtest, ytest, features)
    report["shap_importance"] = shap_importance(Xtrain, ytrain, Xtest, ytest, features)
    
    with open('report.json', 'w+') as f:
        json.dump(report, f)

