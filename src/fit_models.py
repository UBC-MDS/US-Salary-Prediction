# author: Andy Yang
# date: 2021-11-27

"""This script transforms the cleaned data and fits a Ridge and a RandomForestRegressor model to the preprocessed train data. 
This script also pre-processes test data and outputs it as a sparse matrix.

Usage: fit_model.py --source_data=<filepath> --output_dir=<filepath>

Options:
--source_data=<filepath>     directory containing transformed data (this is a required option)
--output_dir=<filepath>      directory to output figures and tables (this is a required option)
""" 

from docopt import docopt
import random
import numpy as np
import pandas as pd
import altair as alt
import sklearn.metrics as sk
import math
import pickle
import scipy.sparse

from sklearn.model_selection import train_test_split
from altair_saver import save
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import RFE, RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    ShuffleSplit,
    cross_val_score,
    cross_validate,
    train_test_split,
)

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from sklearn.pipeline import Pipeline, make_pipeline

opt = docopt(__doc__)

def main(opt):
    input_dir = opt['--source_data']
    output_dir = opt['--output_dir']

    # read in data
    train_df = pd.read_csv(input_dir + '/train_df.csv')
    test_df = pd.read_csv(input_dir + '/test_df.csv')
    
    # pre-process train data
    # remove outliers and rare data from train dataset
    train_df = train_df[(train_df["annual_salary"] < 1000000) & (train_df["annual_salary"] > 10000)]
    train_df["state"] = train_df["state"].mask(train_df["state"].map(train_df["state"].value_counts()) < 5, "")
    train_df["city"] = train_df["city"].mask(train_df["city"].map(train_df["city"].value_counts()) < 5, "")
    test_df.fillna({"industry":"", "job_title":""}, inplace=True)

    # replace nan values with blanks in industry and job_title features
    train_df.fillna({"industry":"", "job_title":""}, inplace=True)

    # split into X and y
    X_train, y_train = train_df.drop(columns=["annual_salary"]), train_df["annual_salary"]
    X_test, y_test = test_df.drop(columns=["annual_salary"]), test_df["annual_salary"]

    # define feature types
    categorical_features = ["state", "city"]
    text_feature1 = "industry"
    text_feature2 = "job_title"
    numeric_features = ["other_monetary_comp"]
    drop_features = ["additional_context_on_job_title", "additional_context_on_income"]
    ordinal_features = ["how_old_are_you", "overall_years_of_professional_experience",
                        "years_of_experience_in_field", "highest_level_of_education_completed"]

    # create lists for ordinal encoding
    age_order = ["under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65 or over"]
    exp_order = ["1 year or less", "2 - 4 years", "5-7 years", "8 - 10 years",
                "11 - 20 years", "21 - 30 years", "31 - 40 years", "41 years or more"]
    edu_order = ["High School", "Some college", "College degree",
                "Master's degree", "Professional degree (MD, JD, etc.)",
                "PhD"]

    # construct preprocessor
    preprocessor = make_column_transformer(
        (
            make_pipeline(
                SimpleImputer(strategy="constant"),
                OneHotEncoder(handle_unknown="ignore", sparse=False, dtype ="int")
            ),
            categorical_features
        ),
        (
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0),
                StandardScaler()
            ),
            numeric_features
        ),
        (
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OrdinalEncoder(categories=[age_order, exp_order, exp_order, edu_order], dtype=int),
                StandardScaler()
            ),
            ordinal_features
        ),
        (
            CountVectorizer(max_features=5000, stop_words="english"),
            text_feature1
        ),
        (
            CountVectorizer(max_features=5000, stop_words="english"),
            text_feature2
        )
    )

    # transform data
    print("pre-processing data.. ")
    X_transformed_train = preprocessor.fit_transform(X_train)
    X_transformed_test = preprocessor.transform(X_test)

    # extract feature names
    feats =\
        list(preprocessor.transformers_[0][1][1].get_feature_names_out(categorical_features)) +\
        numeric_features + ordinal_features +\
        list(preprocessor.transformers_[3][1].get_feature_names_out()) +\
        list(preprocessor.transformers_[4][1].get_feature_names_out())

    X_transformed_wcoef_test = pd.DataFrame(X_transformed_test.todense(), columns = feats)
    
    # output preprocessed test data
    x_test_sparse = scipy.sparse.csc_matrix(X_transformed_wcoef_test)
    scipy.sparse.save_npz(input_dir + '/x_test_sparse.npz', x_test_sparse)
    y_test.to_csv(input_dir + '/y_test.csv', index = False)
    print("processed test data saved to: " + input_dir)
    

    # define scoring metrics
    score_types_reg = {
        "neg_root_mean_squared_error": "neg_root_mean_squared_error",
        "r2": "r2"
    }

    # conduct hyperparameter optimization on Ridge and save the results
    print("conducting hyperparameter optimization for Ridge..")
    param_grid_cgamma = {"alpha": np.logspace(-5, 5, 20)}

    random_search_ridge = RandomizedSearchCV(
        Ridge(),
        param_distributions=param_grid_cgamma,
        verbose=1,
        n_jobs=-1,
        n_iter=20,
        cv=5,
        random_state=123,
        scoring=score_types_reg,
        refit=False)

    random_search_ridge.fit(X_transformed_train, y_train)

    gridsearch_results_ridge = pd.DataFrame(random_search_ridge.cv_results_)[
        [
            "mean_test_r2",
            "mean_test_neg_root_mean_squared_error",
            "param_alpha",
            "rank_test_r2"
        ]
    ].set_index("rank_test_r2").sort_index()

    gridsearch_results_ridge.columns = ["r2", "Negative RMSE", "alpha"]
    gridsearch_results_ridge.index.names = ["r2 score rank"]
    gridsearch_results_ridge.to_csv(output_dir + '/tables/grid_search_ridge.csv', index = False)
    print("saved grid search results for ridge model to: " + output_dir)

    # conduct hyperparameter optimization on RandomForest and save the results  
    print("conducting hyperparameter optimization for RandomForest (please be patient)..")
    param_grid_n_estimators = {"n_estimators": [10, 20, 50, 100]}

    random_search_forest = GridSearchCV(
        RandomForestRegressor(),
        param_grid=param_grid_n_estimators,
        return_train_score=True,
        verbose=1,
        n_jobs=-1,
        cv=3,
        scoring=score_types_reg,
        refit=False)

    random_search_forest.fit(X_transformed_train, y_train)

    gridsearch_results_rf = pd.DataFrame(random_search_forest.cv_results_)[
        [
            "mean_test_r2",
            "mean_train_r2",
            "mean_test_neg_root_mean_squared_error",
            "param_n_estimators",
            "rank_test_r2"
        ]
    ].set_index("rank_test_r2").sort_index()

    gridsearch_results_rf.columns = ["test r2", "train r2", "Negative RMSE", "n_estimators"]
    gridsearch_results_rf.to_csv(output_dir + '/tables/grid_search_rf.csv', index = False)
    print("saved grid search results for random forest to: " + output_dir)

    # define ideal parameters for models
    ideal_alpha = gridsearch_results_ridge["alpha"][1]
    ideal_n_estimators = gridsearch_results_rf["n_estimators"][2]                          # we use [2] because we want to select the second best value to save time
    
    # fit the ideal models to training data and output pickled models
    print("Fitting and pickling optimized models.. ")
    ideal_model_ridge = Ridge(alpha=ideal_alpha)
    ideal_model_ridge.fit(X_transformed_train, y_train)
    ideal_model_rf = RandomForestRegressor(n_estimators=ideal_n_estimators)
    ideal_model_rf.fit(X_transformed_train, y_train)

    pickle.dump(ideal_model_ridge, open("results/models/ridge_model.pickle", 'wb'))
    pickle.dump(ideal_model_rf, open("results/models/rf_model.pickle", 'wb'))
    print("Fitted models saved to: " + output_dir)

if __name__ == "__main__":
    main(opt)