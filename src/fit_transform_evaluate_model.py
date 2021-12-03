# author: Andy Yang
# date: 2021-11-27

"""This script transforms the cleaned data and fits a Ridge model to the preprocessed train data and evaluates it on the test data. 
Results are returned as tables a figure.

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
    PolynomialFeatures,
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
    
    # pre-process train and test data
    # remove outliers and rare data from train dataset
    train_df = train_df[(train_df["annual_salary"] < 1000000) & (train_df["annual_salary"] > 10000)]
    train_df["state"] = train_df["state"].mask(train_df["state"].map(train_df["state"].value_counts()) < 5, "")
    train_df["city"] = train_df["city"].mask(train_df["city"].map(train_df["city"].value_counts()) < 5, "")

    # replace nan values with blanks in industry and job_title features
    train_df.fillna({"industry":"", "job_title":""}, inplace=True)
    test_df.fillna({"industry":"", "job_title":""}, inplace=True)

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
    passthrough_features = []

    # create lists for ordinal encoding
    age_order = ["missing_value", "under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65 or over"]
    exp_order = ["missing_value", "1 year or less", "2 - 4 years", "5-7 years", "8 - 10 years",
                "11 - 20 years", "21 - 30 years", "31 - 40 years", "41 years or more"]
    edu_order = ["missing_value", "High School", "Some college", "College degree",
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
                SimpleImputer(strategy="constant"),
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
    X_transformed_train = preprocessor.fit_transform(X_train)
    X_transformed_test = preprocessor.transform(X_test)

    #extract feature names from pre-processor
    feats = list(preprocessor.transformers_[0][1][1].get_feature_names_out(categorical_features)) +\
        numeric_features + ordinal_features +\
        list(preprocessor.transformers_[3][1].get_feature_names_out()) +\
        list(preprocessor.transformers_[4][1].get_feature_names_out())

    # define scoring metrics
    score_types_reg = {
        "neg_root_mean_squared_error": "neg_root_mean_squared_error",
        "r2": "r2"
    }

    # conduct hyperparameter optimization on Ridge and save the results
    print("conducting hyperparameter optimization..")
    param_grid_cgamma = {"alpha": np.logspace(-5, 5, 20)}

    random_search = RandomizedSearchCV(
        Ridge(),
        param_distributions=param_grid_cgamma,
        verbose=1,
        n_jobs=-1,
        n_iter=20,
        cv=5,
        random_state=123,
        scoring=score_types_reg,
        refit=False)

    random_search.fit(X_transformed_train, y_train)

    gridsearch_results = pd.DataFrame(random_search.cv_results_)[
        [
            "mean_test_r2",
            "mean_test_neg_root_mean_squared_error",
            "param_alpha",
            "rank_test_r2"
        ]
    ].set_index("rank_test_r2").sort_index()

    gridsearch_results.columns = ["r2", "Negative RMSE", "alpha"]
    gridsearch_results.index.names = ["r2 score rank"]
    gridsearch_results.to_csv(output_dir + '/tables/grid_search.csv', index = False)
    print("saved grid search results to: " + output_dir)

    # define ideal_alpha
    ideal_alpha = gridsearch_results["alpha"][1]
    
    # fit the ideal model to training data and generate salary predictions for the test set
    ideal_model = Ridge(alpha=ideal_alpha)
    ideal_model.fit(X_transformed_train, y_train)
    y_pred = ideal_model.predict(X_transformed_test)

    # create scores dataframe and save it to output directory
    r2 = round(sk.r2_score(y_test, y_pred), 2)
    rmse = round(math.sqrt(sk.mean_squared_error(y_test, y_pred)), 2)

    scores = {
        "Metric": ["R2", "RMSE"],
        "Scores": [r2, rmse]
    }
    test_scores = pd.DataFrame(scores)
    test_scores.to_csv(output_dir + '/tables/test_scores.csv', index = False)
    print("saved model test results to: " + output_dir)

    # Plot the predicted values against true values, then save the graph in the output directory
    y_data = {
        "y_pred": y_pred,
        "y_actual": y_test
    }
    salary_data = pd.DataFrame(y_data)

    point = alt.Chart(salary_data, title='Ridge regression effectiveness in predicting salary values').mark_circle(opacity = 0.6).encode(
        alt.X("y_pred", title="Predicted Salary"),
        alt.Y('y_actual', title="Actual Salary")
    )

    line = pd.DataFrame({
        'x': [0, 500000],
        'y':  [0, 500000],
    })

    line_plot = alt.Chart(line).mark_line(color= 'red').encode(
        x= 'x',
        y= 'y',
    )

    chart = point + line_plot
    chart.save(output_dir + "/figures/predicted_vs_actual_chart.png")
    print("saved model evaluation chart to: " + output_dir)

    # create model coefficient dataframes and save them to the output directory
    neg_coefficients_df = pd.DataFrame(data=ideal_model.coef_, index=feats, columns=["coefficient"]).sort_values("coefficient")[:10].reset_index()
    neg_coefficients_df.columns = ["Feature", "Coefficient"]
    pos_coefficients_df =pd.DataFrame(data=ideal_model.coef_, index=feats, columns=["coefficient"]).sort_values("coefficient", ascending = False)[:10].reset_index()
    pos_coefficients_df.columns = ["Feature", "Coefficient"]
    
    neg_coefficients_df.to_csv(output_dir + '/tables/negative_coefficients.csv', index = False)
    pos_coefficients_df.to_csv(output_dir + '/tables/positive_coefficients.csv', index = False)
    print("saved coefficient tables to: " + output_dir)


if __name__ == "__main__":
    main(opt)