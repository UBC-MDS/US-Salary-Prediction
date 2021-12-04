# author: Andy Yang
# date: 2021-11-27

"""This script imports preprocessed test data and fitted Ridge and RandomForestRegressor models. 
It then evaluates them on the test set and outputs evaluation metrics to the output directory.

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
    PolynomialFeatures,
    StandardScaler,
)

from sklearn.pipeline import Pipeline, make_pipeline

opt = docopt(__doc__)

def main(opt):
    input_dir = opt['--source_data']
    output_dir = opt['--output_dir']

    # load test data from input directory
    print("loading test data..")
    X_transformed_test_sparse = scipy.sparse.load_npz(input_dir + '/x_test_sparse.npz')
    X_transformed_test = pd.DataFrame.sparse.from_spmatrix(X_transformed_test_sparse)
    y_test = pd.read_csv(input_dir + '/y_test.csv')
    feats = pd.read_csv(input_dir + '/feature_names.csv').iloc[:,0]

    # load models from pickle files
    print("loading fitted models..")
    ridge_model = pickle.load(open("results/models/ridge_model.pickle", 'rb'))
    rf_model = pickle.load(open("results/models/rf_model.pickle", 'rb'))

    # generate predictions on test set
    y_pred_ridge = ridge_model.predict(X_transformed_test)
    y_pred_rf = rf_model.predict(X_transformed_test)

    print("creating tables and figures..")
    # create scores dataframe and save it to output directory
    r2_ridge = round(sk.r2_score(y_test, y_pred_ridge), 2)
    r2_rf = round(sk.r2_score(y_test, y_pred_rf), 2)
    rmse = round(math.sqrt(sk.mean_squared_error(y_test, y_pred_ridge)), 2)
    rmse_rf = round(math.sqrt(sk.mean_squared_error(y_test, y_pred_rf)), 2)

    scores = {
        "Metric": ["R2", "RMSE"],
        "Ridge Scores": [r2_ridge, rmse],
        "Random Forest Scores": [r2_rf, rmse_rf]
    }
    
    test_scores = pd.DataFrame(scores)
    test_scores.to_csv(output_dir + '/tables/test_scores.csv', index = False)
    print("saved model test results to: " + output_dir)

    # Plot the predicted values against true values, then save the graph in the output directory
    y_data = {
        "Ridge precitions": y_pred_ridge,
        "Random Forest predictions": y_pred_rf,
        "y_actual": y_test.iloc[:, 0]
    }
    salary_data = pd.DataFrame(y_data)
    salary_data = salary_data.melt(value_vars = ["Ridge precitions", "Random Forest predictions"], id_vars = "y_actual")


    point = alt.Chart(salary_data, title='Ridge and Random Forest regression effectiveness in predicting salary values').mark_circle(opacity = 0.3).encode(
        alt.X("value", title="Predicted Salary"),
        alt.Y('y_actual', title="Actual Salary"),
        color = "variable"
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
    neg_coefficients_df = pd.DataFrame(data=ridge_model.coef_, index=feats, columns=["coefficient"]).sort_values("coefficient")[:10].reset_index()
    neg_coefficients_df.columns = ["Feature", "Coefficient"]
    pos_coefficients_df =pd.DataFrame(data=ridge_model.coef_, index=feats, columns=["coefficient"]).sort_values("coefficient", ascending = False)[:10].reset_index()
    pos_coefficients_df.columns = ["Feature", "Coefficient"]
    
    ridge_feats = pd.DataFrame(data=ridge_model.coef_, index=feats, columns=["coefficient"]).sort_values(by = "coefficient", ascending = False).reset_index()
    rf_feats = pd.DataFrame(data=rf_model.feature_importances_, index=feats, columns=["coefficient"]).sort_values(by = "coefficient", ascending = False).reset_index()

    rf_coef_df = pd.DataFrame(rf_feats)
    ridge_coef_df = pd.DataFrame(ridge_feats)
    combined_df = pd.merge(ridge_coef_df[:10], rf_coef_df[:10], left_index=True, right_index=True).reset_index().round(4)
    combined_df.columns = ["Significance Rank", "Ridge Feature", "Ridge Coefficient", "Random Forest Feature", "RandomForest Coefficient"]
    combined_df["Significance Rank"] = combined_df["Significance Rank"] + 1

    neg_coefficients_df.to_csv(output_dir + '/tables/negative_coefficients_ridge.csv', index = False)
    pos_coefficients_df.to_csv(output_dir + '/tables/positive_coefficients_ridge.csv', index = False)
    combined_df.to_csv(output_dir + '/tables/coefficient_comparison.csv', index = False)
    print("saved coefficient tables to: " + output_dir)


if __name__ == "__main__":
    main(opt)
