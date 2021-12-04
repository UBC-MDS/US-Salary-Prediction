# salary prediction data pipe
# author: Cuthbert Chow
# date: 2021-12-03

all: doc/final_report.md

# download data
data/raw_data.csv: src/download_data.py
	python src/download_data.py --URL=https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-05-18/survey.csv --filepath=data/raw_data.csv

# pre-process data 
data/processed/train_df.csv data/processed/test_df.csv: data/raw_data.csv src/data_processing.py
	python src/data_processing.py --source_data=data/raw_data.csv --output_dir=data/processed

# Exploratory Data Analysis
results/tables/eda_summary_table.csv results/figures/eda_category_distribution.png results/figures/eda_target_distribution.png: data/processed/train_df.csv src/generate_eda.py
	python src/generate_eda.py --filepath=data/processed/train_df.csv --outfigure=results/figures/eda_target_distribution.png --outfigure2=results/figures/eda_category_distribution.png --outcsv=results/tables/eda_summary_table.csv

# Model Tuning and Fitting 

data/processed/x_test_sparse.npz data/processed/y_test.csv data/processed/feature_names.csv results/tables/grid_search_ridge.csv results/models/rf_model.pickleresults/models/ridge_model.pickle: src/fit_models.py data/processed/train_df.csv data/processed/test_df.csv
	python src/fit_models.py --source_data=data/processed --output_dir=results

results/tables/test_scores.csv results/tables/negative_coefficients_ridge.csv results/tables/positive_coefficients_ridge.csv results/tables/coefficient_comparison.csv results/figures/predicted_vs_actual_chart.png: data/processed/x_test_sparse.npz data/processed/y_test.csv data/processed/feature_names.csv results/models/rf_model.pickle results/models/ridge_model.pickle src/compare_models.py
	python src/compare_models.py --source_data=data/processed --output_dir=results

# Creating Final Report
doc/final_report.md: doc/final_report.Rmd results/figures/eda_category_distribution.png results/figures/eda_target_distribution.png results/tables/negative_coefficients_ridge.csv results/tables/positive_coefficients_ridge.csv results/tables/test_scores.csv results/tables/eda_summary_table.csv  results/tables/grid_search_ridge.csv results/tables/grid_search_ridge.csv results/tables/coefficient_comparison.csv results/figures/predicted_vs_actual_chart.png
	Rscript -e "rmarkdown::render('doc/final_report.Rmd', output_format = 'all')"

# Removal of artifact files (figures, data, models, outpur reports)
clean:
	rm -rf data/raw_data.csv data/processed/train_df.csv data/processed/test_df.csv data/processed/x_test_sparse.npx data/processed/y_test.csv
	rm -rf results/tables/grid_search_ridge.csv results/tables/grid_search_rf.csv results/tables/negative_coefficients_ridge.csv results/tables/positive_coefficients_ridge.csv results/tables/test_scores.csv results/tables/eda_summary_table.csv
	rm -rf results/tables/coefficient_comparison.csv
	rm -rf data/processed/feature_names.csv data/processed/x_test_sparse.npz
	rm -rf results/models/rf_model.pickle results/models/ridge_model.pickle
	rm -rf results/figures/eda_category_distribution.png results/figures/eda_target_distribution.png results/figures/predicted_vs_actual_chart.png
	rm -rf doc/final_report.pdf doc/final_report.md doc/final_report.html