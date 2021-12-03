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
results/grid_search.csv results/negative_coefficients.csv results/positive_coefficients.csv results/test_scores.csv: src/fit_transform_evaluate_model.py data/processed/train_df.csv data/processed/test_df.csv
	python src/fit_transform_evaluate_model.py --source_data=data/processed --output_dir=results

# Creating Final Report
doc/final_report.md: doc/final_report.Rmd results/figures/eda_category_distribution.png results/figures/eda_target_distribution.png results/grid_search.csv results/negative_coefficients.csv results/positive_coefficients.csv results/test_scores.csv results/tables/eda_summary_table.csv 
	Rscript -e "rmarkdown::render('doc/final_report.Rmd', output_format = 'all')"

clean:
	rm -rf data/raw_data.csv data/processed/train_df.csv data/processed/test_df.csv
	rm -rf results/grid_search.csv results/negative_coefficients.csv results/positive_coefficients.csv results/test_scores.csv results/tables/eda_summary_table.csv
	rm -rf results/figures/eda_category_distribution.png results/figures/eda_target_distribution.png
	rm -rf doc/final_report.pdf doc/final_report.md

