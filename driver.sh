# download data
python src/DownloadData.py --URL=https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-05-18/survey.csv --filepath=data/raw_data.csv

# pre-process data 
python src/DataProcessing.py --source_data=data/raw_data.csv --output_dir=data/processed

# Exploratory Data Analysis
python src/generate_eda.py --filepath=data/processed/train_df.csv --outfigure=results/figures/eda_target_distribution.png --outfigure2=results/figures/eda_category_distribution.png --outcsv=results/tables/eda_summary_table.csv

# Model Tuning and Fitting 
python src/fit_transform_evaluate_model.py --source_data=data/processed --output_dir=results

# Creating Final Report
Rscript -e "rmarkdown::render('doc/final_report.Rmd', output_format = 'html_document')"