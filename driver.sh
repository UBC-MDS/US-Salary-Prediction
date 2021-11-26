# download data
python src/DownloadData.py --URL=https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-05-18/survey.csv --filepath=data/raw_data.csv

# pre-process data 
python src/DataProcessing.py --source_data=data/raw_data.csv --output_dir=data/processed

# Exploratory Data Analysis
python src/generate_eda.py --filepath=data/processed/train_df.csv --outfigure=results/figures/eda_target_distribution.png --outfigure2=results/figures/eda_category_distribution.png --outcsv=results/tables/eda_summary_table.csv

# Model Tuning and Fitting 


# Creating Final Report
Rscript -e "rmarkdown::render('doc/final_report.Rmd', output_format = 'html_document')"





# # create exploratory data analysis figure and write to file 
# Rscript src/eda_wisc.r --train=data/processed/training.feather --out_dir=results

# # tune model
# Rscript src/fit_breast_cancer_predict_model.r --train=data/processed/training.feather --out_dir=results

# # test model
# Rscript src/breast_cancer_test_results.r --test=data/processed/test.feather --out_dir=results

# # render final report
# Rscript -e "rmarkdown::render('doc/breast_cancer_predict_report.Rmd', output_format = 'github_document')"
