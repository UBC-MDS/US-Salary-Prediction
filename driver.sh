# download data
python src/DownloadData.py --URL=https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-05-18/survey.csv --filepath=data/raw_data.csv

# pre-process data 
python src/DataProcessing.py --source_data=data/raw_data.csv --output_dir=data/processed

# Model Tuning


# Exploratory Data Analysis





# create exploratory data analysis figure and write to file 
Rscript src/eda_wisc.r --train=data/processed/training.feather --out_dir=results

# tune model
Rscript src/fit_breast_cancer_predict_model.r --train=data/processed/training.feather --out_dir=results

# test model
Rscript src/breast_cancer_test_results.r --test=data/processed/test.feather --out_dir=results

# render final report
Rscript -e "rmarkdown::render('doc/breast_cancer_predict_report.Rmd', output_format = 'github_document')"
