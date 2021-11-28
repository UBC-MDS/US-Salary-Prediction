# US Salary Prediction

Authors: Cuthbert Chow, Rong Li, Andy Yang\
Data analysis project to predict salaries for DSCI 522 (Data Science Workflows); a course in the Master of Data Science program at the University of British Columbia.

## About

One of the most important things in the job search is about the salaries, specifically, does this job's salary meet our expectations? However, it is not that easy to set proper expectations. Setting an expectation too high or too low will both be harmful to our job search.

Here, this project is to help you to answer this question: What we can expect a person's salary to be in the US?

According to Martín et al. (2018), a linear regression model with an R2 score is a good combination for predicting salaries, so we will use that to do the prediction. In the process, we wish to understand which factors provide the most predictive power when trying to predict a person's salary.

The dataset we are analyzing comes from a salary survey from the "Ask a Manager" blog by Alison Green. This dataset contains survey data gathered from "Ask a Manager" readers working in a variety of industries, and can be found [here](https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-05-18/survey.csv).

## Report

We first did some exploratory data analysis (a complete EDA file can be found [here](https://github.com/UBC-MDS/US-Salary-Prediction/blob/main/results/EDA.ipynb)). We noticed that most of the columns are text columns and there are lots of unique values. Therefore, we dropped some columns in our analysis and did column transformation, and we use Ridge to create a model with a 0.38 R2 score. Our final report can be found [here](https://github.com/UBC-MDS/US-Salary-Prediction/blob/main/doc/final_report.md).

## Usage

To replicate the analysis conducted in this repository, ensure that all listed dependencies are installed, and the run the commands below from the command line, whilst located at the root directory of this project. 

### Dependencies  

- python version 3.9.5 and Python packages:  
  - docopt=0.6.2  
  - pandas=1.3.3  
  - scikit-learn=1.0.1  
  - altair=4.1.0  
  - altair_data_server=0.4.1  
  - altair_saver=0.5.0  
- R version 4.1.1 and R packages:  
  - knitr=1.33
  
```
# download data
python src/DownloadData.py --URL=https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-05-18/survey.csv --filepath=data/raw_data.csv

# pre-process data 
python src/DataProcessing.py --source_data=data/raw_data.csv --output_dir=data/processed

# Exploratory Data Analysis
python src/generate_eda.py --filepath=data/processed/train_df.csv --outfigure=results/figures/eda_target_distribution.png --outfigure2=results/figures/eda_category_distribution.png --outcsv=results/tables/eda_summary_table.csv

# Model Tuning and Fitting 
python src/fit_transform_evaluate_model.py --source_data=data/processed --output_dir=results

# Creating Final Report
Rscript -e "rmarkdown::render('doc/final_report.Rmd', output_format = 'all')"
```

## Reference

Green, Alison. 2021. "How Much Money Do You Make?" Ask A Manager. <https://www.askamanager.org/2021/04/how-much-money-do-you-make-4.html>. Martín, Ignacio, Andrea Mariello, Roberto Battiti, and José Alberto Hernández. 2018. "Salary Prediction in the IT Job Market with Few High-Dimensional Samples: A Spanish Case Study." International Journal of Computational Intelligence Systems 11: 1192--1209. <https://doi.org/https://doi.org/10.2991/ijcis.11.1.90>.
