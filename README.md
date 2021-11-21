# US Salary Prediction
DSCI_522_group29

## Project Proposal 
Authors: Cuthbert Chow, Rong Li, Andy Yang  
Data analysis project to predict salaries for DSCI 522 (Data Science Workflows); a course in the Master of Data Science program at the University of British Columbia.

### Data
The dataset we are analysing comes from a salary survey from the "Ask a Manager" blog by Alison Green. This dataset contains survey data gathered from "Ask a Manager" readers working in a variety of industries, and can be found [here](https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-05-18/survey.csv).

### Research question and method of analysis
The main predictive question we wish to answer is what we can expect a person's salary to be in the US, given a certain professional history (such as years of experience, industry, or age). We will use a linear regression model to do the prediction. In the process, we wish to understand which factors provide the most predictive power when trying to predict a person's salary. 

### Exploratory data analysis
We will generate EDA tables which list all data features, along with their data types and distributions (i.e. mean, quartiles, etc.) if they are numeric. One EDA figure we will create is a scatterplot matrix of continuous variables against the output variable (salary), and histograms of all the continuous variables which will further help us quantify their distributions.  
Our EDA file can be found [here](https://github.com/UBC-MDS/US-Salary-Prediction/blob/main/eda/EDA.ipynb). As shown in the summary table, most of our columns are text columns and there are lots of null values in the additional information (additional_context_on_job_title, additional_context_on_income, etc). Although there seems to be a positive correlation between the numeric column "other monetary compensation" and target (as shown in figure 3), we further looked at this column and noticed that there are too many Nulls and we probably won't have this data when the very thing we are trying to predict is salaries. So we may drop this column when making the prediction model.

### Results
We may share the results of our analysis in a literate coding document (specifically, a jupyter notebook), which would interleave the resulting figures and tables as well as other code contained in scripts with narrative explanations. 


### Reference:
Green, Alison. “How Much Money Do You Make?” Ask a Manager, 27 Apr. 2021, https://www.askamanager.org/2021/04/how-much-money-do-you-make-4.html. 
