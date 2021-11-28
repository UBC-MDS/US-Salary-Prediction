# author: Rong Li
# date: 2021-11-25

"""This script using the training dataset and produce figures 
Usage: generate_eda.py --filepath=<filepath> --outfigure=<outfigure> --outfigure2=<outfigure2> --outcsv=<outcsv>

Options:
--filepath=<filepath>     Local filepath for training dataset with filename (this is a required option)
--outfigure=<outfigure>   Local filepath for saving figure with filename (this is a required option)
--outfigure2=<outfigure2> Local filepath for saving figure 2 with filename (this is a required option)
--outcsv=<outcsv>         Local filepath for saving csv files with filename (this is a required option)
""" 

from docopt import docopt
import pandas as pd
import altair as alt
alt.data_transformers.disable_max_rows()
opt = docopt(__doc__)

def main(opt):
    # Interpret dictionary
    filepath = opt["--filepath"]
    outfigure = opt["--outfigure"]
    outfigure2 = opt["--outfigure2"]
    outcsv = opt["--outcsv"]
    
    # Read training dataset
    train_df = pd.read_csv(filepath)

    # Save a figure for the target distribution
    chart = alt.Chart(train_df).mark_bar(clip=True).encode(
            alt.X("annual_salary", bin=alt.Bin(maxbins=200), scale=alt.Scale(domain=(0, 500000)), title = "Annual Salary"),
            y='count()')
    chart = chart + alt.Chart(train_df).mark_rule(color="red").encode(
        x=alt.X("median(annual_salary)"),
    )
    chart.save(outfigure)
    
    # Save a summary table for cateforical features as csv file
    train_X = train_df.drop(columns=["annual_salary"])
    result = {}
    result["Not Null Count"] = train_X.notnull().sum()
    result["Null Count"] = train_X.isnull().sum()
    result["Number of Unique Values"] = train_X.nunique()
    result["Some Unique Values"] = df_uniques(train_X)
    result["Types"] = train_X.dtypes
    result_df = pd.DataFrame(result)
    result_df.index.names = ['Features'] 
    result_df.to_csv(outcsv)

    # Save a figure for the categorical features
    col = result_df[result_df["Number of Unique Values"] < 20]
    collist = col.index.tolist()
    
    chart =alt.Chart(train_df).mark_bar(clip=True).encode(
            x=alt.X("median(annual_salary)",scale=alt.Scale(domain=(0, 200000)), title = "Median Annual Salary"),
            y=alt.Y(alt.repeat(), type='ordinal',sort='x'),
        ).properties(
            width=300,
            height=200
        ).repeat(
            collist,
            columns=2
        )

    chart.save(outfigure2)
    
def df_uniques(df):
    """Find unique values for each dataframe columns"""
    result = {}
    for x in df.columns:
        result[df[x].name] = df[x].unique().tolist()[0:5]
    return result

if __name__ == "__main__":
    main(opt)