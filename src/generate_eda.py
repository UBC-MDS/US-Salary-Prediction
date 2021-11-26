# author: Rong Li
# date: 2021-11-25

"""This script using the training dataset and produce figure 
Usage: generate_eda.py --filepath=<filepath> --outfigure=<outpath>

Options:
--filepath=<filepath>     Local filepath for training dataset with filename (this is a required option)
--outfigure=<outfigure>   Local filepath for figures with filename (this is a required option)
""" 

from docopt import docopt
import pandas as pd
import altair as alt
alt.data_transformers.disable_max_rows()
opt = docopt(__doc__)

def main(opt):
    
    filepath = opt["--filepath"]
    outfigure = opt["--outfigure"]
    
    train_df = pd.read_csv(filepath)
    
    chart = alt.Chart(train_df).mark_bar().encode(
        alt.X("annual_salary", bin=alt.Bin(maxbins=200), title = "Annual Salary"),
        y='count()')

    chart.save(outfigure)

if __name__ == "__main__":
    main(opt)