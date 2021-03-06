# author: Cuthbert Chow
# date: 2021-11-25

"""This script splits splits the input data file .
Usage: data_processing.py --source_data=<URL> --output_dir=<filepath>

Options:
--source_data=<filepath>     filepath to retrieve raw data from (this is a required option)
--output_dir=<filepath>      Local filepath with filename (this is a required option)
""" 

#Import Dependencies
from docopt import docopt
import pandas as pd
from sklearn.model_selection import train_test_split

opt = docopt(__doc__)

def main(opt):
    # Read raw CSV files
    source = opt['--source_data']
    filepath = opt['--output_dir']
    df = pd.read_csv(source)

    # Remove Currency Column from Data
    df_dropcurrency = df[df["currency"] == "USD"].drop(
        columns=["currency", "currency_other"]
    )
    
    # Keep only country values equivalent to 'US', and drop unused columns
    df_clean = df_dropcurrency[
        df_dropcurrency["country"]
        .str.lower()
        .isin(
            [
                "us",
                "usa",
                "u.s.",
                "u.s.a",
                "united states",
                "america",
                "united states of america",
            ]
        )
    ].drop(columns=['country', 'race', 'gender', 'timestamp',"additional_context_on_job_title", "additional_context_on_income"])

    # Create train and test datasets, export as CSV
    train_df, test_df = train_test_split(df_clean, test_size=0.3, random_state=123)
    train_df.to_csv(filepath + '/train_df.csv', index = False)
    test_df.to_csv(filepath + '/test_df.csv', index = False)

    print("Processed Data Saved To: " + filepath)

if __name__ == "__main__":
    main(opt)


