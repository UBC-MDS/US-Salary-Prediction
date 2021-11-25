# author: Cuthbert Chow
# date: 2021-11-25

"""This script splits splits the input data file .
Usage: DataProcessing.py --source_data=<URL> --output_dir=<filepath>

Options:
--source_data=<filepath>     filepath to retrieve raw data from (this is a required option)
--output_dir=<filepath>      Local filepath with filename (this is a required option)
""" 

from docopt import docopt
import pandas as pd
from sklearn.model_selection import train_test_split

opt = docopt(__doc__)

def main(opt):
    source = opt['--source_data']
    filepath = opt['--output_dir']
    df = pd.read_csv(source)

    df_dropcurrency = df[df["currency"] == "USD"].drop(
        columns=["currency", "currency_other"]
    )
    
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
    ].drop(columns=['country', 'race', 'gender', 'timestamp'])

    train_df, test_df = train_test_split(df_clean, test_size=0.3, random_state=123)
    train_df.to_csv(filepath + '/train_df.csv', index = False)
    test_df.to_csv(filepath + '/test_df.csv', index = False)

    print("Processed Data Saved To: " + filepath)

if __name__ == "__main__":
    main(opt)
