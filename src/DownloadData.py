# author: Andy Yang
# date: 2021-11-18

"""This script downloads data from a given filepath.
Usage: demo.py --URL=<URL> --filepath=<filepath>

Options:
--URL=<URL>             URL to download csv data from (this is a required option)
--filepath=<filepath>   Local filepath with filename (this is a required option)
""" 

from docopt import docopt
import pandas as pd
opt = docopt(__doc__)

def main(opt):
    url = opt["--URL"]
    filepath = opt["--filepath"]
    dataset = pd.read_csv(url)
    dataset.to_csv(filepath)
    print("Data successfully downloaded to: " + filepath)

if __name__ == "__main__":
    main(opt)
