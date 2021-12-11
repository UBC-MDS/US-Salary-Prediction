# Authors: Cuthbert Chow, Rong Li, Andy Yang
# Date: 2021-12-11
# Create docker image required for running US-Salary-Prediction analysis

# Use the r-notebook base image from Jupyter
FROM jupyter/r-notebook

# Install Python dependencies through the conda distribution
RUN conda install --yes docopt=0.6.*  \
                    pandas=1.3.*  \
                    scikit-learn=1.0.* \
                    altair=4.1.* \
                    altair_data_server=0.4.* \
                    altair_saver=0.5.* 
