FROM jupyter/r-notebook

RUN conda install --yes docopt=0.6.*  \
                    pandas=1.3.*  \
                    scikit-learn=1.0.* \
                    altair=4.1.* \
                    altair_data_server=0.4.* \
                    altair_saver=0.5.* 
