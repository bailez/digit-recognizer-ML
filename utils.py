import pandas as pd
import os

def get_data():
    try:
        os.chdir(r'./data/')
    except FileNotFoundError:
        pass
    # %% Read data
    x = os.listdir()
    rd = pd.read_csv
    test ,train =  rd(x[0]), rd(x[1])
    return train.values, test.values
