import numpy as np
import pandas as pd


def convert(trends_df):
    # trends_df.describe()
    print(trends_df)

if __name__ == '__main__':
    df = pd.read_csv('Trends.csv')
    convert(df)