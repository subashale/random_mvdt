import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def DDScatterDFSns(column1, column2, df):
    if df.shape[1] <= 3:
        # assume last column is dependable variable
        sns.lmplot(x=[key for key in df.columns[[column1]]][0], y=[key for key in df.columns[[column2]]][0], 
                   data=df, fit_reg=False, hue=[key for key in df.columns[[-1]]][0], legend=False)

        # Move the legend to an empty part of the plot
        plt.legend(loc='lower right')
        plt.show()
    elif df.shape[1] > 3 and df.shape[1] <= 10:
        # selection only features 
#         fet = df.loc[:, df.columns != df.keys()[-1]]
        g = sns.pairplot(df, hue=df.keys()[-1])


def read_data(dataLocation):
    """
   Reading data and returning in proper format
   :param dataLocation: location of data
   :return: set of features, labels and combine
   """
    df = pd.read_csv(dataLocation)
    DDScatterDFSns(0, 1, df)
#     sns.pairplot(df.loc[:,df.dtypes == 'float64'])
    X = np.array(df[df.columns[:-1]].values.tolist())
    y = np.array(df[df.columns[-1]].values.tolist())
    return [X, y], df