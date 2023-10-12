import re
import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn import metrics, preprocessing




def data_split(df_rating):
    
    split = int(0.8*df_rating.shape[1])

    train_rating = df_rating.loc[:, :split-1]
    test_rating = df_rating.loc[:, split:]
    col_dict = {0:"y"}

    model_train = train_rating.stack(dropna=True).reset_index().rename(columns=col_dict)
    x=[model_train["user"], model_train["product"]]

    model_test = test_rating.stack(dropna=True).reset_index().rename(columns=col_dict)


    return x , model_train, model_test
