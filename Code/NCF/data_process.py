import re
import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn import metrics, preprocessing




regex = "[\(\[].*?[\)\]]"

def findnum(x):
    return x.split("(")[-1].replace(")","")[0].strip()

def findtime(date):
    return datetime.fromtimestamp(date)

def check_time(date):
    return 1 if 6<int(date.strftime("%H"))<20 else 0

def check_weekdays(date):
    return 1 if date.weekday() in [5,6] else 0

def check_release(data):
    return 1 if data < 2000 else 0

def find_genre(ls):
    temp = list(set([x for inner in ls for x in inner]))
    temp.remove('(no genres listed)')
    return temp

def movie_genre(col,genre):
    return 1 if col in genre else 0

def  make_dataframe(df):
    transform = (0.5,1)
    return pd.DataFrame(preprocessing.MinMaxScaler(feature_range=transform).fit_transform(df.values), 
                         columns=df.columns,index=df.index)



def clean_data(df_movies, df_rating):

    dump = df_movies.iloc[:,2].isna()
    dum_ls=["movieId","product"]
    method = "left"
    retain1 = ["product","name","release","genres"]
    retain2 = ["user","product","daytime","weekend","y"]
    retain3 = ["user","product","y"]
    df_movies = df_movies[~dump]
    df_movies["product"] = range(0,len(df_movies))
    df_movies["name"] = df_movies.iloc[:,1]
    df_movies["release"] = df_movies.iloc[:,1].apply(lambda x: int(findnum(x)) if "(" in x else np.nan)
    df_rating["user"] = df_rating.iloc[:,0].apply(lambda userId: userId-1)
    df_rating["timestamp"] = df_rating.iloc[:,3].apply(lambda timestamp: findtime(timestamp))
    df_rating["weekend"] = df_rating.iloc[:,3].apply(lambda timestamp: check_weekdays(timestamp))
    df_movies["release"] = df_movies["release"].fillna(9999)
    df_rating["daytime"] = df_rating.iloc[:,3].apply(lambda timestamp: check_time(timestamp))
    df_movies["release"] = df_movies["release"].apply(lambda x: check_release(x))
    df_movies["name"] = df_movies["name"].apply(lambda x: re.sub(regex, "", x).strip())
    df_rating = df_rating.merge(df_movies[dum_ls], how=method)
    df_rating = df_rating.rename(columns={"rating":"y"})
    df_movies = df_movies[retain1].set_index("product")
    df_rating = df_rating[retain2]
    df_context = df_rating[retain2[:-1]]
    df_rating = df_rating[retain3]
    rating_dum = df_rating.copy()
    

    tags = list()
    unique_genre = set(df_movies["genres"])
    for genre in unique_genre:
        tags.append(genre.split("|"))
        
    columns = find_genre(tags)

    for col in columns:
        df_movies[col] = df_movies["genres"].apply(lambda x: movie_genre(col,x))


    df_rating = rating_dum.pivot_table(index="user", columns="product", values="y")
    missing_cols = list(set(df_movies.index) - set(df_rating.columns))

    for col in missing_cols:
        df_rating[col] = np.nan
    df_rating = df_rating[sorted(df_rating.columns)]


    df_rating =make_dataframe(df_rating)


    
    return df_movies, df_rating