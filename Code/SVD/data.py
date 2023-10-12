import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split


def create_data(d_r,d_m, d_l):
    rat_len = d_r.shape[0]
    for ind in tqdm(range(rat_len)):
        desc_movie = d_m[d_m['movieId'] == d_r.loc[ind,'movieId']]
        l = d_l[d_l['movieId'] == d_r.loc[ind,'movieId']]
        movie_gene = desc_movie.loc[:,'genres']
        d_r.iloc[ind,5] = movie_gene
        movie_tit = desc_movie.loc[:,'title']
        d_r.iloc[ind,4] = movie_tit
        movie_iId = l.loc[:,'imdbId']
        d_r.iloc[ind,6] = movie_iId
        movie_tId = l.loc[:,'tmdbId']
        d_r.iloc[ind,7] = movie_tId
    
    return d_r


def add_tags(d_r,d_t):
    Tags = dict()
    tag_len = d_t.shape[0]
    for i in tqdm(range(tag_len)):
        userId,movieId,tag = d_t.loc[i,'userId'],d_t.loc[i,'movieId'],d_t.loc[i,'tag']
        if userId in Tags:
            if movieId not in Tags[userId]:
                Tags[userId][movieId] = []    
        else:
            Tags[userId] = dict()
            Tags[userId][movieId] = []
        Tags[userId][movieId].append(tag)
        
    for id_user, d in Tags.items():
        for id_mv, ls_tg in d.items():
            condi1 = (id_user == d_r['userId'])
            condi2 = (id_mv == d_r['movieId'] )
            d_r.loc[condi1 & condi2 , 'Tags'] = str(ls_tg)

    
    return d_r


def make_dataset(d_tr, d_tt,d_r): 
    users = list(set(d_r.iloc[:,0]))
    user_len = len(users)
    for ind in tqdm(range(user_len)):
        data = d_r[users[ind] == d_r.iloc[:,0]]
        train,test = train_test_split(data, test_size=0.2)
        d_tt = pd.concat([d_tt, test])
        d_tr = pd.concat([d_tr, train])
    d_tr.reset_index(inplace=True, drop=True)
    d_tt.reset_index(inplace=True, drop=True)
    return d_tr, d_tt
