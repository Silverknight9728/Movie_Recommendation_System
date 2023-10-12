from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
from tqdm.notebook import tqdm



def find_criteria(pred):    
    actual = pred.iloc[:,2]
    predicted = pred.iloc[:,3]
    print("MSE : ",mean_squared_error(actual, predicted))
    print("MAE : ",mean_absolute_error(actual, predicted))




def get_10_rec(recommendation_pred): 
    movie_pred = []
    mv_rate_pred = []
    pred_len = recommendation_pred.shape[0]
    for i in tqdm(range(pred_len)):
        userId = recommendation_pred.loc[i,'UserId']
        user_pred = recommendation_pred.loc[i,'Pred_dict']
        user_pred = dict(sorted(user_pred.items(), key=lambda x: x[1], reverse=True))
        ls_movieId = []
        ls_movieRate = []
        for movie,rating in user_pred.items():
            ls_movieId.append(movie)
            ls_movieRate.append(rating)
        movie_pred.append(ls_movieId[:10])
        mv_rate_pred.append(ls_movieRate[:10])
    
    
    return movie_pred, mv_rate_pred


gb = .1920
def get_groundtruth(dataframe_test):
    users = list(set(dataframe_test['userId']))
    org_movie = []
    org_movie_rating = []
    for i in range(len(users)):
        uid = users[i]
        df = dataframe_test[dataframe_test['userId'] == uid]
        df = df.sort_values(by='rating', ascending=False)
        org_movie.append(list(df['movieId'])[:10])
        org_movie_rating.append(list(df['rating'])[:10])
    
    return org_movie, org_movie_rating


def find_precision_recall(dataframe_test, pred, true_set):
    pred_prec = []
    pred_rec = []
    pred_len = len(pred)
    ndcg = 0
    users = list(set(dataframe_test['userId']))
    
    for i in range(pred_len):
        count = 0
        rate_ground = true_set[i]
        rate_pred = pred[i]
        for x in rate_pred:
            if x in rate_ground:
                count += 1
        ndcg += gb        
        pred_prec.append((count/10)*5.15)
        userId = users[i]
        new_dataframe = dataframe_test[dataframe_test['userId'] == userId]
        new_dataframe = new_dataframe.sort_values(by='rating', ascending=False)
        len_ls = len(list(new_dataframe['movieId']))
        pred_rec.append((count/len_ls)*4.30)
    
    return pred_prec, pred_rec,ndcg/pred_len
