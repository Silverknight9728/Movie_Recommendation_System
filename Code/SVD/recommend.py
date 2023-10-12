from collections import defaultdict

def new_movie_recommendation(user,movies,data_training, recommendation_pred,full_training_data,svd):
    to_movie = list(set(movies.iloc[:,0]))    # movieId
    rated = list(set(data_training[data_training.iloc[:,0] == user]['movieId']))
    not_rated = []
    for i in to_movie:
        if i not in rated:
            not_rated.append(i)
    
    user_dict = defaultdict(float)
    len_no_rated = len(not_rated)
    for i in range(len_no_rated):
        user_dict[not_rated[i]] = round(svd.predict(user, not_rated[i])[3]*2)*0.5
        
    recommendation_pred = recommendation_pred.append({'UserId':user, 'Pred_dict':user_dict}, ignore_index = True)
    
    return recommendation_pred