from scipy.sparse.linalg import svds
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from tqdm.notebook import tqdm

def SVD_rating_cross(t_d, eval_ctr, folds):
    svd = SVD()
    return cross_validate(svd, t_d, measures=eval_ctr, cv= folds, verbose = True)


def rate_pred(full_training_data, pred):
    svd = SVD()
    svd.fit(full_training_data)
    for entry in tqdm(range(pred.shape[0])):
        pred.loc[entry,"svd_prediction"] = round(svd.predict(pred.loc[entry,"userId"], pred.loc[entry,"movieId"])[3]*2)/2
    
    return pred