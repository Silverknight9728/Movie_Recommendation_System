import math
def eval_matrix(model_test):
    ls_test_user = model_test["user"].unique()
    # single case
    
    precision = []
    recall = []
    ndcg = []
    f1 = []
    for i in ls_test_user:
        #print("--- user", i, "---")

        top = 10
        y_test = model_test[model_test["user"]==i].sort_values("y", ascending=False)["product"].values[:top]
        #print("y_test:", y_test)

        predicted = model_test[model_test["user"]==i].sort_values("yhat", ascending=False)["product"].values[:top]
        #print("predicted:", predicted)

        dcg= 0
        idcg=0
        count = 0

        temp_ytest = list(y_test)
        temp_predicted = list(predicted)
        if(len(temp_predicted)== 10):

            for ind in temp_ytest:
                #print(math.log2(1 + temp_ytest.index(ind)))
                idcg += (1 / math.log2(1 + (1+temp_ytest.index(ind))))
                if ind in temp_predicted:
                    dcg += (1 / math.log2(1+(1+temp_predicted.index(ind))))             

            ndcg.append(dcg / idcg)


        true_positive = len(list(set(y_test) & set(predicted)))
        prec = true_positive / top
        rec = true_positive / (top*2.5)
        precision.append(prec)
        recall.append(rec)

        if(prec != 0 or rec != 0):
            f1.append(2* (prec * rec) / (prec+rec))
        #print("true positive:", true_positive, "("+str(round(true_positive/top*100,1))+"%)")
        #print("accuracy:", str(round(metrics.accuracy_score(y_test,predicted)*100,1))+"%")
        #print("mrr:", round(mean_reciprocal_rank(y_test, predicted),2))
        #print("ndcg : ", (dcg / idcg))
        
        
    return precision, recall,ndcg,f1