from sklearn.metrics import ndcg_score
from math import log2

def rprecision(prediction, ground_truth):
    preds_set = set(prediction)
    trues_set = set(ground_truth)

    return len( preds_set.intersection(trues_set) ) / len(trues_set)

def ndcg(prediction, ground_truth):
    return dcg(prediction, ground_truth) / dcg(prediction, prediction)

def dcg(prediction, ground_truth):
    rel = lambda x : x in ground_truth

    _dcg = rel(prediction[0])

    for i in range( 1, len(prediction) ):
        _dcg += rel(prediction[i])/(log2(i+1))

    return _dcg


def nclicks(prediction, ground_truth):
    return 1