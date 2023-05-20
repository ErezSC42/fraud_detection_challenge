from sklearn.metrics import precision_score, recall_score, f1_score


def count_correct_anomalies(pred_, true_):
    counter = 0
    for x, y in zip(pred_, true_):
        if y == -1 and x == -1:
            counter +=1
    return counter / 10


def detection_metrics(pred_, true_):
    return {
        "detection_score": count_correct_anomalies(pred_, true_),
        "precision_score": float(f"{precision_score(pred_, true_):.3f}"),
        "recall_score": float(f"{recall_score(pred_, true_):.3f}"),
        "f1_score": float(f"{f1_score(pred_, true_):.3f}")
    }