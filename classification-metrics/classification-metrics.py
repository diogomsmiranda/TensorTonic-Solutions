import numpy as np

def classification_metrics(y_true: list[int], y_pred: list[int], average: str = "micro", pos_label: int = 1) -> dict:
    """
    Returns a dictionary containing accuracy, precision, recall, and f1 rounded to six decimals.
    """
    n = len(y_pred)
    classes = list(set(y_true) | set(y_pred))
    precisions = []
    recalls = []
    f1s = []
    tps = []
    fps = []
    fns = []
    tns = []
    counts = []
    correct = 0.0
    for c in classes:
        tp = 0.0
        fp = 0.0
        fn = 0.0
        tn = 0.0
        recall = 0.0
        precision = 0.0
        count = 0
        for y_i in range(n):
            if y_true[y_i] == c:
                count += 1
            if y_pred[y_i] == c and y_true[y_i] == c:
                tp += 1
                correct += 1
            elif y_pred[y_i] == c and y_true[y_i] != c:
                fp += 1
            elif y_pred[y_i] != c and y_true[y_i] == c :
                fn += 1
            elif y_pred[y_i] != c and y_true[y_i] != c:
                tn += 1
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        tns.append(tn)
        counts.append(count)
        precision = tp / max(1.0,(tp + fp))
        recall = tp / max(1.0,(tp + fn))
        precisions.append(precision)
        recalls.append(recall)
        f1 = (2 * precision * recall) / max(1.0,(precision + recall))
        f1s.append(f1)
        if average == "binary" and pos_label == c:
            return {"accuracy": round(sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / n,6), "precision":round(precision,6), "recall":round(recall,6), "f1": round(f1,6)}

    match average:
        case "micro":
            tp = sum(tps)
            fp = sum(fps)
            fn = sum(fns)
            tn = sum(tns)
            precision = tp / max(1.0,(tp + fp))
            recall = tp / max(1.0,(tp + fn))
            return {"accuracy": round(correct / n, 6), "precision":round(tp / max(1.0,tp+fp),6), "recall":round((tp / max(1.0,(tp+fn))),6), "f1": round((2 * precision * recall / max(1.0,(precision + recall))),6)}
        case "macro":
            return {"accuracy": round(correct / n, 6), "precision":round(np.mean(precisions),6), "recall":round(np.mean(recalls),6), "f1": round(np.mean(f1s),6)}
        case "weighted":
            return {"accuracy": round(correct / n, 6), "precision":round(np.average(precisions, weights=counts),6), "recall":round(np.average(recalls, weights=counts),6), "f1": round(np.average(f1s, weights=counts),6)}

    return 

        
        