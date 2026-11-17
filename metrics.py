import torch
import numpy as np
import sklearn.metrics
from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events
# f1 / f1pa / f1pa%k / affiliation-based F1-score / composite f1  Event-based-F1

def get_metrics(target, pred_label, score):

    f1 = sklearn.metrics.f1_score(y_true=target, y_pred=pred_label)
    accuracy = sklearn.metrics.accuracy_score(y_true=target, y_pred=pred_label)
    precision = sklearn.metrics.precision_score(y_true=target, y_pred=pred_label)
    recall = sklearn.metrics.recall_score(y_true=target, y_pred=pred_label)
    auc_roc = sklearn.metrics.roc_auc_score(y_true=target, y_score=score)
    auc_pr = sklearn.metrics.average_precision_score(y_true=target, y_score=score)

    return f1, accuracy, precision, recall, auc_roc, auc_pr

## point adjustment f1
def get_metrics_PA(target, pred_label):
    anomaly_state = False
    for i in range(len(target)):
        if target[i] == 1 and pred_label[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if target[j] == 0:
                    break
                else:
                    if pred_label[j] == 0:
                        pred_label[j] = 1
            for j in range(i, len(target)):
                if target[j] == 0:
                    break
                else:
                    if pred_label[j] == 0:
                        pred_label[j] = 1
        elif target[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred_label[i] = 1

    f1 = sklearn.metrics.f1_score(y_true=target, y_pred=pred_label)
    accuracy = sklearn.metrics.accuracy_score(y_true=target, y_pred=pred_label)
    precision = sklearn.metrics.precision_score(y_true=target, y_pred=pred_label)
    recall = sklearn.metrics.recall_score(y_true=target, y_pred=pred_label)

    return f1, accuracy, precision, recall

## F1 PA%K
def get_metrics_PA_K(target, pred_label):
    f1_k = []
    acc_k = []
    prec_k = []
    recall_k = []
    for k in range(0,101):
        f1_pa, acc_pa, prec_pa, recall_pa = get_metrics_PA(target, pred_label)
        f1_k.append(f1_pa)
        acc_k.append(acc_pa)
        prec_k.append(prec_pa)
        recall_k.append(recall_pa)

    f1_pak_auc = np.trapz(f1_k)
    acc_ak_auc = np.trapz(acc_k)
    prec_pak_auc = np.trapz(prec_k)
    recall_pak_auc = np.trapz(recall_k)

    return f1_pak_auc/100, acc_ak_auc/100, prec_pak_auc/100, recall_pak_auc/100

## affiliation-based F1-score
def get_metrics_affiliation(target, pred_label):
    if len(np.where(pred_label == 1)[0]) == 0 or len(np.where(target == 1)[0]) == 0:
        print('all 0s no event to evaluate')
        return 0.0, 0.0, 0.0

    events_pred = convert_vector_to_events(pred_label)
    events_gt = convert_vector_to_events(target)
    Trange = (0, len(pred_label))
    aff_res = pr_from_events(events_pred, events_gt, Trange)
    P = aff_res['precision']
    R = aff_res['recall']
    f1 = 2 * P * R / (P + R)

    return f1, P, R


## composite f1  Event-based-F1 
def get_events(y_test, outlier=1, normal=0):
    events = dict()
    label_prev = normal
    event = 0  # corresponds to no event
    event_start = 0
    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
        else:
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label

    if label_prev == outlier:
        event_end = tim - 1
        events[event] = (event_start, event_end)
    return events

def get_metrics_fc1(target, pred_label):
    true_events = get_events(target)

    tp = np.sum([pred_label[start:end + 1].any() for start, end in true_events.values()])
    fn = len(true_events) - tp

    rec_e = tp/(tp + fn) # recall self.eps
    prec_t = sklearn.metrics.precision_score(target, pred_label) # precision
    EventF1PA1 = 2 * rec_e * prec_t / (rec_e + prec_t + 1e-15) # f1

    return EventF1PA1, prec_t, rec_e

def print_all_metrics(target, pred_label, score):

    print('----------F1PA----------')
    f1, accuracy, precision, recall = get_metrics_PA(target, pred_label)
    print('f1 ', f1, 'accuracy ', accuracy, 'precision ', precision, 'recall ', recall)

    print('----------F1PA%K----------')
    f1, accuracy, precision, recall = get_metrics_PA_K(target, pred_label)
    print('f1 ', f1, 'accuracy ', accuracy, 'precision ', precision, 'recall ', recall)

    print('----------affiliation-based F1----------')
    f1, P, R = get_metrics_affiliation(target, pred_label)
    print('f1 ', f1, 'precision ', P, 'recall ', R)

    print('----------composite F1----------')
    EventF1PA1, prec_t, rec_e = get_metrics_fc1(target, pred_label)
    print('f1 ', EventF1PA1, 'precision ', prec_t, 'recall ', rec_e)