import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix, balanced_accuracy_score



def root_brier_score(labels, y_pred):
    brier_score = brier_score_loss(labels, y_pred)
    root_brier = np.sqrt(brier_score)
    return root_brier

def calculate_metrics(y_true, y_pred_probs, threshold=0.5, is_multilabel=False):
    """
    Calculate performance metrics for binary or multi-label classification.

    Args:
        y_true: Ground truth labels, shape (N,) for binary or (N, num_labels) for multi-label
        y_pred_probs: Predicted probabilities (softmax or sigmoid outputs), shape (N,) for binary or (N, num_labels) for multi-label
        threshold: Probability threshold to convert probabilities to binary labels (default 0.5)
        is_multilabel: Whether the task is multi-label classification (default False)

    Returns:
        Dictionary of performance metrics
    """
    # For binary classification
    if not is_multilabel:
        # For binary classification, y_pred_probs should be the probabilities for class 1 (positive class)
        y_pred = (y_pred_probs >= threshold).astype(int)

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        bal_accuracy = balanced_accuracy_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        # Confusion matrix to calculate sensitivity and specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (fp + tn)  # True Negative / (False Positive + True Negative)

        auc = roc_auc_score(y_true, y_pred_probs)  # AUC for binary classification

        return {
            'accuracy': accuracy,
            'bal_accuracy': bal_accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'auc': auc,
            'f1_score': f1,
        }

    # For multi-label classification
    else:
        # For multi-label classification, threshold the probabilities for each label/class
        y_pred = (y_pred_probs >= threshold).astype(int)

        # Calculate accuracy (average of per-label accuracy)
        bal_accuracy = balanced_accuracy_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        # Precision, recall, and F1 score averaged across all labels (macro-average)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        # AUC for multi-label classification (average over all classes)
        auc = roc_auc_score(y_true, y_pred_probs, average='macro', multi_class='ovr')

        return {
            'accuracy': accuracy,
            'bal_accuracy': bal_accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'f1_score': f1,
        }

def negative_predictive_value(y_true, y_pred):
    """
    Calculate the negative predictive value (NPV) for binary classification.

    Parameters:
    y_true (np.array): The true labels.
    y_pred (np.array): The predicted labels.

    Returns:
    npv (float): The negative predictive value.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    npv = tn / (tn + fn)
    
    return npv

def calculate_ese(true_labels, pos_confidences, num_bins=15, scheme='equal'): # From nannyml paper 2025
    """Calculates the calibration error in either in the form of ECE or AdaECE
    Parameters:
        true_labels (NumPy Array): The (binary) labels for all the datapoints used in the calculations
        pos_confidences (NumPy Array): The confidences for the positive class for all the datapoints used in the calculations
        num_bins (int): The number of bins used 
        scheme (srt): Either 'equal' for equiwidth binning (ECE) or 'dynamic' for adaptive binning (AdaECE)
    Returns:
        ece (float): The calibration error     
    """
    # Perform binning
    if scheme == 'equal':
        bins = np.linspace(0.0, 1.0, num_bins + 1)
    elif scheme == 'dynamic':
        borders = np.linspace(0.0, 1.0, num_bins + 1)
        bins = np.array([np.quantile(pos_confidences, q) for q in borders])
        if np.isnan(bins).any():
            print("Revert to equiwidth binning")
            bins = np.linspace(0.0, 1.0, num_bins + 1)
    else:
        raise NameError(f"Binning scheme '{scheme}' is not recognized.")
    bin_indices = np.digitize(pos_confidences, bins, right=True)
    
    # Bin indices should range from 1 to num_bins
    zero_mask = bin_indices[bin_indices == 0]
    zero_amount = zero_mask.sum()
    if zero_amount > 0:
        print(f"{zero_amount} zero indices found. Replace with ones.")
        bin_indices[zero_mask] = 1
        
    # Calculate statistics for each bin
    bin_fraction_of_positives = np.zeros(num_bins, dtype=float)
    bin_confidences = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=int)

    for b in range(num_bins):
        selected = np.where(bin_indices == b + 1)[0]
        if len(selected) > 0:
            bin_fraction_of_positives[b] = np.mean(true_labels[selected] == 1)
            bin_confidences[b] = np.mean(pos_confidences[selected])
            bin_counts[b] = len(selected)

    gaps = np.abs(bin_fraction_of_positives - bin_confidences)

    # Calculate statistics over all bins
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    return ece


def CBPE_accuracy(outs, balanced=False):
    """
    Calculate the estimated accuracy of the model outputs.

    Parameters:
    outs (list or numpy array): List or array of model output scores.

    Returns:
    float: Estimated accuracy.
    """    
    # Calculate estimated performance metrics
    TP = []
    TN = []
    FP = []
    FN = []

    for score in outs:
        pred = np.round(score)
        p_not_eq = np.abs(pred - score)
        p_eq = 1 - p_not_eq 
    
        if pred == 1:
            TP.append(p_eq)
            FP.append(p_not_eq)
        else:
            TN.append(p_eq)
            FN.append(p_not_eq)
    if not balanced:
        acc_estim = (np.sum(TP) + np.sum(TN)) / len(outs)
    else:
        acc_estim = 0.5 * (np.sum(TP)/(np.sum(TP) + np.sum(FN)) + np.sum(TN)/(np.sum(TN) + np.sum(FP)))
    return acc_estim

def CBPE_auroc(outs, comet_logger=None, cfg=None, class_names=None, show_plots=True):
    """
    Calculate the estimated AUROC (Area Under the Receiver Operating Characteristic curve) of the model outputs.

    Parameters:
    outs (list or numpy array): List or array of model output scores.
    comet_logger (object): Logger object for logging the ROC curve.
    cfg (object): Configuration object.
    model_pathology (str): pathology name.

    Returns:
    None
    """
    # Calculate estimated performance metrics
    TPR_list = []
    FPR_list = []

    for t in np.linspace(0, 1, 100):
        TP = []
        TN = []
        FP = []
        FN = []

        for score in outs:
            # Round based on threshold t
            pred = (score >= t).astype(int)
            
            p_not_eq = np.abs(pred - score)
            p_eq = 1 - p_not_eq 
        
            if pred == 1:
                TP.append(p_eq)
                FP.append(p_not_eq)
            else:
                TN.append(p_eq)
                FN.append(p_not_eq)

        TPR_list.append(np.sum(TP) / (np.sum(TP) + np.sum(FN)))
        FPR_list.append(np.sum(FP) / (np.sum(FP) + np.sum(TN)))
    
    fig = plt.figure()
    plt.plot(FPR_list, TPR_list)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC curve {class_names}')
    plt.xlim(0, 1)
    plt.ylim(0, 1)


    if show_plots:
        plt.show()
    plt.close(fig)

    auroc_estim = np.trapz(TPR_list[::-1], FPR_list[::-1])
    return auroc_estim

def CBPE_F1(outs):
    """
    Calculate the estimated F1 score of the model outputs.

    Parameters:
    outs (list or numpy array): List or array of model output scores.

    Returns:
    float: Estimated F1 score.
    """
    # Calculate estimated performance metrics
    TP = []
    FP = []
    FN = []

    for score in outs:
        pred = np.round(score)
        p_not_eq = np.abs(pred - score)
        p_eq = 1 - p_not_eq 
    
        if pred == 1:
            TP.append(p_eq)
            FP.append(p_not_eq)
        else:
            FN.append(p_not_eq)
    # Debug print
    # print('TP:', TP, '\nFP:', FP, '\nFN:', FN)
    # print('TP:', np.sum(TP), 'FP:', np.sum(FP), 'FN:', np.sum(FN))
    precision = np.sum(TP) / (np.sum(TP) + np.sum(FP))
    recall = np.sum(TP) / (np.sum(TP) + np.sum(FN))
    f1_estim = 2 * (precision * recall) / (precision + recall)
    return f1_estim

def CBPE_plr(outs):
    # Calculate estimated performance metrics
    TP = []
    FP = []
    FN = []
    TN = []

    for score in outs:
        pred = np.round(score)
        p_not_eq = np.abs(pred - score)
        p_eq = 1 - p_not_eq 
    
        if pred == 1:
            TP.append(p_eq)
            FP.append(p_not_eq)
        else:
            FN.append(p_not_eq)
            TN.append(p_eq)
    
    recall = np.sum(TP) / (np.sum(TP) + np.sum(FN))
    specificity = np.sum(TN) / (np.sum(TN) + np.sum(FP))
    plr = recall / (1 - specificity)
    return plr

def CBPE_confusion_matrix(outs):
    """
    Calculate the estimated confusion matrix of the model outputs.

    Parameters:
    outs (list or numpy array): TP, FP, FN, TN.

    Returns:
    numpy array: Estimated confusion matrix.
    """
    # Calculate estimated performance metrics
    TP = []
    FP = []
    FN = []
    TN = []

    for score in outs:
        pred = np.round(score)
        p_not_eq = np.abs(pred - score)
        p_eq = 1 - p_not_eq 
    
        if pred == 1:
            TP.append(p_eq)
            FP.append(p_not_eq)
        else:
            FN.append(p_not_eq)
            TN.append(p_eq)
    TP = np.sum(TP)
    FP = np.sum(FP)
    FN = np.sum(FN)
    TN = np.sum(TN)


    return TP, FP, TN, FN

def CBPE_precision(outs):
    TP, FP, TN, FN = CBPE_confusion_matrix(outs)
    precision = TP / (TP + FP)
    return precision

def CBPE_recall(outs):
    """
    Calculate the recall based on the CBPE confusion matrix.
    Recall = TP / (TP + FN)
    """
    TP, FP, TN, FN = CBPE_confusion_matrix(outs)
    denominator = (TP + FN)
    if denominator == 0:
        return 0.0
    return TP / denominator

def CBPE_specificity(outs):
    """
    Calculate the specificity based on the CBPE confusion matrix.
    Specificity = TN / (TN + FP)
    """
    TP, FP, TN, FN = CBPE_confusion_matrix(outs)
    denominator = (TN + FP)
    if denominator == 0:
        return 0.0
    return TN / denominator

def ATC_metric_estim(val_outs, test_outs, value, threshold=0.5, debug=False):
    s_val = [val_s[0] if val_s >= threshold else 1 - val_s[0] for val_s in val_outs]
    s_test = [test_s[0] if test_s >= threshold else 1 - test_s[0] for test_s in test_outs]

    # print(acc)
    # threshs = np.linspace(0.5, 1, 1000)
    # min_diff = 1e5
    # best_t = 0
    # for t in threshs:
    #     counter = np.sum(np.where(s_val <= t, 1, 0))
    #     diff = np.abs(counter/len(s_val) - value)
    #     if diff < min_diff:
    #         min_diff = diff
    #         best_t = t
    
    best_t = np.percentile(s_val, (1-value) * 100)
    acc_pred_test = np.sum(np.where(s_test >= best_t, 1, 0)) / len(s_test)
    
    if debug:
        print(f'ATC metric: {value}')
        print('best_t: ', best_t, ' with ', np.sum(np.where(s_val >= best_t, 1, 0))/len(s_val))
        print('acc_pred_test: ', acc_pred_test)
        plt.hist(s_val, density=False, label=f'val acc: {value}')
        plt.vlines(best_t, 0,len(s_val), colors='r', label=f'best_t = {best_t}')
        plt.legend()
        plt.show()
        plt.hist(s_test, density=False, label=f'test acc: {acc_pred_test}')
        plt.legend()
        plt.vlines(best_t, 0, len(s_test), colors='r')
        plt.show()
    return acc_pred_test

def average_confidence(outs):
    return np.mean(outs)


def calculate_CBPE_metrics(y_pred_probs):
    accuracy = CBPE_accuracy(y_pred_probs)
    bal_accuracy = CBPE_accuracy(y_pred_probs, balanced=True)
    precision = CBPE_precision(y_pred_probs)
    recall = CBPE_recall(y_pred_probs)
    specificity = CBPE_specificity(y_pred_probs)
    f1 = CBPE_F1(y_pred_probs)
    auroc = CBPE_auroc(y_pred_probs, show_plots=False)

    return {
        'accuracy': accuracy,
        'bal_accuracy': bal_accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'auc': auroc,
        'f1_score': f1,
    }

def calculate_ATC_metrics(y_pred_validation, y_labels_validation, y_pred_test_probs, is_multilabel=False):
    realized_val_metrics = calculate_metrics(y_labels_validation, y_pred_validation, is_multilabel=is_multilabel)

    # Calculate ATC metrics
    accuracy = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['accuracy'])
    bal_accuracy = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['bal_accuracy'])
    precision = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['precision'])
    recall = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['recall'])
    specificity = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['specificity'])
    f1 = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['f1_score'])
    auroc = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['auc'])

    return {
        'accuracy': accuracy,
        'bal_accuracy': bal_accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'auc': auroc,
        'f1_score': f1,
    }

