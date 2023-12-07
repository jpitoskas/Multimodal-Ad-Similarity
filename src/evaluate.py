from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, auc



# Val/Test function
def test(pair_loader, model, processor, loss_fn, device, thresholds=torch.arange(-1, 1, 0.001), optimization_metric='precision', fbeta=0.5, test_threshold=None):

    
    all_similarities = []
    all_targets = []


    model.eval()
    running_loss = 0
    with torch.no_grad():
        for _, (data, targets) in enumerate(tqdm(pair_loader)):

            (text1, image1), (text2, image2) = data
    
            image1 = image1.to(device)
            image2 = image2.to(device)

            targets = targets.to(device)
            
            inputs1 = processor(text=text1, images=image1, return_tensors="pt", padding=True, truncation=True)
            inputs2 = processor(text=text2, images=image2, return_tensors="pt", padding=True, truncation=True)

            # Move tensors to the device
            inputs1 = {key: value.to(device) for key, value in inputs1.items()}
            inputs2 = {key: value.to(device) for key, value in inputs2.items()}
        

            outputs1, outputs2 = model(inputs1, inputs2)


            cosine_similarities = F.cosine_similarity(outputs1, outputs2)

            all_similarities.extend(cosine_similarities.tolist())
            all_targets.extend(targets.tolist())




            loss = loss_fn(outputs1, outputs2, targets)
            running_loss += loss.item() * targets.size(0) # smaller batches count less

        
        
        metrics, optimal_threshold, auc = optimal_metric_score_with_threshold(similarities=torch.tensor(all_similarities),
                                                                              y_true=torch.tensor(all_targets),
                                                                              thresholds=thresholds,
                                                                              optimization_metric=optimization_metric,
                                                                              return_auc=True,
                                                                              fbeta=fbeta)
        
        if test_threshold is not None:
            metrics, _ = optimal_metric_score_with_threshold(similarities=torch.tensor(all_similarities),
                                                             y_true=torch.tensor(all_targets),
                                                             thresholds=torch.tensor([test_threshold]),
                                                             return_auc=False,
                                                             fbeta=fbeta)

        running_loss /= len(pair_loader.dataset)
        
        return running_loss, metrics, optimal_threshold, auc


    
# def predict_by_threshold(similarities, threshold):
#     preds = (torch.tensor(similarities) > threshold).int()
#     return preds


# def get_classification_metrics(y_true, y_pred):

#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

#     accuracy = (tp + tn) / (tn + fp + fn + tp)
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     f1_score = tp / (tp + 0.5 * (fp + fn))

#     classification_metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score}

#     return classification_metrics

def get_confusion_matrix_by_thresholds(similarities, y_true, thresholds):


    y_pred = predict_by_thresholds(similarities, thresholds)

    tp = torch.logical_and(y_pred.bool(), y_true.view(1, -1).bool()).sum(dim=1)
    fp = torch.logical_and(y_pred.bool(), ~y_true.view(1, -1).bool()).sum(dim=1)
    tn = torch.logical_and(~y_pred.bool(), ~y_true.view(1, -1).bool()).sum(dim=1)
    fn = torch.logical_and(~y_pred.bool(), y_true.view(1, -1).bool()).sum(dim=1)



    return tp, fp, tn, fn


def predict_by_thresholds(similarities, thresholds):
    preds = (similarities > thresholds.unsqueeze(1)).int()
    return preds


def get_classification_metrics_by_threholds(y_true, y_pred, fbeta=0.5):

    tp = torch.logical_and(y_pred.bool(), y_true.view(1, -1).bool()).sum(dim=1)
    fp = torch.logical_and(y_pred.bool(), ~y_true.view(1, -1).bool()).sum(dim=1)
    tn = torch.logical_and(~y_pred.bool(), ~y_true.view(1, -1).bool()).sum(dim=1)
    fn = torch.logical_and(~y_pred.bool(), y_true.view(1, -1).bool()).sum(dim=1)

    # print(tp, fp, tn, fn)


    accuracy = (tp + tn) / (tn + fp + fn + tp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = tp / (tp + 0.5 * (fp + fn))
    f_beta = (1 + fbeta**2) * (precision * recall / ((fbeta**2)*precision + recall))



    classification_metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score, f'f_beta(beta={fbeta})': f_beta}

    return classification_metrics


def optimal_metric_score_with_threshold(similarities, y_true, thresholds, optimization_metric='precision', return_auc=True, fbeta=0.5, return_metrics_per_threshold=False):

    y_pred = predict_by_thresholds(similarities, thresholds)
    classification_metrics = get_classification_metrics_by_threholds(y_true, y_pred, fbeta=fbeta)

    if return_auc:
        pr_auc = auc(classification_metrics['recall'], classification_metrics['precision'])


    max_idx = classification_metrics[optimization_metric].argmax()
    optimal_threshold = thresholds[max_idx].item()

    if return_metrics_per_threshold:
        optimal_metrics = classification_metrics
    else:
        optimal_metrics = {metric_key : metric_by_threshold[max_idx].item() for metric_key, metric_by_threshold in classification_metrics.items()}


    if return_auc:
        return optimal_metrics, optimal_threshold, pr_auc
    else:
        return optimal_metrics, optimal_threshold


    





# def binary_search_threshold(similarities, y_true, lo=-1, hi=1, optimization_metric='precision'):
    
    
#     if optimization_metric not in ['accuracy', 'precision', 'recall', 'f1_score']:
#         raise ValueError("Invalid metric provided")

#     best_threshold = 0
#     best_metric_score = 0
    
#     while lo <= hi:
#         mid = (lo + hi) / 2
#         y_pred = predict_by_threshold(similarities, mid)

#         metrics = get_classification_metrics(y_true, y_pred)

        
#         if metrics[optimization_metric] > best_metric_score:
#             best_metric_score = metrics[optimization_metric]
#             best_threshold = mid
        

#         if metrics[optimization_metric] > 0.5:
#             lo = mid + 0.0001  # Adjust this value for precision
#         else:
#             hi = mid - 0.0001  # Adjust this value for precision
            
#     return best_threshold, best_metric_score





def compute_similarities(pair_loader, model, processor, device):
    
    all_similarities = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for _, (data, targets) in enumerate(tqdm(pair_loader)):

            (text1, image1), (text2, image2) = data

            image1 = image1.to(device)
            image2 = image2.to(device)

            targets = targets.to(device)

            inputs1 = processor(text=text1, images=image1, return_tensors="pt", padding=True, truncation=True)
            inputs2 = processor(text=text2, images=image1, return_tensors="pt", padding=True, truncation=True)

            # Move tensors to the device
            inputs1 = {key: value.to(device) for key, value in inputs1.items()}
            inputs2 = {key: value.to(device) for key, value in inputs2.items()}


            outputs1, outputs2 = model(inputs1, inputs2)


            cosine_similarities = F.cosine_similarity(outputs1, outputs2)

            all_similarities.extend(cosine_similarities.tolist())
            all_targets.extend(targets.tolist())
    
    return all_similarities, all_targets




