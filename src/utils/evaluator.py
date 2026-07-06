import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, classification_report
import numpy as np

class Evaluator:
    def __init__(self, args):
        self.args = args

    def evaluate(self, model, dataset, split, epoch = None):
        # self.args.logger.write('\n' + '='*60)
        eval_ind = dataset.splits[split]
        num_samples = len(eval_ind)
        model.eval()

        true, pred = [], []
        num_batches = (num_samples + self.args.eval_batch_size - 1) // self.args.eval_batch_size
        for i, start in enumerate(range(0, num_samples, self.args.eval_batch_size)):
            batch_ind = eval_ind[start:min(num_samples,
                                           start+self.args.eval_batch_size)]
            batch = dataset.get_batch(batch_ind)
            true.append(batch['labels'])
            del batch['labels']
            batch = {k:v.to(self.args.device) for k,v in batch.items()}
            with torch.no_grad():
                pred.append(model(**batch).cpu())
        
        true, pred = torch.cat(true), torch.cat(pred)
        true_np = true.numpy()
        pred_np = pred.numpy()
        
        precision, recall, thresholds = precision_recall_curve(true_np, pred_np)
        pr_auc = auc(recall, precision)
        minrp = np.minimum(precision, recall).max()
        roc_auc = roc_auc_score(true_np, pred_np)
        
        # Calculate accuracy using threshold 0.5
        pred_labels = (pred_np >= 0.5).astype(int)
        accuracy = np.mean(true_np == pred_labels)
        
        result = {'auroc':roc_auc, 'auprc':pr_auc, 'minrp':minrp, 'accuracy':accuracy}
        
        if epoch is not None:
            self.args.logger.write('{:<12} Epoch: {:5d} | AUROC: {:.4f} | AUPRC: {:.4f} | MinRP: {:.4f}'.format(
                split.upper(), epoch, roc_auc, pr_auc, minrp))
        else:
            self.args.logger.write('{:<12} AUROC: {:.4f} | AUPRC: {:.4f} | MinRP: {:.4f} | Accuracy: {:.4f}'.format(
                split.upper(), roc_auc, pr_auc, minrp, accuracy))
        
        # Print confusion matrix and classification report for test set
        if split == 'test':
            self.args.logger.write('\n' + '='*60)
            self.args.logger.write('Classification Report:')
            for report_line in classification_report(true_np, pred_labels, digits=4).split('\n'):
                self.args.logger.write(report_line)
            self.args.logger.write('Confusion Matrix:')
            for cm_line in confusion_matrix(true_np, pred_labels, labels=[0, 1]).tolist():
                self.args.logger.write(str(cm_line))
            self.args.logger.write('='*60)
        
        return result

