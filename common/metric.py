"""
originally from :
    https://github.com/PRBonn/lidar-bonnetal/blob/master/train/tasks/semantic/modules/ioueval.py
"""

import os
import numpy as np
import torch


class IoUEval:
    """
    @Note switched definition compared to original implementation
              P
            1   0
    GT  1   TP  FN
        0   FP  TN
    """

    def __init__(self, n_classes, device):
        self.n_classes = n_classes
        self.device = device
        self.reset()

    def num_classes(self):
        return self.n_classes

    def reset(self):
        self.conf_matrix = torch.zeros(
            (self.n_classes, self.n_classes), device=self.device
        ).long()
        self.ones = None
        self.last_scan_size = None  # for when variable scan size is used

    def add_batch(self, x, y):  # x=preds, y=targets
        # if numpy, pass to pytorch
        # to tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.array(x)).long().to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(np.array(y)).long().to(self.device)

        # sizes should be "batch_size x H x W"
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify

        # idxs are labels and predictions
        idxs = torch.stack([y_row, x_row], dim=0)

        # ones is what I want to add to conf when I
        if self.ones is None or self.last_scan_size != idxs.shape[-1]:
            self.ones = torch.ones((idxs.shape[-1]), device=self.device).long()
            self.last_scan_size = idxs.shape[-1]

        # make confusion matrix (cols = PRED, rows = GT)
        self.conf_matrix = self.conf_matrix.index_put_(
            tuple(idxs), self.ones, accumulate=True
        )

        # print(self.tp.shape)
        # print(self.fp.shape)
        # print(self.fn.shape)

    def get_stats(self):
        # remove fp and fn from confusion on the ignore classes cols and rows
        conf = self.conf_matrix.clone().double()

        # get the clean stats
        tp = conf.diag()
        fp = conf.sum(dim=1) - tp
        fn = conf.sum(dim=0) - tp
        return tp, fp, fn

    def get_IoU(self):
        tp, fp, fn = self.get_stats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        return iou

    def get_mIoU(self):
        return self.get_IoU().mean()

    def get_Acc(self):
        tp, fp, _ = self.get_stats()
        total_tp = tp.sum()
        total = tp.sum() + fp.sum() + 1e-15
        acc_mean = total_tp / total
        return acc_mean  # returns "acc mean"

    def get_precision_vec(self):
        tp = self.conf_matrix.diag()
        tp_plus_fn = self.conf_matrix.sum(dim=0)
        prec = tp / (tp_plus_fn + 1e-15)
        return prec

    def get_recall_vec(self):
        tp = self.conf_matrix.diag()
        tp_plus_fp = self.conf_matrix.sum(dim=1)
        rec = tp / (tp_plus_fp + 1e-15)
        return rec

    def get_F1_score(self):
        tp = self.conf_matrix.diag()
        tp_plus_fn = self.conf_matrix.sum(dim=0)
        tp_plus_fp = self.conf_matrix.sum(dim=1)
        F1 = 2 * tp / (tp_plus_fp + tp_plus_fn + 1e-6)
        return F1

    def save_conf_matrix(self, path):
        np.save(
            os.path.join(path, "confusion_matrix.npy"), self.conf_matrix.cpu().numpy()
        )

    def print_stats(self):
        iou = self.get_IoU()
        mIoU = self.get_mIoU()
        acc = self.get_Acc()
        print("IoU Evaluation Results:")
        print(f"mIoU: {mIoU:.4f}")
        print(f"IoU: {np.round(iou.cpu().numpy(), 3)*100}")
        print(f"pwAcc: {acc:.4f}")
        print(f"Precision: {self.get_precision_vec().cpu().numpy()}")
        print(f"Recall: {self.get_recall_vec().cpu().numpy()}")
        print(f"F1 Score: {self.get_F1_score().cpu().numpy()}")
        print("______________________________________")
