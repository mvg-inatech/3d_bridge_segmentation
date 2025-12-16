import torch
import numpy as np
from common.metric import IoUEval
from dataloader.loader_utils import sample_to_device


def train_epoch(model, optimizer, criterion, loader, device):
    model = model.to(device)
    model.train()

    loss_list = list()

    for i, sample in enumerate(loader):
        sample = sample_to_device(sample, device)
        pred = model(sample)
        loss = criterion(pred, sample["labels"])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.item())

        print(
            ("TRAIN: input {}  of {} --> Loss: {}").format(
                i, len(loader), np.round(np.mean(loss_list), 4)
            ),
            end="\r",
        )
    # clear prints before
    ctrl = "                    "
    print(ctrl + ctrl + ctrl + ctrl)

    return loss_list


def eval_epoch(model, loader, device, n_classes):
    model = model.to(device)
    model.eval()

    iou_eval = IoUEval(n_classes, device)

    for i, sample in enumerate(loader):
        with torch.no_grad():
            sample = sample_to_device(sample, device)
            pred = model(sample)
            pred_max = torch.argmax(pred, 1)

            iou_eval.add_batch(pred_max, sample["labels"])

            m_accuracy = np.round(iou_eval.get_Acc().cpu() * 100, 4)
            m_jaccard = np.round(iou_eval.get_mIoU().cpu() * 100, 4)

            print(
                ("EVAL: input {} of {}\t --> mIoU: {} pwAcc: {}").format(
                    i, len(loader), m_jaccard, m_accuracy
                ),
                end="\r",
            )
    # clear prints before
    ctrl = "                    "
    print(ctrl + ctrl + ctrl + ctrl)

    mean_miou = iou_eval.get_mIoU().cpu()
    mean_iou = iou_eval.get_IoU().cpu()

    confusion_matrix = iou_eval.conf_matrix.cpu().numpy()

    return confusion_matrix, mean_miou, mean_iou
