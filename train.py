import argparse
import torch
import yaml
from os.path import join, exists
from os import makedirs
from datetime import datetime
import torch.nn as nn
import numpy as np
from dataloader.dataloder import get_dataloader
from common.parser import get_params
from models.model_loader import get_model
from engine import train_epoch, eval_epoch


def parse_arguments(parser):
    parser.add_argument("config_dir", type=str, help="dir to config file")
    args = parser.parse_args()
    return args


def get_learning_rate(optimizer):
    learning_rate = 0
    for param_group in optimizer.param_groups:
        learning_rate = param_group["lr"]
    return learning_rate


def create_directory(path):
    now = datetime.now()
    now = now.strftime("%d:%m:%Y-%H:%M")
    path = join(path, now)
    if not exists(path):
        makedirs(path)
    return path


def save_config(config, path):
    file = open(path + "/config.yaml", "w")
    yaml.safe_dump(config, file)


def save_model(model, path):
    name = "bird.pt"
    file_name = join(path, name)
    torch.save(model.state_dict(), file_name)


def save_conf_matrix(conf_matrix, path):
    np.save(join(path, "confusion_matrix.npy"), conf_matrix)


def main(args):
    config = get_params(args.config_dir)
    model = get_model(config["name"], args.config_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on {}".format(device))

    path = create_directory(config["save_dir"])
    save_config(config, path)

    dataloader_train = get_dataloader(config)
    dataloader_val = get_dataloader(config, train=False)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["train"]["lr"],
        momentum=config["train"]["mom"],
        weight_decay=config["train"]["wd"],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config["train"]["gamma"]
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_iou = 0
    for ep in range(config["train"]["ep"]):
        print(
            "Starting with Epoch: {} and LR: {}".format(
                ep, get_learning_rate(optimizer)
            )
        )

        loss_list = train_epoch(model, optimizer, criterion, dataloader_train, device)
        print("TRAIN\t --> mLoss: {}".format(np.mean(loss_list)))

        conf_matrix, miou, iou = eval_epoch(
            model,
            dataloader_val,
            device,
            config["model"]["num_classes"],
        )
        print("VAL\t --> mIoU: {}".format(miou))
        if miou > best_val_iou:
            best_val_iou = miou
            print("Best mean iou in validation set so far, save model!")
            print(iou)
            save_model(model, path)
            save_conf_matrix(conf_matrix, path)

        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="training",
        description="run training for specified model and data",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    args = parse_arguments(parser)
    main(args)
