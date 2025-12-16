from models.point_transformer_v2 import PointTransformerV2
from models.spconv_unet import SpConvUNet
from models.kp_conv import KPFCNN
from common.parser import yaml_cfg_to_class
from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def get_model(name, config_dir):
    config = yaml_cfg_to_class(config_dir)
    if name == "point_transformer_v2":
        assert (
            config.__name__ == "point_transformer_v2"
        ), "Wrong config for model PointTransformerV2"
        model = PointTransformerV2(config)
    elif name == "spconv_unet":
        assert config.__name__ == "spconv_unet", "Wrong config for model SpConvUNet"
        model = SpConvUNet(config)
    elif name == "kp_conv":
        assert config.__name__ == "kp_conv", "Wrong config for model KPConv"
        model = KPFCNN(config)
    else:
        raise NotImplementedError

    print("Model: {} ".format(name))
    count_parameters(model)
    return model
