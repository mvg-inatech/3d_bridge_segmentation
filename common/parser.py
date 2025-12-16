import yaml

#################################################################
# general yml config parse


def get_params(config_file):
    with open(config_file, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg


def get_labels(config_file="config/labels.yml"):
    with open(config_file, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg


#################################################################
# model config parse


def create_class_from_dict(class_name, attributes):
    return type(class_name, (object,), attributes)


def parse_dict(data):
    attributes = {}
    for key, value in data.items():
        if isinstance(value, dict):
            attributes[key] = parse_dict(value)
        else:
            attributes[key] = value
    return attributes


def yaml_cfg_to_class(yaml_file):
    with open(yaml_file, "r") as file:
        yaml_content = yaml.safe_load(file)

    # Define the class name
    class_name = yaml_content.get("name", "model_config")
    # Extract and parse attributes from YAML content
    attributes = parse_dict(yaml_content.get("model", {}))
    # Dynamically create the class
    new_class = create_class_from_dict(class_name, attributes)

    return new_class
