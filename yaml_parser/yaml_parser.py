import yaml

with open('F:/kjYun/2022/07_onnx/config.yml') as file:
    try:
        data = yaml.safe_load(file)
        for key, value in data.items():
            print(key, ":", value)
    except yaml.YAMLError as exception:
        print(exception)
