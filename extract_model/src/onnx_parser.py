import os
import yaml
import onnx

model = onnx.load('./mxnet_exported_resnet18.onnx')
shape_info = onnx.shape_inference.infer_shapes(model)
a = []  # 빈 리스트 생성
b = []
for idx, node in enumerate(shape_info.graph.value_info):
    # a.append(node.name)
    a.append(node.name)
#    print(idx, node.name)

path_dir = './'

file_list = os.listdir(path_dir)
file_list_yaml = [file for file in file_list if file.endswith(".yaml")]
print(len(file_list_yaml))
print("file_list_yaml: {}".format(file_list_yaml))

# Read YAML data.
print('Reading YAML file "%s"' % file_list_yaml)
data = None
with open(file_list_yaml[3], "r") as stream:
    try:
        data = yaml.safe_load_all(stream)
    except yaml.YAMLError as err:
        print(err)

    # Search YAML entry for node value.
    for item in data:
        for idx, node in enumerate(item):
            if "NodeOutputName" in node:
                splitValue = node["NodeOutputName"].split("__")[0]
                b.append(splitValue)


print(a)

m = list(set(a) & set(b))[0]
for i in list(set(a) & set(b)):
    if a.index(i) > a.index(m):
        m = i
print(a.index(m), m)
if a.index(m) > len(b):
    print('The index of a node "%d" cannot be greater than list size "%d"!' % (a.index(m), len(b)))
    exit(1)

