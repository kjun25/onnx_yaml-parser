import argparse
import os
import yaml
import onnx

# Command line options.
parser = argparse.ArgumentParser(
    usage="YAML parser for extract onnx"
)
parser.add_argument('--debug', action='store_true', help='debugging mode')

parser.add_argument(
    "-x", "--onnx", dest="onnx", required=True, type=str, help="ONNX file path."
)

parser.add_argument(
    "-e", "--ex", dest="extractedOnnx", required=True, type=str, help="Extracted ONNX file path."
)

parser.add_argument(
    "-y", "--yaml", dest="yaml", required=True, type=str, help="YAML file path."
)

parser.add_argument(
    "-i",
    "--input",
    dest="inputName",
    required=True,
    type=str,
    help="NodeInputName.",
)
parser.add_argument(
    "-o",
    "--output",
    dest="outputName",
    required=True,
    type=str,
    help="NodeOutputName.",
)

args = parser.parse_args()

# Get arguments.
onnxFile = args.onnx
extractedOnnxFile = args.extractedOnnx
yamlFile = args.yaml
nodeInputName = args.inputName
nodeOutputName = args.outputName

# Verify profile exists.
if not os.path.isfile(yamlFile):
    print('File "%s" not found!' % yamlFile)
    exit(1)

# Read YAML data.
print('Reading file "%s" ...' % yamlFile)
data = None
with open(yamlFile, "r") as stream:
    try:
        data = yaml.safe_load_all(stream)
    except yaml.YAMLError as err:
        print(err)

    # Search YAML entry for node value.
    print('Searching node value name "%s" ...' % nodeOutputName)
    entry = None
    for item in data:
        for i in item:
            if "NodeOutputName" in i:
                if i["NodeOutputName"].split("__")[0] == nodeOutputName:
                    entry = i
                    print(entry)
    if not entry:
        print('Node value "%s" not found!' % nodeOutputName)
        exit(1)

print("Input onnx file path: " + onnxFile, "Output extracted onnx file path" + extractedOnnxFile, sep=", ")
print("Input node name: " + nodeInputName, "Output node name: " + nodeOutputName, sep=", ")

onnx.utils.extract_model(onnxFile, extractedOnnxFile, nodeInputName, nodeOutputName)
