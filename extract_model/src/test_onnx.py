import os
import yaml
import onnx
import pytest
import argparse
import numpy as np
import onnxruntime as ort

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
                splitValue = i["NodeOutputName"].split("__")[0]
                if splitValue == nodeOutputName:
                    entry = splitValue
                    print(entry)
    if not entry:
        print('Node value "%s" not found!' % nodeOutputName)
        exit(1)

print("Input onnx file path: " + onnxFile, "Output extracted onnx file path" + extractedOnnxFile, sep=", ")
print("Input node name: " + nodeInputName, "Output node name: " + entry, sep=", ")

# Extract onnx model.
onnx.utils.extract_model(onnxFile, extractedOnnxFile, nodeInputName, entry)

# ONNXRuntime functions
session = ort.InferenceSession(onnxFile, providers=['CPUExecutionProvider'])
session_div1 = ort.InferenceSession(extractedOnnxFile, providers=['CPUExecutionProvider'])
session_div2 = ort.InferenceSession('test2.onnx', providers=['CPUExecutionProvider'])


def extract_func(x, y, z):
    input_name = x.get_inputs()[0].name
    input_shape = x.get_inputs()[0].shape
    input_type = x.get_inputs()[0].type
    output_name = x.get_outputs()[0].name

    input_name1 = y.get_inputs()[0].name
    input_shape1 = y.get_inputs()[0].shape
    input_type1 = y.get_inputs()[0].type
    output_name1 = y.get_outputs()[0].name

    input_name2 = z.get_inputs()[0].name
    input_shape2 = z.get_inputs()[0].shape
    input_type2 = z.get_inputs()[0].type
    output_name2 = z.get_outputs()[0].name

    data = np.ones(input_shape, dtype=float)
    data = data.astype(np.float32)

    result = x.run([output_name], {input_name: data})

    result1 = y.run([output_name1], {input_name1: data})

    result2 = z.run([output_name2], {input_name2: result1[0]})

    return result, result2


def extract_part_func(x, y):
    input_name = x.get_inputs()[0].name
    input_shape = x.get_inputs()[0].shape
    input_type = x.get_inputs()[0].type
    output_name = x.get_outputs()[0].name

    input_name1 = y.get_inputs()[0].name
    input_shape1 = y.get_inputs()[0].shape
    input_type1 = y.get_inputs()[0].type
    output_name1 = y.get_outputs()[0].name

    data = np.ones(input_shape, dtype=float)
    data = data.astype(np.float32)

    result = x.run([output_name1], {input_name: data})

    result1 = y.run([output_name1], {input_name1: data})

    return result, result1


def compareValue_func(result, result2):
    # 10의 -6승까지 비교
    return np.count_nonzero((np.abs(result[0] - result2[0]) > 0.000001).astype(int))


def compareFull_func(result, result2):
    # 전체 값 비교
    return np.count_nonzero(result[0] - result2[0])


def compare_inputName(session, session_div1):
    return session.get_inputs()[0].name, session_div1.get_inputs()[0].name


def compare_inputShape(session, session_div1):
    return session.get_inputs()[0].shape, session_div1.get_inputs()[0].shape


def compare_inputType(session, session_div1):
    return session.get_inputs()[0].type, session_div1.get_inputs()[0].type


def compare_outputName(session, session_div2):
    return session.get_outputs()[0].name, session_div2.get_outputs()[0].name


def compare_outputShape(session, session_div2):
    return session.get_outputs()[0].shape, session_div2.get_outputs()[0].shape


def compare_outputType(session, session_div2):
    return session.get_outputs()[0].type, session_div2.get_outputs()[0].type


# ------------------------------------------------------------------------------------------
# Function Test
# ------------------------------------------------------------------------------------------

def test_compareValue():
    res1, res2 = extract_func(session, session_div1, session_div2)
    assert compareValue_func(res1, res2) == 0


def test_compare_inputParameters():
    nameResult1, nameResult2 = compare_inputName(session, session_div1)
    shapeResult1, shapeResult2 = compare_inputShape(session, session_div1)
    typeResult1, typeResult2 = compare_inputType(session, session_div1)

    assert nameResult1 == nameResult2
    assert shapeResult1 == shapeResult2
    assert typeResult1 == typeResult2


def test_compare_outputParameters():
    nameResult1, nameResult2 = compare_outputName(session, session_div2)
    shapeResult1, shapeResult2 = compare_outputShape(session, session_div2)
    typeResult1, typeResult2 = compare_outputType(session, session_div2)

    assert nameResult1 == nameResult2
    assert shapeResult1 == shapeResult2
    assert typeResult1 == typeResult2


if __name__ == "__main__":
    pytest.main(args)
