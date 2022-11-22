import os
import yaml
import onnx
import argparse
import onnxruntime as ort
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Parsing yaml to extract onnx model')

    parser.add_argument('--debug', action='store_true', help='Debugging mode')

    parser.add_argument(
        '--onnx_file',
        dest='onnx_file',
        help='The onnx model path.',
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        '--yaml_dir',
        dest='yaml_dir',
        help='The yaml file path.',
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the extract model.',
        type=str,
        default='./output'
    )
    return parser.parse_args()


def extract_model(onnx_path, yaml_dir_path, save_dir_path):
    model = onnx.load(onnx_path)
    input_all = []

    for input in model.graph.node:
        input_all.append(input.input)
    output_all = [output.name for output in model.graph.node]

    file_list = os.listdir(yaml_dir_path)
    file_list_yaml = [file for file in file_list if file.endswith(".yaml")]

    # Read YAML data.
    print('Reading YAML file "%s"' % file_list_yaml)
    data = None
    for file in file_list_yaml:
        with open(yaml_dir_path + file, "r") as stream:
            try:
                data = yaml.safe_load_all(stream)
                print(file)
            except yaml.YAMLError as err:
                print(err)

            # Search YAML entry for node value.
            partition = []
            for item in data:
                for idx, node in enumerate(item):
                    if "NodeOutputName" in node:
                        splitValue1 = node["NodeOutputName"].split("__")[0]
                        splitValue2 = splitValue1.split(":0")[0]
                        partition.append(splitValue2)
            n = []
            for i in input_all:
                if list(set(i) & set(partition)):
                    n.append(i)

            m = list(set(output_all) & set(partition))[0]
            for i in list(set(output_all) & set(partition)):
                if output_all.index(i) > output_all.index(m):
                    m = i

            if output_all.index(m) > len(partition):
                print('The index of a node "%d" cannot be greater than list size "%d"!' % (
                    output_all.index(m), len(partition)))
                exit(1)

            # Extract onnx model.
            input_names = n[0]
            output_names = [m]
            print(input_names)
            print(output_names)
            try:
                onnx.utils.extract_model(onnx_path, save_dir_path + file.split(".yaml")[0] + '.onnx',
                                         input_names, output_names)
            except onnx.onnx_cpp2py_export.checker.ValidationError as e:
                print("pass validation error during exporting.")
                pass


def read_onnx(onnx_path, extracted_onnx_path):
    file_list = os.listdir(extracted_onnx_path)
    file_list_onnx = [file for file in file_list if file.endswith(".onnx")]

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name

    data = np.ones(input_shape, dtype=float)
    data = data.astype(np.float32)

    result = session.run([output_name], {input_name: data})
    sessions = []
    # ONNXRuntime functions

    for file in file_list_onnx:
        sess = ort.InferenceSession(extracted_onnx_path + file, providers=['CPUExecutionProvider'])
        sessions.append(sess)

    result_data = []
    for sess in sessions:
        if sessions.index(sess) == 0:
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
            result_data = sess.run([output_name], {input_name: data})
        else:
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
            result_data = sess.run([output_name], {input_name: result_data[0]})

    return result, result_data


def compareValue_func(result, result2):
    # 10의 -6승까지 비교
    return np.count_nonzero((np.abs(result[0] - result2[0]) > 0.000001).astype(int))


def main(args):
    # Extract ONNX model
    extract_model(args.onnx_file, args.yaml_dir, args.save_dir)

    # Read ONNX models
    res1, res2 = read_onnx(args.onnx_file, args.save_dir)

    print(np.count_nonzero((np.abs(res1[0] - res2[0]) > 0.000001).astype(int)) == 0)


if __name__ == '__main__':
    args = parse_args()
    main(args)


# ------------------------------------------------------------------------------------------
# Function Test
# ------------------------------------------------------------------------------------------

def test_compareValue():
    res1, res2 = read_onnx(args.onnx_file, args.save_dir)
    assert compareValue_func(res1, res2) == 0
