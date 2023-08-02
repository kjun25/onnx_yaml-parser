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
    if not os.path.isdir(save_dir_path):
        os.mkdir(save_dir_path)

    model = onnx.load(onnx_path)

    input_node = [input.input for input in model.graph.node]
    input_all = [node.name for node in model.graph.input]
    input_initializer = [node.name for node in model.graph.initializer]
    dic = {}
    for t in [output for output in model.graph.node]:
        dic[t.name] = t.output
    output_all = []
    for key, val in dic.items():
        output_all.append(key)

    net_feed_input = list(set(input_all) - set(input_initializer))

    file_list = os.listdir(yaml_dir_path)
    file_list_yaml = [file for file in file_list if file.endswith(".yaml")]
    file_list_yaml.sort()

    # Read YAML data.
    print('Reading YAML file "%s"' % file_list_yaml)
    flag = True
    data = None
    temp = None
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
            for i in input_node:
                if list(set(i) & set(partition)):
                    n.append(i)

            m = list(set(output_all) & set(partition))[0]
            for i in list(set(output_all) & set(partition)):
                if output_all.index(i) > output_all.index(m):
                    m = i

            # Extract onnx model.
            if flag:
                input_names = net_feed_input
            else:
                input_names = net_feed_input
                input_names.append(temp)
            output_names = dic[m]
            print("Input Node: ", input_names, ", Output Node: ", output_names)

            try:
                flag = False
                temp = dic[m][0]
                onnx.utils.extract_model(onnx_path, save_dir_path + file.split(".yaml")[0] + '.onnx',
                                         input_names, output_names)

            except onnx.onnx_cpp2py_export.checker.ValidationError as e:
                # print("[warning] validation warning during exporting.")
                pass


def read_onnx(onnx_path, extracted_onnx_path):
    if os.path.isdir(extracted_onnx_path):
        file_list = os.listdir(extracted_onnx_path)
    else:
        os.mkdir(extracted_onnx_path)
        file_list = os.listdir(extracted_onnx_path)

    file_list_onnx = [file for file in file_list if file.endswith(".onnx")]
    file_list_onnx.sort()

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    #####batch_size 임의로 지정 (23_08_02)#####
    if(input_shape[0]=="batch_size"):
    	input_shape[0]=1;

    data = np.ones(input_shape, dtype=float)
    data = data.astype(np.float32)

    # result = session.run([output_name], {input_name: data})
    result = session.run([output_name], {input_name: data})[0]
    sessions = []
    # ONNXRuntime functions

    for file in file_list_onnx:
        # If "spatial = 0" does not work for "BatchNormalization", change "spatial=1"
        # else comment this "if" condition
        md = onnx.load(extracted_onnx_path + file)
        for node in md.graph.node:
            if node.op_type == "BatchNormalization":
                for attr in node.attribute:
                    if attr.name == "spatial":
                        attr.i = 1
        onnx.save(md, extracted_onnx_path + file)

        sess = ort.InferenceSession(extracted_onnx_path + file, providers=['CPUExecutionProvider'])
        sessions.append(sess)

    result_data = []
    for sess in sessions:
        if sessions.index(sess) == 0:
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
            data1 = sess.run([output_name], {input_name: data})[0]
            #####result_data empty state 방지 (23_08_02)#####
            result_data = sess.run([output_name], {input_name: data})
        else:
            input_names = sess.get_inputs()
            output_name = sess.get_outputs()[0].name
            if len(input_names) == 1:
                result_data = sess.run([output_name], {input_names[0].name: data})[0]
            elif len(input_names) == 2:
                result_data = sess.run([output_name], {input_names[0].name: data,
                                                       input_names[1].name: data1})[0]
            elif len(input_names) == 3:
                result_data = sess.run([output_name], {input_names[0].name: data,
                                                       input_names[1].name: data1,
                                                       input_names[2].name: result_data})[0]

    return result, result_data


def compareValue_func(result, result2):
    # 10의 -6승까지 비교
    return np.count_nonzero((np.abs(result[0] - result2[0]) > 0.000001).astype(int))


def main(args):
    # Extract ONNX model
    extract_model(args.onnx_file, args.yaml_dir, args.save_dir)

    # Read ONNX models
    res1, res2 = read_onnx(args.onnx_file, args.save_dir)
    extractedOnnx = os.listdir(args.save_dir)

    print('\nThe original onnx result and the extracted onnx {0} result is [{1}].'
          .format(extractedOnnx, np.count_nonzero((np.abs(res1 - res2) > 0.00001).astype(int)) == 0))


if __name__ == '__main__':
    args = parse_args()
    main(args)


# ------------------------------------------------------------------------------------------
# Function Test
# ------------------------------------------------------------------------------------------

def test_compareValue():
    res1, res2 = read_onnx(args.onnx_file, args.save_dir)
    assert compareValue_func(res1, res2) == 0
