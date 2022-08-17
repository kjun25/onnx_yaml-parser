import numpy as np
import onnxruntime as ort

# ONNXRuntime functions
session = ort.InferenceSession('F:/kjYun/2022/07_onnx/mxnet_exported_resnet18.onnx',  providers=['CPUExecutionProvider'])
session_div1 = ort.InferenceSession('F:/kjYun/2022/07_onnx/test1.onnx',  providers=['CPUExecutionProvider'])
session_div2 = ort.InferenceSession('F:/kjYun/2022/07_onnx/test2.onnx',  providers=['CPUExecutionProvider'])

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

def compare_func(result, result2):
    # 10의 -6승까지 비교
    return np.count_nonzero((np.abs(result[0]-result2[0])>0.000001).astype(int))

def compare_full_func(result, result2):
    # 전체 값 비교
    return np.count_nonzero(result[0] - result2[0])

# ------------------------------------------------------------------------------------------
# Function Test
# ------------------------------------------------------------------------------------------

def test_compare():
    res1, res2 = extract_func(session, session_div1, session_div2)
    assert compare_func(res1, res2) == 0

def test_compare_part():
    res1, res2 = extract_part_func(session, session_div1)
    assert compare_func(res1, res2) == 0
    assert compare_full_func(res1, res2) == 0
