import numpy as np
import onnxruntime as ort

# test ONNXRuntime functions
session = ort.InferenceSession('F:/kjYun/2022/07_onnx/mxnet_exported_resnet18.onnx',  providers=['CPUExecutionProvider'])
session1 = ort.InferenceSession('F:/kjYun/2022/07_onnx/test1.onnx',  providers=['CPUExecutionProvider'])
session2 = ort.InferenceSession('F:/kjYun/2022/07_onnx/test2.onnx',  providers=['CPUExecutionProvider'])
session3 = ort.InferenceSession('F:/kjYun/2022/07_onnx/test3.onnx',  providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_name = session.get_outputs()[0].name

input_name1 = session1.get_inputs()[0].name
input_shape1 = session.get_inputs()[0].shape
output_name1 = session1.get_outputs()[0].name

input_name2 = session2.get_inputs()[0].name
input_shape2 = session2.get_inputs()[0].shape
output_name2 = session2.get_outputs()[0].name

input_name3 = session3.get_inputs()[0].name
input_shape3 = session3.get_inputs()[0].shape
output_name3 = session3.get_outputs()[0].name

x = np.ones(input_shape, dtype=float)
x = x.astype(np.float32)

result = session.run([output_name], {input_name: x})
print("[mxnet_exported_resnet18] input_name: " + input_name)
print("[mxnet_exported_resnet18] output_name: " + output_name)

x1 = np.ones(input_shape1, dtype=float)
x1 = x1.astype(np.float32)

result1 = session1.run([output_name1], {input_name1: x1})
print("[divided_resnet18_part1] input_name: " + input_name1)
print("[divided_resnet18_part1] output_name: " + output_name1)

result2 = session2.run([output_name2], {input_name2: result1[0]})
print("[divided_resnet18_part2] input_name: " + input_name2)
print("[divided_resnet18_part2] output_name: " + output_name2)

result3 = session3.run([output_name3], {input_name3: result2[0]})
print("[divided_resnet18_part3] input_name: " + input_name3)
print("[divided_resnet18_part3] output_name: " + output_name3)

result = np.array_equal(result, result3)

print(result)
