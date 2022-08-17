import numpy as np
import onnxruntime as ort

# test ONNXRuntime functions
sess = ort.InferenceSession('F:/kjYun/2022/07_onnx/mxnet_exported_resnet18.onnx',  providers=['CPUExecutionProvider'])
sess1 = ort.InferenceSession('F:/kjYun/2022/07_onnx/test1.onnx',  providers=['CPUExecutionProvider'])
sess2 = ort.InferenceSession('F:/kjYun/2022/07_onnx/test2.onnx',  providers=['CPUExecutionProvider'])

input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
input_type = sess.get_inputs()[0].type
output_name = sess.get_outputs()[0].name

input_name1 = sess1.get_inputs()[0].name
input_shape1 = sess1.get_inputs()[0].shape
input_type1 = sess1.get_inputs()[0].type
output_name1 = sess1.get_outputs()[0].name

input_name2 = sess2.get_inputs()[0].name
input_shape2 = sess2.get_inputs()[0].shape
input_type2 = sess2.get_inputs()[0].type
output_name2 = sess2.get_outputs()[0].name

x = np.empty(input_shape, dtype=float)
x = x.astype(np.float32)

res = sess.run([output_name], {input_name: x})
print("[mxnet_exported_resnet18] input_name: " + input_name)
print("[mxnet_exported_resnet18] output_name: " + output_name)
# print(res)

x1 = np.empty(input_shape1, dtype=float)
x1 = x1.astype(np.float32)

res1 = sess1.run([output_name1], {input_name1: x1})
print("[divided_resnet18_part1] input_name: " + input_name1)
print("[divided_resnet18_part1] output_name: " + output_name1)

res2 = sess2.run([output_name2], {input_name2: res1[0]})
print("[divided_resnet18_part2] input_name: " + input_name2)
print("[divided_resnet18_part2] output_name: " + output_name2)
# print(res2)

result = np.array_equal(res[0], res2[0])

print(result)
