import onnx

input_path = 'F:/kjYun/2022/07_onnx/mxnet_exported_resnet18.onnx'
output_path = 'F:/kjYun/2022/07_onnx/testCheck.onnx'
input_names = ['data']
output_names = ['resnetv10_relu0_fwd']

onnx.utils.extract_model(input_path, output_path, input_names, output_names)
