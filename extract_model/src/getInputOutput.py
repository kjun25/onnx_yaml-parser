import onnx

model = onnx.load('./bat_test1111.onnx')
output =[node.name for node in model.graph.output]

input_all = [node.name for node in model.graph.input]
input_initializer =  [node.name for node in model.graph.initializer]
net_feed_input = list(set(input_all)  - set(input_initializer))

print(input_all)
print(input_initializer)
print('Inputs: ', net_feed_input)
print('Outputs: ', output)