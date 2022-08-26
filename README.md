# onnx_yaml-parser

Usage

```
python onnx_parser.py \
       --onnx_file =<original_onnx_path>
       --yaml_dir =<yaml_dir>
       --save_dir =<save_extract_model_dir>
       --debug=(true|false)

/*
Usage example
*/
python onnx_parser.py --onnx_file ./org/mxnet_exported_resnet18.onnx --yaml_dir ./yaml/partition1/ --save_dir ./model/1/ --debug

```
Structure
```
yamlProject  (package)
        |
        |------- onnx_parser.py
        |
        |------- org/
        |           |
        |           |----- mxnet_exported_resnet18.onnx
        |
        |------- yaml/
        |           |
        |           |----- partition1/
        |           |      |----- partition1_1.yaml
        |           |      |----- partition1_2.yaml
        |           |     
        |           |----- partition2/
        |           |      |----- partition2_1.yaml
        |           |      |----- partition2_2.yaml
        |           |     
        |------- model/
        |           |
        |           |----- 1/
        |           |      |
        |           |      |----- partition1_1.onnx
        |           |      |----- partition1_2.onnx
        |           |  
        |           |----- 2/
        |           |      |
        |           |      |----- partition2_1.onnx
        |           |      |----- partition2_2.onnx
        
```
