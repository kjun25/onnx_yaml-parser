import cv2
import numpy as np
import onnxruntime as ort
import argparse


#Usage 
#--onnx ./org/resnet50-v2-7.onnx --image ./org/cat_285.png
def parse_args():
    parser = argparse.ArgumentParser(description='Inference in onnx model')

    parser.add_argument('--debug', action='store_true', help='Debugging mode')

    parser.add_argument(
        '--onnx',
        dest='onnx',
        help='The onnx model path.',
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        '--image',
        dest='image',
        help='The image file path.',
        type=str,
        default=None,
        required=True
    )
    return parser.parse_args()


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def main(args):
    img = cv2.imread(args.image).astype(np.float32)
    print("original Image.shape: ", img.shape)

    # [height, width, channel] → [channel, height, width]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    print("transposed Image.shape: ", img.shape)

    session = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    out = session.run([output_name], {input_name: img})[0]

    print("the argmax is \"%s\" in ONNXRuntime" % np.argmax(out))
    print("the softmax is \"%s\" in ONNXRuntime" % softmax(out))


if __name__ == '__main__':
    args = parse_args()
    main(args)
