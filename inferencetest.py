import cv2
import numpy as np
import onnxruntime as ort
import argparse


# Usage
# --onnx ./org/resnet50-v2-7.onnx --image ./org/cat_285.png
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

    # [height, width, channel] → [channel, height, width]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    print("input Image.shape: ", img.shape)

    session = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    out = session.run([output_name], {input_name: img})[0]
    result = softmax(out[0])
    print("sum of softmax value: ", sum(softmax(out[0])))
    sortedOutput = np.sort(result)[::-1]

    for i in range(5):
        string = 'Top{i}: softmax[{softmax}], index{index}'.format(i=i+1, softmax=round(sortedOutput[i], 3), index=np.where(result == sortedOutput[i])[0])
        print(string)


if __name__ == '__main__':
    args = parse_args()
    main(args)
