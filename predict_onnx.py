import argparse
import os

import cv2

import predict


def parse_args():
    parser = argparse.ArgumentParser(description="PaDiM ONNX 单图推理")
    parser.add_argument("--image_path", default="leather/test/glue/010.png")
    parser.add_argument("--output_path", default="output/padim_result.png")
    return parser.parse_args()


def main():
    args = parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        print("图片读取失败:", args.image_path)
        return

    info = predict.predict_image(image)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(args.output_path, info["result"])
    print("结果图片已保存:", args.output_path)


if __name__ == "__main__":
    main()
