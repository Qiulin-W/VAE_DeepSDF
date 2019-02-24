import os
import logging
import numpy as np
from utils.generic_util import parse_args
from inference_api import ExportedModel
from PIL import Image
import cv2


def load_image(filename, output_size):
    """
    Read a depth map(.png) from a filename, change it to gray-scale image and resize it to the required output size

    :param filename: a string representing the file name

    :param output_size: a tuple [height, width]

    :return: Depth map after the above operations performed
    """
    img = Image.open(filename)
    img = np.array(img)
    img_resize = cv2.resize(img, (output_size[1], output_size[0]), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)

    return img_gray


def main():
    # This may provide some performance boost
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Read the arguments to get them from a JSON configuration file
    args = parse_args()

    model = ExportedModel(os.path.join(args.experiment_dir, args.test_model_timestamp_directory), args.image_size, args.num_sample_points)

    test_dm_path = "L:/ISO/deepSDF/VAE_DeepSDF/dataset/depth/1a6f615e8b1b5ae4dbbc9440457e303e_r_450001.png"
    test_dm = load_image(test_dm_path, [args.image_size[1], args.image_size[0]])

    # construct point query matrix
    n = args.n_point_per_edge
    coordinates = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            coordinates[i, j, :] = np.linspace(-0.5, 0.5, n)
    point_queries = np.zeros((1, n**3, 3))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                point_queries[:, i*n**2 + j*n + k, :] = coordinates[i, j, k]

    # inference from the model
    print('start to inference')
    out_sdf, out_scale, out_quaternion = model.predict(test_dm, point_queries)
    out_sdf = np.reshape(out_sdf, (n, n, n))
    np.save('./1a6f615e8b1b5ae4dbbc9440457e303e.npy', out_sdf)
    print('Successfully save the corresponding sdf numpy array')


if __name__ == '__main__':
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    main()


