import os
import logging
import numpy as np
from utils.generic_util import parse_args
from inference_api import ExportedModel
from PIL import Image
import cv2
from volume_visualizer.application import *
import volume_visualizer.sdf_drawer
import volume_visualizer.voxel_drawer


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

    test_dm_path = "/home/qiulin/VAE_DeepSDF/example/9c7b2ed3770d1a6ea6fee8e2140acec9_r_450001.png"
    test_dm = load_image(test_dm_path, [args.image_size[1], args.image_size[0]])

    # construct point query matrix
    n = args.n_point_per_edge
    step = 1.0/n
    coordinates = np.zeros((n, n, n, 3))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                coordinates[i, j, k, :] = np.array([-0.5+step*i, -0.5+step*j, -0.5+step*j])
    coordinates = np.reshape(coordinates, (n*n*n, 3))

    # inference from the model
    print('start to inference')
    out_sdf, out_scale, out_quaternion = model.predict(test_dm, coordinates)
    out_sdf = out_sdf.squeeze()
    out_sdf = np.reshape(out_sdf, (n, n, n))
    #print(out_sdf)
    #np.save('./1a6f615e8b1b5ae4dbbc9440457e303e.npy', out_sdf)
    print('Success!')

    # call the sdf visualizer
    app = Application()
    #out_sdf = np.load('./example/model_sdf_128.npy').astype(np.float16)
    out_sdf = np.clip(out_sdf, -0.01, 0.01)

    print(np.max(-out_sdf))
    print(np.min(-out_sdf))
    out_vox = np.where(np.less_equal(np.abs(out_sdf), 0.03), np.ones_like(out_sdf), np.zeros_like(out_sdf))
    drawer = volume_visualizer.voxel_drawer.VoxelDrawer(0.02)
    xx, yy, zz = np.where(out_vox)
    vox_pos = np.stack([xx, yy, zz], axis=-1) * 0.02
    print(vox_pos.shape)
    drawer.set_data(vox_pos)
    # drawer = volume_visualizer.sdf_drawer.sdf_drawer_from_npy(-out_sdf)
    app.add_drawer(drawer)
    app.show()


if __name__ == '__main__':
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    main()


