"""
This script is used to export a graph for inference. It reads the same .json config file that was used previously for training.
Then, it outputs the exported timestamped graph to the same experiment directory.

Usage: python export_inference_graph.py --config [config_filename]
Example: python export_inference_graph.py --config config/train_config.json
"""
import os
import tensorflow as tf
from VAE_deepSDF import VAE_deepSDF_estimator_fn
from utils.generic_util import parse_args
import logging


def main():
    # This may provide some performance boost
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Read the arguments to get them from a JSON configuration file
    args = parse_args()

    # Call TFEstimator and pass the model function to it
    model = tf.estimator.Estimator(
        model_fn=VAE_deepSDF_estimator_fn,
        model_dir=args.experiment_dir,
        params={
            'experiment_dir': args.experiment_dir,
            'pretrained_model_dir': args.pretrained_model_dir,
            'num_sample_points': args.num_sample_points,
            'delta': args.delta,
            'initial_learning_rate': args.initial_learning_rate,
            'final_learning_rate': args.final_learning_rate,
            'learning_rate_power': args.learning_rate_decay_power,
            'num_iterations': None,
            'log_every': args.log_every,
            'data_format': args.data_format,
            'num_epochs': None,
            'tensorboard_update_every': None,
            'downsampling_factor': args.output_stride,
            'width_multiplier': args.width_multiplier,
            'weight_decay': args.weight_decay,
            'dropout_keep_prob': args.dropout_keep_prob,
            'batchnorm': args.enable_batchnorm,
            'batchnorm_decay': args.batchnorm_decay,
            'latent_dim': args.latent_dim,
            'batch_size': args.batch_size,
            'export': True
        })

    # Export the model

    def serving_input_receiver_fn():
        features = {'depth_map': tf.placeholder(tf.float32, [None, args.image_size[0], args.image_size[1]],
                                                name='depth_map_tensor'),
                    'normal_map': tf.placeholder(tf.float32, [None, args.image_size[0], args.image_size[1],
                                                 args.image_size[2]], name='normal_map_tensor'),
                    'foreground_map': tf.placeholder(tf.float32, [None, args.image_size[0], args.image_size[1]],
                                                     name='foreground_map_tensor'),
                    'points': tf.placeholder(tf.float32, [None, args.num_sample_points, 3], name='points')}
        receiver_tensors = features

        return tf.estimator.export.ServingInputReceiver(
            features=features,
            receiver_tensors=receiver_tensors)

    tf.logging.info("Exporting the model to {} ...".format(args.experiment_dir))
    model.export_savedmodel(args.experiment_dir, serving_input_receiver_fn)
    tf.logging.info("Exported successfully!")


if __name__ == '__main__':
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    main()
