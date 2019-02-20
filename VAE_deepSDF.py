from os.path import join
import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError
from Encoder import Encoder
from Decoder import Decoder


class VAE_deepSDF:
    def __init__(self, input, labels, params, mode='train', scope='VAE_deepSDF'):

        if params['downsampling_factor'] not in [8, 16]:
            raise ValueError("Currently supported downsampling factors are 8 and 16")

        if mode not in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
            raise ValueError("Modes are \'train\', \'eval\', or \'infer\'")

        # parse the input
        self.depth_map_input = input['depth_map']  #input shape: [batch, 3, 255, 255]
        self.samples = input['points']  #[batch, K, 3]

        # parse the labels
        self.gt_scale = labels['scale']  #[batch, 1]
        self.gt_quaternion = labels['quaternion']  #[batch, 4]
        self.gt_sdf = labels['sdf']  #[batch, K, 1]

        # private attribute
        # parameter settings
        self.__NORMALIZATION_FACTOR = 255

        if mode != tf.estimator.ModeKeys.PREDICT:
            self.__num_epochs = params['num_epochs']
            self.__num_iterations = params['num_iterations']
        self.__scope = scope
        self.__object_per_batch = params['object_per_batch']
        self.__num_views = params['num_views']
        self.__num_sample_points = params['num_sample_points']
        # Delta controls the distance from the surface over which we expect to maintain a metric SDF
        self.__delta = params['delta']
        self.__initial_learning_rate = params['initial_learning_rate']
        self.__final_learning_rate = params['final_learning_rate']
        self.__learning_rate_power = params['learning_rate_power']
        # MobileNet parameter setting
        self.__downsampling_factor = params['downsampling_factor']
        self.__width_multiplier = params['width_multiplier']
        self.__weight_decay = params['weight_decay']
        self.__dropout_keep_prob = params['dropout_keep_prob']
        self.__batchnorm = params['batchnorm']
        self.__batchnorm_decay = params['batchnorm_decay']
        self.__mode = mode
        self.__export = params['export'] if 'export' in params else False
        self.__data_format = params['data_format']
        self.__is_training = True if self.__mode == 'train' else False

        # public attributes
        self.loss = None
        self.learning_rate = None
        self.global_step = None
        self.train_op = None
        self.metrics = None
        self.z_mu = None
        self.z_logvar = None
        self.z = None
        self.quaternion = None
        self.scale = None
        self.sdf_pred = None
        self.cliped_sdf_pred = None
        self.cliped_sdf_gt = None

# Perform architecture building..
        self.__network()
        self.__output()

    def __network(self):
        """
        Define the VAE network
        """
        # Network Construction begins here
        with tf.variable_scope(self.__scope):
            # encoder
            encoder = Encoder(self.depth_map_input/self.__NORMALIZATION_FACTOR, 3, self.__downsampling_factor,
                              self.__width_multiplier, self.__weight_decay,
                              self.__dropout_keep_prob, self.__batchnorm,
                              self.__batchnorm_decay, self.__is_training, self.__data_format)
            self.z_mu = encoder.shape_mu  # [batch, 1024]
            self.z_logvar = encoder.shape_logvar  #[batch, 1024]
            self.quaternion = encoder.quaternion  #[batch, 4]
            self.scale = encoder.scale  #[batch, 1]

            # reparameterization
            z_std = tf.exp(0.5*self.z_logvar)  #[batch, 1024]
            eps = tf.random_normal(shape=tf.shape(self.z_mu),
                                   mean=0, stddev=1, dtype=tf.float32) #[batch, 1024]
            self.z = z_std*eps + self.z_mu  #[batch, 1024]

            # decoder
            z = tf.reshape(self.z, shape=[-1, 1, self.z_mu.shape[1]])
            decoder_input = tf.tile(z, multiples=[1, self.__num_sample_points, 1])
            decoder_input = tf.concat([decoder_input, self.samples], axis=2)
            decoder_input = tf.reshape(decoder_input, shape=[-1, self.z_mu.shape[1]+3])
            decoder = Decoder(decoder_input)

            # output sdf values
            self.sdf_pred = tf.reshape(decoder.end_point, [-1, self.__num_sample_points, 1])

    def __output(self):

        if self.__mode == 'predict' or self.__export == True:
            return

        # Collect regularization losses
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # pose estimation loss is mean squared error loss
        pose_estimation_loss = tf.losses.mean_squared_error(self.quaternion, self.gt_quaternion) + tf.losses.mean_squared_error(self.scale, self.gt_scale)

        # view loss is defined as the variance of z_mu
        view_loss = 0
        for i in range(self.__object_per_batch):
            _, var_z_mu = tf.nn.moments(x=self.z_mu[i*self.__num_views:(i+1)*self.__num_views, :], axes=[0])  # [1024, ]
            view_loss += tf.reduce_sum(var_z_mu)
        view_loss = view_loss / (1024*self.__object_per_batch)

        # KL divergence for posterior and prior of the latent variable
        latent_var_constrains = (-0.5*tf.reduce_sum(1 + self.z_logvar - self.z_mu**2 - tf.exp(self.z_logvar))) / (1024*self.__object_per_batch*self.__num_views)

        # reconstruction loss is defined as in the DeepSDF paper
        self.cliped_sdf_pred = tf.clip_by_value(self.sdf_pred, -self.__delta, self.__delta)
        self.cliped_sdf_gt = tf.clip_by_value(self.gt_sdf, -self.__delta, self.__delta)
        reconstruction_loss = (tf.losses.absolute_difference(self.cliped_sdf_gt, self.cliped_sdf_pred))

        # try different factors here to train the model
        pose_estimation_loss = pose_estimation_loss * 20
        view_loss = view_loss * 5
        latent_var_constrains = latent_var_constrains * 5
        reconstruction_loss = reconstruction_loss * 1000

        self.loss = pose_estimation_loss + view_loss + latent_var_constrains + reconstruction_loss + \
                    self.__weight_decay * regularization_loss

        self.global_step = tf.train.get_or_create_global_step()

        # Create a learning rate tensor which supports decaying
        self.learning_rate = tf.train.polynomial_decay(self.__initial_learning_rate,
                                                       tf.cast(self.global_step, tf.int32),
                                                       self.__num_iterations * self.__num_epochs,
                                                       self.__final_learning_rate, power=self.__learning_rate_power)

        # Create Adam optimizer [This optimizer can be changed according to the needs]
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss, self.global_step)

        # Construct extra needed metrics for training and validation
        self.metrics = {'pose_estimation_loss': pose_estimation_loss, 'view_loss': view_loss,
                        'reconstruction_loss': reconstruction_loss, 'KL_divergence': latent_var_constrains,
                        'regularization_loss': regularization_loss}


def VAE_deepSDF_estimator_fn(features, labels, mode, params):
    """
    This is the model function needed for TensorFlow Estimator API. ALL of its parameters are passed by the estimator automatically.
    ----
    Return:
    ----
    A TFEstimator spec either for training or evaluation
    """
    # Construct the whole network graph
    network_graph = VAE_deepSDF(features, labels, params, mode)

    predictions = {
        'sdf': network_graph.cliped_sdf_pred,
        'scale': network_graph.scale,
        'quaternion': network_graph.quaternion
    }

    # Do the following only if the TF estimator is in PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            # These are the predictions that are needed from the model
            predictions=predictions,
            # This is very important for TensorFlow serving API. It's the response from a TensorFlow server.
            # remember to generate two different models for segmentation and feature extract
            export_outputs={
                # if the outputs is not dict, or any of its keys are not strings, or any of its values are not Tensor
                # a ValueError will be raised
                'outputs': tf.estimator.export.PredictOutput(predictions),
            })

    # Restore variables from a pretrained model (with the same names) except those in the last layer.
    # This works only in training and in validation modes ONLY.
    try:
        if params['pretrained_model_dir'] != "":
            variables = tf.trainable_variables(scope='VAE_deepSDF')
            names = {v.name.split(":")[0]: v.name.split(":")[0] for v in variables}
            tf.train.init_from_checkpoint(params['pretrained_model_dir'], names)
    except NotFoundError:
        tf.logging.warning("No pretrained model directory exists. Skipping.")

    def create_summaries_and_logs():
        """
        Construct summaries and logs during training and evaluation
        ----
        Return:
        ----
        a logging hook object and a summary hook object
        """
        # Construct extra metrics for Training and Evaluation
        extra_summary_ops = [tf.summary.scalar('total_loss', network_graph.loss),
                             tf.summary.scalar('reconstruction_loss', network_graph.metrics['reconstruction_loss']),
                             tf.summary.scalar('view_loss', network_graph.metrics['view_loss']),
                             tf.summary.scalar('pose_estimation_loss', network_graph.metrics['pose_estimation_loss'])]

        # TFEstimator automatically creates a summary hook during training. So, no need to create one.
        if mode == tf.estimator.ModeKeys.TRAIN:
            extra_summary_ops.append(tf.summary.scalar('learning_rate', network_graph.learning_rate))

            # Construct tf.logging tensors
            train_tensors_to_log = {'epoch': network_graph.global_step // params['num_iterations'],
                                    'learning_rate': network_graph.learning_rate,
                                    'train_reconstruction_loss': network_graph.metrics['reconstruction_loss'],
                                    'train_view_loss': network_graph.metrics['view_loss'],
                                    'train_pose_estimation_loss': network_graph.metrics['pose_estimation_loss']}

            logging_hook = tf.train.LoggingTensorHook(tensors=train_tensors_to_log,
                                                      every_n_iter=params['log_every'])

            return [logging_hook]

        summary_output_dir = join(params['experiment_dir'], 'eval')

        # Construct tf.logging tensors
        val_tensors_to_log = {'epoch': network_graph.global_step // params['num_iterations'] - 1,
                              'global_step': network_graph.global_step,
                              'val_reconstruction_loss': network_graph.metrics['reconstruction_loss'],
                              'val_pose_estimation_loss': network_graph.metrics['pose_estimation_loss']}
        
        logging_hook = tf.train.LoggingTensorHook(tensors=val_tensors_to_log, every_n_iter=params['log_every'])

        summary_hook = tf.train.SummarySaverHook(params['tensorboard_update_every'], output_dir=summary_output_dir,
                                                 summary_op=tf.summary.merge(extra_summary_ops))

        return [logging_hook, summary_hook]

    hooks = create_summaries_and_logs()

    # Do the following only if the TF estimator is in TRAIN or EVAL modes
    # It computes the loss and optimizes it using the train_op.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=network_graph.loss,
        train_op=network_graph.train_op,
        training_hooks=hooks,
        evaluation_hooks=hooks)
