import cv2
import numpy as np
from tensorflow.contrib.predictor.saved_model_predictor import SavedModelPredictor



class ExportedModel:
    """
    This is an easy-to-use API for predicting the output based on an input
    """

    def __init__(self, filename, image_size, num_sample_points):
        """
        Args:
        ----
        :param filename: Filename of the exported model
        :param image_size: Image size on which the model was trained
        """
        self.__image_size = image_size
        self.__predictor = self.load_exported_model(filename)
        self.n = num_sample_points

    def load_exported_model(self, filename):
        return SavedModelPredictor(filename)

    def predict(self, depth_map, point_queries):
        """
        This method takes NHWC input and then performs the prediction using the trained model
        ----
        Args:
        ----
        rgb_input: input to the trained model
        ----
        Return:
        ----
        rgb_input: after being resized to the specified size
        predictions: predicted classes [0,1,2,3, etc.]
        predictions_decoded: predictions with their colors decoded. See label_colours in image_util.py.
        """

        depth_map = cv2.resize(depth_map, (self.__image_size[1], self.__image_size[0]), interpolation=cv2.INTER_LINEAR)
        depth_map = np.expand_dims(depth_map, axis=0)
        depth_map = np.expand_dims(depth_map, axis=3)

        num_point_to_query = point_queries.shape[1]
        out_sdf = np.zeros((1, num_point_to_query, 1))
        point_left = point_queries.shape[1]

        if point_queries.shape[1] > self.n:
            i = 0
            while point_left > self.n:
                predictions = self.__predictor({'depth_map': depth_map,
                                                'points': point_queries[:, i*self.n: (i+1)*self.n, :]})
                out_sdf[:, i*self.n: (i+1)*self.n, :] = predictions['sdf']
                i = i + 1
                point_left = point_left - self.n
            res = self.n - point_left
            dummy_in = np.zeros((1, res, 3))
            predictions = self.__predictor({'depth_map': depth_map,
                                            'points': np.concatenate((point_queries[:, (num_point_to_query-point_left):, :],
                                                                      dummy_in), axis=1)})
            out_sdf[:, (num_point_to_query-point_left):, :] = predictions['sdf'][:, :point_left, :]

        return out_sdf, predictions['scale'], predictions['quaternion']
