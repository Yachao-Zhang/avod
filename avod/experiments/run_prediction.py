"""
This file is intended for running prediction on a single image. This is useful in an on-line case, where we don't want the overhead of
writing predictions to file, then reading them, and displaying them.
"""
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core.models.avod_model import AvodModel
import tensorflow as tf
import numpy as np
from avod.datasets.kitti.kitti_utils import KittiUtils
from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import obj_utils
from avod.core import constants
from avod.core.models.rpn_model import RpnModel
from avod.core import box_3d_encoder
from avod.core import anchor_projector
import os
from avod.core.anchor_generators import grid_anchor_3d_generator
from avod.core import anchor_filter
import cv2
from avod.core import trainer_utils
from avod.core import box_3d_projector
import matplotlib.pyplot as plt
import logging
from wavedata.tools.core.calib_utils import FrameCalibrationData
from wavedata.tools.visualization import vis_utils


class AvodInstance:
    """ This class contains a bunch of stolen code from different parts of the avod project.
        It borrows the bare minimum in order to rn inference in an on-line setting.
     """

    def __init__(self, experiment_config_path, planes_dir, calib_dir):
        model_config, _, eval_config, dataset_config = \
            config_builder.get_configs_from_pipeline_file(
                experiment_config_path, is_training=False)
        self.dataset_config = config_builder.proto_to_obj(dataset_config)
        dataset_config.data_split_dir = 'testing'
        # These two lines below are necessary for KittiUtils to function properly
        self.cluster_split = self.dataset_config.cluster_split
        self.config = dataset_config

        self.data_split = self.config.data_split
        self.name = self.config.name
        self.dataset_dir = os.path.expanduser(self.config.dataset_dir)
        self.has_labels = self.config.has_labels
        self.cluster_split = self.config.cluster_split
        self.classes = list(self.config.classes)
        self.num_classes = len(self.classes)
        self.num_clusters = np.asarray(self.config.num_clusters)
        self.model_config = config_builder.proto_to_obj(model_config)
        self.paths_config = self.model_config.paths_config
        self.checkpoint_dir = self.paths_config.checkpoint_dir
        # Build the dataset object
        self.dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                          use_defaults=False)
        self.dataset.train_val_test = "test"

        # Score threshold for drawing bounding boxes
        self.avod_score_threshold = 0.25

    def predict(self, image, point_cloud, frame_calib):
        with tf.Graph().as_default():
            model = AvodModel(self.model_config,
                              train_val_test='test', dataset=self.dataset)

            # The model should return a dictionary of predictions
            prediction_dict = model.build()
            self._saver = tf.train.Saver()
            feed_dict = model.create_pred_feed_dict(
                image, point_cloud, frame_calib)
            sess = self._create_sess()
            restored_checkpoint = self._restore_chkpt()
            self._saver.restore(sess, restored_checkpoint)
            predictions = sess.run(prediction_dict,
                                   feed_dict=feed_dict)

            box_rep = self.model_config.avod_config.avod_box_representation
            proposals_and_scores = \
                self.get_rpn_proposals_and_scores(predictions)
            predictions_and_scores = \
                self.get_avod_predicted_boxes_3d_and_scores(predictions,
                                                            box_rep)

            # print(proposals_and_scores)
            # print(predictions_and_scores)
            print("Attempting to draw predictions")
            # TODO: Write these detections to the image instead of giving these scores
            # np.savetxt("rpn_file_path", proposals_and_scores, fmt='%.3f')
            # np.savetxt("avod_file_path", predictions_and_scores, fmt='%.5f')
            self._draw_predictions(image, predictions_and_scores, frame_calib)

    def _draw_predictions(self, image, predictions_and_scores, frame_calib):
        """ This code draws the bounding boxes with score threshold above the given threshold onto the image.
            This code is borrowed in part from show_predictions_2d.py
        """
        prediction_boxes_3d = predictions_and_scores[:, 0:7]
        print("Number of predictions boxes 3d: ", len(prediction_boxes_3d))
        prediction_scores = predictions_and_scores[:, 7]
        prediction_class_indices = predictions_and_scores[:, 8]
        image_size = image.shape[:2]
        # process predictions only if we have any predictions left after
        # masking
        if len(prediction_boxes_3d) > 0:
            # Apply score mask
            avod_score_mask = prediction_scores >= self.avod_score_threshold
            prediction_boxes_3d = prediction_boxes_3d[avod_score_mask]
            prediction_scores = prediction_scores[avod_score_mask]
            prediction_class_indices = \
                prediction_class_indices[avod_score_mask]
        else:
            return

        image_filter = []
        final_boxes_2d = []
        for i in range(len(prediction_boxes_3d)):
            box_3d = prediction_boxes_3d[i, 0:7]
            img_box = box_3d_projector.project_to_image_space(
                box_3d, frame_calib.p2,
                truncate=True, image_size=image_size,
                discard_before_truncation=False)
            if img_box is not None:
                image_filter.append(True)
                final_boxes_2d.append(img_box)
            else:
                image_filter.append(False)
        final_boxes_2d = np.asarray(final_boxes_2d)
        final_prediction_boxes_3d = prediction_boxes_3d[image_filter]
        final_scores = prediction_scores[image_filter]
        final_class_indices = prediction_class_indices[image_filter]

        num_of_predictions = final_boxes_2d.shape[0]
        print("Drawing {} predictions".format(num_of_predictions))
        # Convert to objs
        final_prediction_objs = \
            [box_3d_encoder.box_3d_to_object_label(
                prediction, obj_type='Prediction')
                for prediction in final_prediction_boxes_3d]
        for (obj, score) in zip(final_prediction_objs, final_scores):
            obj.score = score
        # Overlay prediction boxes on image
        filtered_gt_objs = []
        draw_orientations_on_pred = True
        fig, ax = self._create_fig(image)
        # Plot the image
        ax.imshow(image)
        # Draw predictions over image
        draw_3d_predictions(filtered_gt_objs,
                            frame_calib.p2,
                            num_of_predictions,
                            final_prediction_objs,
                            final_class_indices,
                            final_boxes_2d,
                            ax,
                            draw_orientations_on_pred)

    def _create_fig(self, image):
        """ Creates fig and ax object """
        # Create the figure'
        fig, ax = plt.subplots(1, figsize=image.shape[:2], facecolor='black')
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0,
                            hspace=0.0, wspace=0.0)
        # Set axes settings
        ax.set_axis_off()
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        return fig, ax

    def _restore_chkpt(self):
         # Load the latest checkpoints available
        trainer_utils.load_checkpoints(self.checkpoint_dir,
                                       self._saver)

        num_checkpoints = len(self._saver.last_checkpoints)
        # Select the last checkpoint
        ckpt_idx = num_checkpoints - 1
        return self._saver.last_checkpoints[ckpt_idx]

    def _create_sess(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def _get_point_cloud(self, image_shape, pointcloud, frame_calib, min_intensity=None):
        im_size = [image_shape[1], image_shape[0]]

        x, y, z, i = pointcloud
        print("Shape of x, y, z, i: ", x.shape, y.shape, z.shape, i.shape)
        # Calculate the point cloud
        pts = np.vstack((x, y, z)).T
        pts = calib_utils.lidar_to_cam_frame(pts, frame_calib)

        # The given image is assumed to be a 2D image
        if not im_size:
            point_cloud = pts.T
            return point_cloud

        else:
            # Only keep points in front of camera (positive z)
            pts = pts[pts[:, 2] > 0]
            point_cloud = pts.T

            # Project to image frame
            point_in_im = calib_utils.project_to_image(
                point_cloud, p=frame_calib.p2).T

            # Filter based on the given image size
            image_filter = (point_in_im[:, 0] > 0) & \
                (point_in_im[:, 0] < im_size[0]) & \
                (point_in_im[:, 1] > 0) & \
                (point_in_im[:, 1] < im_size[1])

        if not min_intensity:
            point_cloud = pts[image_filter].T

        else:
            intensity_filter = i > min_intensity
            point_filter = np.logical_and(image_filter, intensity_filter)
            point_cloud = pts[point_filter].T
        return point_cloud

    def load_point_cloud(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as fid:
                data_array = np.fromfile(fid, np.single)

            xyzi = data_array.reshape(-1, 4)

            x = xyzi[:, 0]
            y = xyzi[:, 1]
            z = xyzi[:, 2]
            i = xyzi[:, 3]

            return x, y, z, i
        else:
            return []

    def get_rpn_proposals_and_scores(self, predictions):
        """Returns the proposals and scores stacked for saving to file.

        Args:
            predictions: A dictionary containing the model outputs.

        Returns:
            proposals_and_scores: A numpy array of shape (number_of_proposals,
                8), containing the rpn proposal boxes and scores.
        """

        top_anchors = predictions[RpnModel.PRED_TOP_ANCHORS]
        top_proposals = box_3d_encoder.anchors_to_box_3d(top_anchors)
        softmax_scores = predictions[RpnModel.PRED_TOP_OBJECTNESS_SOFTMAX]

        proposals_and_scores = np.column_stack((top_proposals,
                                                softmax_scores))

        return proposals_and_scores

    def get_avod_predicted_boxes_3d_and_scores(self, predictions,
                                               box_rep):
        """Returns the predictions and scores stacked for saving to file.

        Args:
            predictions: A dictionary containing the model outputs.
            box_rep: A string indicating the format of the 3D bounding
                boxes i.e. 'box_3d', 'box_8c' etc.

        Returns:
            predictions_and_scores: A numpy array of shape
                (number_of_predicted_boxes, 9), containing the final prediction
                boxes, orientations, scores, and types.
        """

        if box_rep == 'box_3d':
            # Convert anchors + orientation to box_3d
            final_pred_anchors = predictions[
                AvodModel.PRED_TOP_PREDICTION_ANCHORS]
            final_pred_orientations = predictions[
                AvodModel.PRED_TOP_ORIENTATIONS]

            final_pred_boxes_3d = box_3d_encoder.anchors_to_box_3d(
                final_pred_anchors, fix_lw=True)
            final_pred_boxes_3d[:, 6] = final_pred_orientations

        elif box_rep in ['box_8c', 'box_8co', 'box_4c']:
            # Predictions are in box_3d format already
            final_pred_boxes_3d = predictions[
                AvodModel.PRED_TOP_PREDICTION_BOXES_3D]

        elif box_rep == 'box_4ca':
            # boxes_3d from boxes_4c
            final_pred_boxes_3d = predictions[
                AvodModel.PRED_TOP_PREDICTION_BOXES_3D]

            # Predicted orientation from layers
            final_pred_orientations = predictions[
                AvodModel.PRED_TOP_ORIENTATIONS]

            # Calculate difference between box_3d and predicted angle
            ang_diff = final_pred_boxes_3d[:, 6] - final_pred_orientations

            # Wrap differences between -pi and pi
            two_pi = 2 * np.pi
            ang_diff[ang_diff < -np.pi] += two_pi
            ang_diff[ang_diff > np.pi] -= two_pi

            def swap_boxes_3d_lw(boxes_3d):
                boxes_3d_lengths = np.copy(boxes_3d[:, 3])
                boxes_3d[:, 3] = boxes_3d[:, 4]
                boxes_3d[:, 4] = boxes_3d_lengths
                return boxes_3d

            pi_0_25 = 0.25 * np.pi
            pi_0_50 = 0.50 * np.pi
            pi_0_75 = 0.75 * np.pi

            # Rotate 90 degrees if difference between pi/4 and 3/4 pi
            rot_pos_90_indices = np.logical_and(pi_0_25 < ang_diff,
                                                ang_diff < pi_0_75)
            final_pred_boxes_3d[rot_pos_90_indices] = \
                swap_boxes_3d_lw(final_pred_boxes_3d[rot_pos_90_indices])
            final_pred_boxes_3d[rot_pos_90_indices, 6] += pi_0_50

            # Rotate -90 degrees if difference between -pi/4 and -3/4 pi
            rot_neg_90_indices = np.logical_and(-pi_0_25 > ang_diff,
                                                ang_diff > -pi_0_75)
            final_pred_boxes_3d[rot_neg_90_indices] = \
                swap_boxes_3d_lw(final_pred_boxes_3d[rot_neg_90_indices])
            final_pred_boxes_3d[rot_neg_90_indices, 6] -= pi_0_50

            # Flip angles if abs difference if greater than or equal to 135
            # degrees
            swap_indices = np.abs(ang_diff) >= pi_0_75
            final_pred_boxes_3d[swap_indices, 6] += np.pi

            # Wrap to -pi, pi
            above_pi_indices = final_pred_boxes_3d[:, 6] > np.pi
            final_pred_boxes_3d[above_pi_indices, 6] -= two_pi

        else:
            raise NotImplementedError('Parse predictions not implemented for',
                                      box_rep)

        # Append score and class index (object type)
        final_pred_softmax = predictions[
            AvodModel.PRED_TOP_CLASSIFICATION_SOFTMAX]

        # Find max class score index
        not_bkg_scores = final_pred_softmax[:, 1:]
        final_pred_types = np.argmax(not_bkg_scores, axis=1)

        # Take max class score (ignoring background)
        final_pred_scores = np.array([])
        for pred_idx in range(len(final_pred_boxes_3d)):
            all_class_scores = not_bkg_scores[pred_idx]
            max_class_score = all_class_scores[final_pred_types[pred_idx]]
            final_pred_scores = np.append(final_pred_scores, max_class_score)

        # Stack into prediction format
        predictions_and_scores = np.column_stack(
            [final_pred_boxes_3d,
             final_pred_scores,
             final_pred_types])

        return predictions_and_scores


def draw_3d_predictions(filtered_gt_objs,
                        p_matrix,
                        predictions_to_show,
                        prediction_objs,
                        prediction_class,
                        final_boxes,
                        pred_3d_axes,
                        draw_orientations_on_pred):
    BOX_COLOUR_SCHEME = {
        'Car': '#00FF00',           # Green
        'Pedestrian': '#00FFFF',    # Teal
        'Cyclist': '#FFFF00'        # Yellow
    }
    gt_classes = ['Car']
    for pred_idx in range(predictions_to_show):
        pred_obj = prediction_objs[pred_idx]
        pred_class_idx = prediction_class[pred_idx]

        rgb_box_2d = final_boxes[pred_idx]

        box_x1 = rgb_box_2d[0]
        box_y1 = rgb_box_2d[1]

        # Draw 3D boxes
        box_cls = gt_classes[int(pred_class_idx)]

        vis_utils.draw_box_3d(pred_3d_axes, pred_obj, p_matrix,
                              show_orientation=draw_orientations_on_pred,
                              color_table=['#00FF00', 'y', 'r', 'w'],
                              line_width=2,
                              double_line=False,
                              box_color=BOX_COLOUR_SCHEME[box_cls])
    plt.savefig("testingpleasework.png")


def create_framecalib(from_pandora=True):
    # These values are collected from the ROS calibration matrix
    if from_pandora:
        frame_calib = FrameCalibrationData()
        p2 = [1275.28898946, 0.0, 622.0, 0.0, 0.0,
              725.783914414, 185.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        p2 = np.reshape(p2, (3, 4))
        tr_velodyne_to_cam = [0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0]
        tr_velodyne_to_cam = np.reshape(tr_velodyne_to_cam, (3, 4))
        r0 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        frame_calib.p2 = p2
        frame_calib.tr_velodyne_to_cam = tr_velodyne_to_cam
        frame_calib.r0_rect = np.reshape(r0, (3, 3))
        return frame_calib
    else:
        # This is the correct form if testing with kitti data
        return calib_utils.read_calibration("/notebooks/DATA/Kitti/object/testing/calib", 1)


if __name__ == "__main__":
    model = AvodInstance(
        experiment_config_path="avod/data/outputs/cars_max_density_8/cars_max_density_8.config",
        planes_dir="/notebooks/DATA/Kitti/object/training/planes",
        calib_dir="/notebooks/DATA/Kitti/object/testing/calib")

    frame_calib = create_framecalib(from_pandora=False)
    image = cv2.imread(
        "/notebooks/DATA/Kitti/object/testing/image_2/000001.png", 1)
    print("Shape of image: ", image.shape)
    pointcloud = model.load_point_cloud(
        "/notebooks/DATA/Kitti/object/testing/velodyne/000001.bin")
    pointcloud = model._get_point_cloud(image.shape, pointcloud, frame_calib)
    print("loaded pointcloud: ", pointcloud)

    model.predict(image, pointcloud, frame_calib)
