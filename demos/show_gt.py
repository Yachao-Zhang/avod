import os
import sys
import time

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects

from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.obj_detection import evaluation
from wavedata.tools.visualization import vis_utils

import avod
from avod.builders.dataset_builder import DatasetBuilder
from avod.core import box_3d_encoder
from avod.core import box_3d_projector
from avod.core import anchor_projector


BOX_COLOUR_SCHEME = {
    'Car': '#00FF00',           # Green
    'Pedestrian': '#00FFFF',    # Teal
    'Cyclist': '#FFFF00'        # Yellow
}

def main():
    """This demo shows RPN proposals and AVOD predictions in 3D
    and 2D in image space. Given certain thresholds for proposals
    and predictions, it selects and draws the bounding boxes on
    the image sample. It goes through the entire proposal and
    prediction samples for the given dataset split.

    The proposals, overlaid, and prediction images can be toggled on or off
    separately in the options section.
    The prediction score and IoU with ground truth can be toggled on or off
    as well, shown as (score, IoU) above the detection.
    """
    
    fig_size = (10, 6.1)
    gt_classes = ['Car', 'Pedestrian', 'Cyclist']

    # Output images directories
    output_dir_base = 'images_2d'
    data_dir = '../../DATA/Kitti/object/'
    label_dir = data_dir + 'training/label_2'
    image_dir = data_dir + 'training/image_2'
    filepath = data_dir + 'val.txt'
    calib_dir = data_dir + 'training/calib'

    filenames = open(filepath, 'r').readlines()
    filenames = [int(filename) for filename in filenames]

    i = 0
    i_max = len(filenames)

    for filename in filenames:
        ##############################
        # Ground Truth
        ##############################

        # Get ground truth labels
        gt_objects = obj_utils.read_labels(label_dir, filename)

        boxes2d, _, _ = obj_utils.build_bbs_from_objects(
            gt_objects, class_needed=gt_classes)

        image_path = image_dir + "/%06d.png" % filename
        image = Image.open(image_path)
        image_size = image.size

        prop_fig, prop_2d_axes, prop_3d_axes = \
                vis_utils.visualization(image_dir,
                                        filename,
                                        display=False)

        # Read the stereo calibration matrix for visualization
        stereo_calib = calib_utils.read_calibration(calib_dir, filename)
        calib_p2 = stereo_calib.p2

        draw_gt(gt_objects, prop_2d_axes, prop_3d_axes, calib_p2)

        out_name = output_dir_base + "/%06d.png" % filename
        plt.savefig(out_name)
        plt.close(prop_fig)

        i += 1
        print(str(i) + '/' + str(i_max))

    print('\nDone')


def draw_gt(gt_objects, prop_2d_axes, prop_3d_axes, p_matrix):
    # Draw filtered ground truth boxes
    for obj in gt_objects:
        # Draw 2D boxes
        vis_utils.draw_box_2d(prop_2d_axes, obj, test_mode=True, color_tm='r')

        # Draw 3D boxes
        vis_utils.draw_box_3d(prop_3d_axes, obj, p_matrix,
                              show_orientation=False,
                              color_table=['r', 'y', 'r', 'w'],
                              line_width=2,
                              double_line=False)

if __name__ == '__main__':
    main()
