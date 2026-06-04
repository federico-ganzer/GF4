from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

from week3.two_view_utils import (
    ThirdViewResult,
    TwoViewResult,
    build_2d3d_correspondences,
    compute_depths,
    compute_reprojection_errors,
    draw_reprojection_overlay,
    draw_single_image_reprojection_overlay,
    ensure_dir,
    estimate_essential_matrix,
    estimate_camera_pose_pnp,
    filter_reconstructed_points,
    make_camera_matrix,
    plot_multi_view_reconstruction,
    plot_patch_cloud_reconstruction,
    plot_two_view_reconstruction,
    recover_relative_pose,
    sample_point_colours,
    save_csv,
    triangulate_points,
    write_ply,
)


'''
1. Compute RANSAC Inlier Graph
2. Start with best pair - This becomes your anchor-set
3. Choose image in unused-set with most ransac inliers to a single image in the anchor-set
4. Perform 2d_3d_correspondence between unused image and anchor pair
5. image gets added to anchor set
6. Redo from step 3
'''

