# Week 4 Incremental Reconstruction Pipeline

This script runs an incremental sparse 3D reconstruction pipeline from a sequence of images. It loads helper code from the Week 2 and Week 3 folders, extracts image features, builds a pairwise match graph, initializes reconstruction from a strong image pair, then incrementally registers new images and triangulates 3D points and plots them. 

## Requirements

Make sure the following are available in your Python environment:

- Python 3
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Open3D

The script also expects these local modules/files to exist:

- `week2/sfm_utils.py`
- `week3/two_view_utils.py`
- `ba_utils.py`
- `re_utils.py`

## How to run

You must provide either:

- `--image-dir <folder>` to use all supported images in a directory, or
- `--images <img1> <img2> ...` to list images manually

You must also provide:

- `--output-dir <folder>`

### Example: run from an image directory

```bash
python week4_pipeline.py \
  --image-dir path/to/images \
  --output-dir output
```

### Example: run with explicit image paths

```bash
python week4_pipeline.py \
  --images img1.jpg img2.jpg img3.jpg img4.jpg \
  --output-dir output
```

## Arguments

- `--week2-dir PATH` : path to the Week 2 folder containing `sfm_utils.py`
- `--week3-dir PATH` : path to the Week 3 folder containing `two_view_utils.py`
- `--max-image-size INT` : resize images so the long edge is at most this value
- `--max-features INT` : maximum number of SIFT features per image
- `--ratio FLOAT` : Lowe ratio-test threshold for descriptor matching
- `--focal-length-px FLOAT` : manually set focal length in pixels
- `--principal-point CX CY` : manually set the principal point
- `--ransac-threshold FLOAT` : essential matrix RANSAC threshold
- `--confidence FLOAT` : RANSAC confidence
- `--max-reprojection-error FLOAT` : reprojection threshold for keeping triangulated points
- `--pnp-ransac-threshold FLOAT` : PnP RANSAC reprojection threshold
- `--min-pnp-inliers INT` : minimum PnP inliers required to accept a new image
- `--draw-graph` : save a visualization of the pairwise match graph
- `--bundle-adjustment` : enable bundle adjustment

### Example with more controls

```bash
python week4_pipeline.py \
  --image-dir path/to/images \
  --output-dir output \
  --max-features 4000 \
  --max-image-size 1600 \
  --ratio 0.775 \
  --ransac-threshold 1.0 \
  --pnp-ransac-threshold 6.0 \
  --min-pnp-inliers 20 \
  --bundle-adjustment \
```

## Output

The script writes results to the output directory. Depending on the run settings and code path, this can include:

- `reconstruction.ply`
- `pairwise_match_graph.png`
- CSV metric files such as incremental reconstruction statistics

A Live Open3D visualization window is also opened during the reconstruction.

