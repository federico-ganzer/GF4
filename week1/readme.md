# GF4 Structure from Motion — Week 1 Guide
## Seeing the Big Picture with COLMAP

**Theme:** Run a professional SfM/MVS pipeline end-to-end before implementing the internals yourself.  
**Main output this week:** A working COLMAP reconstruction, a short experimental comparison across captures, and a clear understanding of what SfM succeeds and fails on.

---

## 0. Week 1 learning objectives

By the end of Week 1, you should be able to:

1. Explain the difference between **sparse reconstruction** and **dense reconstruction**.
2. Run COLMAP on a small image collection using the GUI or command line.
3. Interpret the key COLMAP outputs:
   - registered images,
   - sparse points,
   - camera poses,
   - dense point cloud / mesh.
4. Design a good image capture for SfM.
5. Diagnose common failure modes such as insufficient overlap, blur, specular surfaces, textureless surfaces, repeated structures, and moving objects.
6. Produce a concise evidence-based comparison of different data captures.

---

## 1. Required software and provided materials

You will be given:

- A project GitHub repository.
- A small known-good image dataset.
- A `README.md` with installation instructions.
- A `hello_sift.py` script.
- A `colmap_quickstart.md` sheet.
- A `report_template_week1.md` template.
- Utility scripts for collecting COLMAP statistics where possible.

You should use **Python 3.10+** and either **conda** or **venv**.

Minimum Python packages:

```bash
opencv-python
numpy
matplotlib
scipy
open3d
pycolmap   # optional, if installation works
```

You must also install **COLMAP** with GUI support if possible.

---

## 2. Phase A — Environment setup and sanity checks

**Estimated time:** 3–5 hours  
**Do this first. Do not start capturing your own data before the sanity checks pass.**

### Tasks

1. Clone/download the project repository.
2. Create a clean Python environment.
3. Install Python dependencies.
4. Install COLMAP.
5. Run:

```bash
python hello_sift.py --image data/example/image01.jpg
```

6. Open COLMAP and verify that the GUI launches.
7. Run COLMAP on the provided known-good mini dataset.

### Required outputs

You should save the following in your Week 1 folder:

```text
week1/
  setup_notes.md
  hello_sift_output.png
  colmap_known_good_screenshot.png
  colmap_known_good_stats.txt
```

### Expected output

For the provided known-good dataset, you should normally see:

- most or all images registered,
- a visible sparse point cloud,
- camera frustums arranged around the object/scene,
- dense reconstruction possible, although not necessarily perfect.

Exact numbers may differ by machine and COLMAP version. You are not being marked on matching a particular number exactly.

### If it does not work

Use this debugging order:

1. Does COLMAP open?
2. Can COLMAP import the images?
3. Does feature extraction complete?
4. Does feature matching complete?
5. Does sparse reconstruction register at least two images?
6. Does dense reconstruction fail only after sparse reconstruction?

Record the failure point in `setup_notes.md`.

If you lose more than **90 minutes** on installation, ask a demonstrator or project leader. Do not silently spend a full day debugging installation.

---

## 3. Phase B — First full COLMAP reconstruction

**Estimated time:** 3–4 hours

Use the provided known-good dataset first.

### Tasks

1. Create a new COLMAP project.
2. Import the known-good image set.
3. Run feature extraction.
4. Run feature matching.
5. Run sparse reconstruction.
6. Inspect the sparse model.
7. Run dense reconstruction if your machine supports it.
8. Export screenshots.

### Required screenshots

Save:

```text
known_good_sparse.png
known_good_dense.png      # if dense reconstruction succeeds
known_good_cameras.png
```

### Questions to answer in your notes

1. How many images were registered?
2. Is the camera trajectory plausible?
3. Are the sparse points concentrated on the object/scene?
4. Which parts of the scene are missing?
5. Does the dense reconstruction improve the visual result compared with the sparse cloud?

---

## 4. Phase C — Systematic data acquisition experiment

**Estimated time:** 7–9 hours

You will capture your own image sets for a single object or small static scene.

Choose something with:

- good texture,
- non-reflective surfaces,
- stable lighting,
- no moving objects,
- enough 3D structure.

Good choices:

- a textured sculpture,
- a pile of books,
- a chair with textured surroundings,
- a building façade with distinctive features,
- an engineering object with labels/texture.

Bad first choices:

- a plain white mug,
- a shiny metal bottle,
- glass,
- a mirror,
- a blank wall,
- a screen,
- a moving person,
- an object on a rotating turntable unless explicitly analysed as a failure case.

### Capture protocol

Create three image sets of the **same object/scene**.

#### Set 1: Gold-standard capture

- 35–60 images.
- Approx. 70–85% overlap between neighbouring views.
- Move the camera around the object/scene.
- Keep the object/scene static.
- Avoid motion blur.
- Avoid large exposure changes.
- Avoid zoom changes.
- Use landscape orientation unless there is a good reason not to.

#### Set 2: Minimal capture

- 8–12 images.
- Same object/scene.
- Try to cover the scene with many fewer images.

#### Set 3: Challenging capture

Choose **one** challenge deliberately:

- low light,
- motion blur,
- shiny/specular surface,
- textureless surface,
- repeated pattern,
- low overlap,
- object moved between shots,
- camera mostly rotating in place with little translation.

Do not mix many challenges at once. The goal is to understand one failure mode clearly.

### Folder structure

Use:

```text
data/my_capture/
  gold/
  minimal/
  challenging/
```

---

## 5. Phase D — Run COLMAP on your three captures

**Estimated time:** 4–6 hours

Run COLMAP on all three image sets.

For each set, record:

| Quantity | Meaning |
|---|---|
| number of input images | How many images you provided |
| number of registered images | How many images COLMAP successfully used |
| number of sparse points | Approximate sparse reconstruction size |
| dense reconstruction success/failure | Whether dense stage produced usable output |
| qualitative quality | Good / partial / poor / failed |
| main observed failure | Your diagnosis |

### Required comparison table

Create a table like this:

| Dataset | Input images | Registered images | Sparse points | Dense result | Main observation |
|---|---:|---:|---:|---|---|
| Known-good | | | | | |
| Gold | | | | | |
| Minimal | | | | | |
| Challenging | | | | | |

Approximate values are acceptable if exact extraction is inconvenient, but be clear how you obtained them.

---

## 6. Phase E — Short theory reading

**Estimated time:** 2–3 hours

Read enough to answer the questions below. Suggested sources:

- Hartley and Zisserman, *Multiple View Geometry*, introductory material.
- COLMAP documentation.
- Szeliski, *Computer Vision: Algorithms and Applications*, chapters on feature matching and structure from motion.

### Concepts to understand

1. Sparse vs dense reconstruction.
2. Feature detection and description.
3. Feature matching.
4. Camera poses.
5. Triangulation.
6. Bundle adjustment.
7. Why SfM needs parallax.
8. Why pure rotation is degenerate for 3D reconstruction.

---

## 7. Week 1 deliverables

Submit or prepare the following.

### Group deliverables

1. Screenshots of:
   - known-good sparse reconstruction,
   - your gold-standard sparse/dense reconstruction,
   - your minimal reconstruction,
   - your challenging reconstruction or failed attempt.
2. A comparison table of the four datasets.
3. A short group log of major setup/capture decisions.

### Individual deliverable

A **1-page individual reflection** answering:

1. What made your best capture successful?
2. What caused your challenging capture to fail or degrade?
3. What is one thing COLMAP did that you do not yet understand mathematically?
4. What is one question you want to answer when building your own pipeline?

---

## 8. Minimum success criteria

By the end of Week 1, every group should have:

- successfully run COLMAP on the known-good dataset,
- captured at least one usable image set,
- obtained at least one sparse reconstruction from their own data,
- produced screenshots and a comparison table,
- identified at least one SfM failure mode.

Dense reconstruction is desirable but not required for minimum success if hardware prevents it.

---

## 9. Stretch goals

For groups that finish early:

1. Try COLMAP command-line reconstruction.
2. Compare SIFT vs another feature extractor if available.
3. Try a public benchmark dataset.
4. Export your sparse cloud or dense cloud as `.ply` and visualize it in Open3D or MeshLab.
5. Produce a short flythrough video of the reconstructed scene.

---

## 10. Common failure modes

### Failure: Very few images are registered

Likely causes:

- insufficient overlap,
- images too blurry,
- object/scene changed between images,
- repeated or textureless surfaces,
- too few images.

### Failure: Sparse cloud exists but looks flat or unstable

Likely causes:

- camera motion was mostly rotation,
- little baseline between images,
- scene mostly planar,
- bad calibration.

### Failure: Dense reconstruction fails

Likely causes:

- sparse reconstruction too weak,
- insufficient GPU/CPU resources,
- images too large,
- inconsistent lighting.

### Failure: Reconstruction contains duplicate surfaces or ghosting

Likely causes:

- moving objects,
- object moved during capture,
- reflective/translucent surfaces,
- inconsistent exposure.

---

## 11. What to bring to the Week 2 session

Bring:

1. Your three image sets.
2. Your Week 1 comparison table.
3. A pair of images from your gold-standard set with good overlap.
4. A pair of images from your challenging set.
5. Your questions about how COLMAP estimated camera motion and 3D points.
