# Pipeline Overview

This document outlines the pipeline stages available in the Stroke Transfer system. Each pipeline stage can be run individually or grouped using predefined pipeline groups.

---

## Data Preparation

Before running any pipeline stages, you must set up the required **volume data** for your target scene.

Use the following command from the `output/` directory:

```bash
python setup_scene.py RisingSmoke
```

This will:

- Create `output/RisingSmoke/`
- Create a symbolic link: `output/RisingSmoke/assets → ../assets/RisingSmoke/`
- Clone the external volume data (~4.2 GB) into: `assets/RisingSmoke/volume_data/`

All external volume repositories are defined in:

```
volume_data_repositories.json
```

After setting up the volume data, you can run the feature computation and subsequent pipeline stages.

---

## Pipeline List

| Pipeline Name                   | Description                                                             |
|---------------------------------|-------------------------------------------------------------------------|
| `p2_compute_feature_basis`      | Compute features and basis vectors from volume data (requires `volume_data`) |
| `p3_preprocess_feature_basis`   | Standardizes features and smooths basis vectors                         |
| `p4_annotation_interpolation`   | Interpolates annotations across frames                                  |
| `p5_regression`                 | Learns regression models for stroke attributes                          |
| `p6_relocation`                 | Relocates attribute fields spatially/temporally                         |
| `p7_transfer`                   | Transfers learned attributes to target frames                           |
| `p8_smoothing`                  | Applies temporal smoothing to orientation fields                        |
| `p9_gen_strokes`                | Generates stroke data from attributes                                   |
| `p10_render_strokes`            | Renders strokes using lighting and texture                              |
| `p10_render_strokes_pencil`     | Renders strokes in a pencil-like style (optional)                       |
| `p11_final_composite`           | Composites rendered strokes onto background                             |
| `aux_exemplar_frame_estimation` | Estimates best exemplar frames for learning (auxiliary)                 |
| `aux_transfer_region_labels`    | Transfers region labels from attribute maps (auxiliary)                 |

---

## Pipeline Groups

To simplify usage, multiple stages can be grouped. The following named groups can be passed to `--pipelines`:

| Group Name              | Includes Pipelines                                                                                                           |
|------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `exemplar_frame_estimation` | `p2_compute_feature_basis`, `p3_preprocess_feature_basis`, `aux_exemplar_frame_estimation`                                   |
| `prepare_feature_basis`       | `p2_compute_features`, `p3_preprocess_feature_basis`                                                                         |
| `regression_transfer`       | `p4_annotation_interpolation`, `p5_regression`, `p6_relocation`, `p7_transfer`, `p8_smoothing`, `aux_transfer_region_labels` |
| `stroke_rendering`          | `p9_gen_strokes`, `p10_render_strokes`, `p10_render_strokes_pencil`, `p11_final_composite`                                   |

---

## Usage Example

```bash
# Run only the regression and transfer stages
python run.py --scene_names RisingSmoke --pipelines regression_transfer

# Run a custom subset of stages
python run.py --scene_names RisingSmoke --pipelines p5_regression p7_transfer

# Specify a frame range (e.g., only frames 120 to 160, every 2 frames)
python run.py --scene_names RisingSmoke --pipelines regression_transfer \
              --frame_start 120 --frame_end 160 --frame_skip 2

# Enable plotting and verbose logging for debugging
python run.py --scene_names RisingSmoke --pipelines p5_regression \
              --plot --verbose

# List all available pipeline stages
python run.py --list-pipelines

# List available pipeline groups
python run.py --list-pipeline-groups
```

---

## Notes

- `p2_compute_feature_basis` requires that `assets/[scene_name]/volume_data/` is already prepared.
- Use `python setup_scene.py [scene_name]` to set up `volume_data` before running any pipeline.
- Auxiliary stages are not required for most users but support advanced workflows.
- You can run all stages by omitting the `--pipelines` option.
