# pipeline_settings.json Format

This document provides an overview of the `pipeline_settings.json` file, which defines all inputs, intermediate paths, and parameters for running the Stroke Transfer pipeline.

Due to its length and flexibility, we highlight only the essential sections and their purposes. Users should use example files as a reference and customize only necessary fields.

---

## Core Concepts

- **File Templates**: Most fields define file path templates using `%03d` for frame indices.
- **temp/** and **final/**: Temporary files (intermediate computations) are placed under `temp/`, while final outputs go to `final/`.
- **Per-scene Settings**: Each `pipeline_settings.json` is specific to a scene.

---

## Key Sections

### `scene_name`
- Name of the scene, e.g., `"RisingSmoke"`

### `exemplar_file_templates`
- Paths to input images, annotations, and visualizations used for training.

### `raw_feature_file_templates`
- Volume-derived raw feature maps (e.g., curvature, velocity, transmittance).
- Includes `temperature` if available for the scene.

### `feature_file_templates`
- Processed features after standardization, used for learning and transfer.

### `basis_file_templates` / `basis_smooth_file_templates`
- Directional basis vectors for strokes; smoothed versions are used for stability.

### `model_files`
- Output paths for trained attribute models: orientation, color, width, and length.

### `attribute_file_templates`
- Transferred attributes for each frame, e.g., `orientation`, `color`, `width`, `length`.

### `stroke_file_templates` / `video_stroke_files`
- Output for generated strokes, rendered images, and videos.

### `frame_settings`

Defines the frame range and key subsets used in the pipeline: 

- `learn_frames`: Frame indices used for training
- `start`, `end`, `skip`: Frame range and sampling interval for transfer/stroke rendering
  

### `resolution`
- [width, height] of the feature/basis images

### `*_parameters`
- Parameter dictionaries for specific stages:
  - `standardization_parameters`
  - `exemplar_frame_estimation_parameters`
  - `regression_parameters`
  - `smoothing_parameters`
  - `relocation_parameters`
  - `stroke_rendering_parameters`
  - `render_strokes_parameters`

---

## Notes

- Unused fields (e.g., `temperature: null`) can be ignored.
- Paths are relative to `output/[scene_name]/`.

---

For an annotated example, see: [`assets/RisingSmoke/pipeline_settings.json`](../assets/RisingSmoke/pipeline_settings.json)

