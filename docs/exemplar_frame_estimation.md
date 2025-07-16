# Exemplar Frame Estimation

This document is part of the release documentation for the **Stroke Transfer for Participating Media** project.  
It provides exemplar frame selections and instructions for reproducing the frame estimation process.
## How to Run

You can run the estimation pipeline from the `output` directory:
```bash
python run.py --scene_names RisingSmoke --pipelines aux_exemplar_frame_estimation
```

This process generates a log file at `log/exemplar_frame_estimation.log`,
which records the selected frames and associated parameters in the following format:
```json
{
    "RisingSmoke": {
        "a": 0.02857142857142857,
        "lmbd": 53.99999999999999,
        "num_exemplars": 1,
        "learn_frames": [
            114
        ],
        "learn_frames_extra": [
            114,
            210
        ],
        "interpolation_frames": [
            210
        ]
    }
}
```

- `learn_frames` are the exemplar frames actually used in the regression and transfer processes. 
- `interpolation_frames` are the target frames used for evaluation or interpolation.
- For each frame in `learn_frames`, we provide:
  - Exemplar images: `assets/exemplar/exemplar_%03d.png`
  - Annotations: `assets/exemplar/annotation_%03d.json`


## Final Settings (Resolution: 128)

| Scene Name   | #Exemplars | Learn Frames       | Interpolation Frame |
|--------------|----------------|----------------------|----------------------|
| RisingSmoke  | 1              | [114]                | 210                  |
