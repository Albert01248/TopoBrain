# Third-party sources

## CodeBrain

- Source: https://github.com/jingyingma01/CodeBrain
- Pinned commit: `22d350caf68246d2fda4f630ef837420db3fb130`.
- The upstream source contains the Apache License, Version 2.0. Preserve its
  license and applicable notices when separately redistributing upstream source.
- Tokenizer checkpoint: https://huggingface.co/YjMajy/CodeBrain
- Filename: `CodeBrain_Tokenizer.pth`.
- Expected SHA256: `e9560b670d64ea4712fd99a48dc2131326b919744c7d3eb504cf57b1ef3af999`.
- `scripts/fetch_assets.py` obtains these assets separately. Source and checkpoint
  binaries are excluded from this distribution. Review checkpoint-specific terms
  with the model distribution.

## FACED

- Dataset record: https://nemar.org/dataset/nm000112
- Versioned source: https://data.nemar.org/nm000112/v1.1.4
- Dataset description: https://www.nature.com/articles/s41597-023-02650-w
- The preparation script records version 1.1.4, URLs, byte ranges, checksums and
  CC-BY-4.0 metadata. Attribution and data terms apply to separately obtained data.
  EEG data and derived caches are not bundled.
- Electrode coordinates use MNE's `standard_1020` template.

## Optional Sleep-EDF loader

- Source: https://physionet.org/content/sleep-edfx/1.0.0/
- `prepare_sleepedf_sample.py` prepares an interface sample. Tests also use the
  annotation and signal-processing utilities in this module.
- Obtain data and review its terms separately from PhysioNet.

## Dependencies

PyTorch, torchvision, NumPy, SciPy, MNE-Python, timm and einops retain their own
licenses. No dependency binaries are included. This file records sources and
does not select a license for the project's original code.
