# SOAR: Surgical Outcomes and Adverse event Recognition

This is the official code repository for the manuscript:

> **Computer vision-based retrieval of steps and intraoperative adverse events in laparoscopic cholecystectomy**
> Elaine Sui, Charlotte Egeland, Xiaohan Wang, Alfred Song, Rui Li, Joshua Villarreal, Anita Rau, Josiah Aklilu, Alan Brown, Shelly Goel, Brian Sutjiadi, Roger Bohn, Eric Sorenson, Vanessa Palter, Teodor Grantcharov, Jeffrey Jopling, Serena Yeung-Levy
> *npj Digital Surgery*, 2026. [https://doi.org/10.1038/s44484-026-00010-w](https://doi.org/10.1038/s44484-026-00010-w)

The repository contains two components used to analyze laparoscopic cholecystectomy video:

- [**SOAR-step-model**](SOAR-step-model/README.md) — temporal segmentation of surgical steps and active-surgery periods. See the [subdirectory README](SOAR-step-model/README.md) for environment setup, training, and inference instructions, and [SOAR-step-model/extract_features](SOAR-step-model/extract_features/README.md) for the frame-feature extraction tool it depends on.
- [**SOAR-error-model**](SOAR-error-model/README.md) — detection of intraoperative adverse events (bleeding, bile spillage, and thermal injury). See the [subdirectory README](SOAR-error-model/README.md) for environment setup, training, and inference instructions.

Each subdirectory is self-contained, with its own conda environment(s), training scripts, and pre-trained model checkpoints (linked from within each README).

## Citation

If you use this code, please cite:

```bibtex
@article{sui2026computer,
  title   = {Computer vision-based retrieval of steps and intraoperative adverse events in laparoscopic cholecystectomy},
  author  = {Sui, Elaine and Egeland, Charlotte and Wang, Xiaohan and Song, Alfred and Li, Rui and Villarreal, Joshua and Rau, Anita and Aklilu, Josiah and Brown, Alan and Goel, Shelly and Sutjiadi, Brian and Bohn, Roger and Sorenson, Eric and Palter, Vanessa and Grantcharov, Teodor and Jopling, Jeffrey and Yeung-Levy, Serena},
  journal = {npj Digital Surgery},
  volume  = {1},
  number  = {1},
  pages   = {12},
  year    = {2026},
  doi     = {10.1038/s44484-026-00010-w},
  url     = {https://doi.org/10.1038/s44484-026-00010-w}
}
```