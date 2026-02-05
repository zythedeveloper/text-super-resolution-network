# ICPR 2026 Low-Resolution License Plate Recognition (LRLPR)

[![Competition](https://img.shields.io/badge/Competition-ICPR%202026-blue)](https://icpr26lrlpr.github.io/)
[![Platform](https://img.shields.io/badge/Platform-Codabench-orange)](https://www.codabench.org/competitions/12259/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository contains the source code and documentation for my submission to the **ICPR 2026 Competition on Low-Resolution License Plate Recognition**.

## 📖 Overview

License plate recognition (LPR) in surveillance scenarios is often hampered by low resolution, heavy compression, and motion blur. This competition focuses on recovering and recognizing text from **Low-Resolution (LR)** license plate images, challenging participants to use techniques like Super-Resolution (SR), temporal modeling, and robust OCR.

**Official Website:** [https://icpr26lrlpr.github.io/](https://icpr26lrlpr.github.io/)

## 📂 Dataset Structure

The dataset consists of "tracks," where each track contains a sequence of images of the same license plate.

* **Input:** 5 consecutive **Low-Resolution (LR)** images per track.
* **Target:** The license plate text string.
* **Training Aid:** Corresponding **High-Resolution (HR)** images are provided for training tracks (to enable Super-Resolution approaches).

```text
data/
├── train/
│   ├── track_00001/
│   │   ├── lr_0.jpg, ..., lr_4.jpg  (Input)
│   │   └── hr_0.jpg, ..., hr_4.jpg  (Ground Truth High-Res)
│   └── ...
├── test_public/
│   ├── track_00101/
│   │   └── lr_0.jpg, ..., lr_4.jpg
│   └── ...
```

## 🛠️ Installation
```bash
# Clone the repository
git clone [https://github.com/zythedeveloper/icpr26-lrlpr.git](https://github.com/zythedeveloper/icpr26-lrlpr.git)
cd icpr26-lrlpr

# Create virtual environment
conda create -n myenv python=3.10.19 -y
conda activate myenv

# Install dependencies
pip install uv
uv python pin 3.10.19
uv sync
```

## 🗓️ Timeline
- Training Set Release: Dec 18, 2025
- Public Test Set (Leaderboard): Jan 19, 2026
- Blind Test Set Release: Feb 25, 2026
- Final Submission Deadline: Mar 1, 2026


## 📜 Citation
If you use this code or dataset, please cite the competition organizers:
```
@misc{icpr2026lrlpr,
  title={ICPR 2026 Competition on Low-Resolution License Plate Recognition},
  author={Laroca, Rayson and Nascimento, Valfride and Menotti, David},
  year={2026},
  publisher={ICPR},
  url={[https://icpr26lrlpr.github.io/](https://icpr26lrlpr.github.io/)}
}
```