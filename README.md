# 3d bridge segmentation

## Introduction

The objective of the present project is the automated creation of digital twins for existing bridge structures. To this end, the initial step involves semantic segmentation of the provided point clouds prior to subsequent processing. To that end, an automated pipeline has been developed for the purpose of generating annotated data, thereby circumventing the necessity of scanning.

## Data

In total, we offer 23 different scenes of bridges. For simplicity we focused on the following classes:

<center>

| ID | Class name | nr points |
| -------- | ------- | ------- |
| 0 | unlabeled | 628.371 |
| 1 | ground | 66.630.880 |
| 2 | high vegetation | 10.532.641 |
| 3 | abutment | 10.838.768 |
| 4 | superstructure | 42.497.680 |
| 5 | top surface | 48.302.362 |
| 6 | railing | 2.169.738 |
| 7 | traffic sign | 341.185 |
| 8 | pillar | 3.462.674 |

</center>

Two examples, colored according to their classes, are visualized below:

<img src="docs/bridge_synth_example_0.png" width="425"/> <img src="docs/bridge_synth_example_1.png" width="425"/> 

### Download

Data can be download via the [mobilithek](https://mobilithek.info/offers/829756627880919040).

## Citation

This data was created for the work presented in [3D bridge segmentation using semi-supervised domain adaptation](https://www.sciencedirect.com/science/article/pii/S0926580525000615). If you find our work useful in your research, please consider citing:

```
@article{KELLNER2025106021,
title = {3D bridge segmentation using semi-supervised domain adaptation},
journal = {Automation in Construction},
volume = {172},
pages = {106021},
year = {2025},
issn = {0926-5805},
doi = {https://doi.org/10.1016/j.autcon.2025.106021},
url = {https://www.sciencedirect.com/science/article/pii/S0926580525000615},
author = {Maximilian Kellner and Timothy König and Jan-Iwo Jäkel and Katharina Klemt-Albert and Alexander Reiterer},
keywords = {3D deep learning, Bridge segmentation, Point cloud generation, Point cloud processing, Domain adaptation},
}
```