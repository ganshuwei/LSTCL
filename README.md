# LSTCL

## Introduction

## Installation
1. git clone this repo.
2. Create a virtual environment
```shell
> conda env create -f environment.yml
```
> * Environment: CUDA 12.1 / Python 3.8

## Data Preparation
We use the dataset [Cholec80](https://camma.unistra.fr/datasets/) and [AutoLaparo](https://autolaparo.github.io/).
1. Download raw videos.
2. Extract frames from videos at a rate of 1 frame per second.
```shell
> ffmpeg –i Your_Video_name.mp4 -vf fps=1 %d.png
```
3. Cut the black margin and resize frames.
```shell
> python cutmargin.py
```
> Please change source_path and save_path to your own directory.

4. Generate PKL
```shell
> python datasets/data_preprosses/generate_labels_ch80.py
> python datasets/data_preprosses/generate_labels_autolaparo.py
```

The final structure of data folder should be arranged as follows:
```
(root folder)
├── data
|  ├── cholec80
|  |  ├── cutMargin
|  |  |  ├── 1
|  |  |  ├── 2
|  |  |  ├── 3
|  |  |  ├── ......
|  |  |  ├── 80
|  |  ├── labels
|  |  |  |  ├── train
|  |  |  |  |  ├── 1pstrain.pickle
|  |  |  |  |  ├── ...
|  |  |  |  ├── test
|  |  |  |  |  ├── 1psval_test.pickle
|  |  |  |  |  ├── ...
|  ├── AutoLparo
      ......
```
