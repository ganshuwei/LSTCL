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
