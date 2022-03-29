 ![Language](https://img.shields.io/badge/language-python--3.7.12-blue) 

<!-- ABOUT THE PROJECT -->

## About The Project

Detecting and Recoginising faces have become an extremly important task in computer vision. 

### Dataset

We have created our own face dataset for the task of identity recoginition. Each class in the dataset contains ~600 images of one person, the class name being the name of the directory. 

To download the dataset run the following command:
```shell
pip install gdown
gdown https://drive.google.com/uc?id=1bnj8sdugWK-L40JemSTNvsML3cNuq-iG
```

Similarly, we have exploited the pretrained models from Yolo-v5. These prtrained models can also be downloaded using the following command:

```shell
gdown https://drive.google.com/uc?id=17V1uJSLp9KheB4t9MFdROaktgSieN6CU
```

This will create ```dataset.zip``` and ```weights.zip``` which should be extracted in the root folder of this project.

### Built With
This project was built with 

* python v3.7.12
* PyTorch v1.7
* The environment used for developing this project is available at [environment.yml](environment.yml).


<!-- GETTING STARTED -->

## Getting Started

Clone the repository into a local machine and enter the root directory of this project using:

```
git clone git@github.com:here-to-learn0/RealTimeFaceDetectionAndTracking.git
cd RealTimeFaceDetectionAndTracking/
```


### Prerequisites

Create a new conda environment and install all the libraries by running the following command

```shell
conda env create -f environment.yml
```

### Instructions to run



## Model overview

<!-- RESULTS -->

## Results

<!-- LICENSE -->

