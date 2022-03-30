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

## Adding own dataset


The main feature of this Face Classifier  is that it is adaptable to new data classes. If you want to adapt this model on your custom live dataset, you have to first do a few steps.

### Adding images
First, to classify your own face you first need to add your images to the dataset. We have made this simple. You can now run the script below which will take ~600 frames. 


```shell
python scripts/capture_video_4_dataset.py --name class-name
```

**Note**
1. Make sure there is just one person in the camera frame. 
2. Try to look around so that the dataset captures your face from all angles

### Getting embeddings 

These newly detected faces have to be converted to embeddings for training the classifier. This is done by running this:

```shell
python scripts/get_embeddings.py
```

This automatically detects the newest class added and creates embeddings and saves it in `dataset/embeddings/`


### Training classifier on this new dataset

For training the classifier (SVM) on this new dataset run this command:

```shell
python scripts/train_classifier.py
```

This makes a new classifier and stores it in the root directory of the project.Now this classifier can be used in classification of faces.



## Runnning face detector and tracker

There are two ways this tracker can be used. First, where we would like to track a person with the identity. This option is given below: 

```shell
python scripts/main.py --name name-of-person-to-track
```

Second, is the **Collective Tracker** where we are not interestedd in the identity of the person but we want to collectively track (keep) `n` number of people in the frame thus we find the mean coodinate that our slider camera system should chase. This can be run by using the command below:


```shell
python scripts/person_track_identity_agnostic.py --num_persons_to_track 2
```

Here the argument `--num_persons_to_track` references to the number of persons to track collectively


<!-- ## Results -->
