 ![Language](https://img.shields.io/badge/language-python--3.8.5-blue) [![Contributors][contributors-shield]][contributors-url] [![Forks][forks-shield]][forks-url] [![Stargazers][stars-shield]][stars-url] [![Issues][issues-shield]][issues-url] [![MIT License][license-shield]][license-url] [![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<!-- <br />

<p align="center">
  <a href="https://github.com/vineeths96/Video-Frame-Prediction">
    <img src="docs/readme/stconv.jpg" alt="Logo" width="300" height="175">
  </a>
  <h3 align="center">Video Frame Prediction</h3>
  <p align="center">
    Video Frame Prediction using Spatio Temporal Convolutional LSTM
    <br />
    <a href=https://github.com/vineeths96/Video-Frame-Prediction><strong>Explore the repository»</strong></a>
    <br />
    <a href=https://github.com/vineeths96/Video-Frame-Prediction/blob/master/docs/report.pdf>View Report</a>
  </p>

</p> -->

> tags : video prediction, frame prediction, spatio temporal, convlstms, generative networks, discriminative networks, movingmnist, deep learning, pytorch 



<!-- ABOUT THE PROJECT -->

## About The Project

Detecting and Recoginising faces have become an extremly important task in computer vision. 

### Dataset

[TODO]: make this clean 

The custom dataset that we have created fot this task is stored in these urls:
* https://drive.google.com/file/d/1bnj8sdugWK-L40JemSTNvsML3cNuq-iG/view?usp=sharing
* https://drive.google.com/file/d/17V1uJSLp9KheB4t9MFdROaktgSieN6CU/view?usp=sharing


### Built With
This project was built with 

[TODO]: add information of the versions used 

* python v3...
* PyTorch v1.7
* The environment used for developing this project is available at [environment.yml](environment.yml).



<!-- GETTING STARTED -->

## Getting Started

Clone the repository into a local machine and enter the [src](src) directory using

```
git clone ...
cd ...
```

### Prerequisites

Create a new conda environment and install all the libraries by running the following command

```shell
conda env create -f environment.yml
```

The dataset used in this project (Moving MNIST) will be automatically downloaded and setup in `data` directory during execution.

### Instructions to run

To train the model on *m* nodes and *g* GPUs per node run,

```sh
python -m torch.distributed.launch --nnode=m --node_rank=n --nproc_per_node=g main.py --local_world_size=g
```

This trains the frame prediction model and saves it in the `model` directory.

This generates folders in the `results` directory for every log frequency steps. The folders contains the ground truth and predicted frames for the train dataset and test dataset. These outputs along with loss and metric are written to Tensorboard as well.



## Model overview

<!-- RESULTS -->

## Results

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->

## Contact


## Acknowledgments

> Base code is taken from:
>
>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!-- 
[contributors-shield]: https://img.shields.io/github/contributors/vineeths96/Video-Frame-Prediction.svg?style=flat-square
[contributors-url]: https://github.com/vineeths96/Video-Frame-Prediction/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/vineeths96/Video-Frame-Prediction.svg?style=flat-square
[forks-url]: https://github.com/vineeths96/Video-Frame-Prediction/network/members
[stars-shield]: https://img.shields.io/github/stars/vineeths96/Video-Frame-Prediction.svg?style=flat-square
[stars-url]: https://github.com/vineeths96/Video-Frame-Prediction/stargazers
[issues-shield]: https://img.shields.io/github/issues/vineeths96/Video-Frame-Prediction.svg?style=flat-square
[issues-url]: https://github.com/vineeths96/Video-Frame-Prediction/issues
[license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/vineeths96/Video-Frame-Prediction/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/vineeths -->
