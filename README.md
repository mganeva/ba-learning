# Deep learning for X-ray or neutron scattering under grazing-incidence: extraction of distributions

Walter Van Herck, Jonathan Fisher, Marina Ganeva

https://doi.org/10.1088/2053-1591/abd590

This repository will contain the source code and the trained models. 

## Requirements

To run the code you need the following software:
* [BornAgain](http://www.bornagainproject.org/)
* Python 3
* Tensorflow 1.x
* h5py


## Data generation

Scripts have been written for BornAgain 1.16 and might need to be adjusted to be used with newer versions of BornAgain. To generate the training and validation data:
1. Adjust the [configuration file](data_generation/config.json)
2. Run
```
$ cd data_generation
$ python gen_lattice.py config_file destination_dir
```
`config_file` is the path to the configuration file and `destination-dir` is the directory where training and validation data will be saved in `train` and `val` subdirectories.

Depending on hardware you use and amout of data required, data generation can easily take from a few hours till several days.

## Neural network training

To run the neural network training:
1. Adjust the [configuration file](densenet169_example.json)
2. Run
```
$ python train.py config_file
```

## Analysis

will come soon
