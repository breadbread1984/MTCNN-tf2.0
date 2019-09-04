# MTCNN-tf2.0
an implement of MTCNN facial detection algorithm

## Install prerequisite packages

install prerequisite packages with the following command

```bash
sudo apt install python-opencv
pip3 install -U tf-nightly-gpu-2.0-preview
```

## Download datasets

training MTCNN requires two datasets downloaded. One is Wider face. Another is CelebA.

### Download and convert Wider Face to tfrecord

Download widerface [here](http://shuoyang1213.me/WIDERFACE/). And unzip the directorys

>WIDER_train WIDER_val wider_face_split

right under current directory.

Then execute the following command to generate wider_face_train.tfrecord and wider_face_val.tfrecord.
```bash
python3 create_widerface_datasets.py
```

### Download and convert Celeb A to tfrecord

execute the following command to generate celeba datasets.
```bash
python3 create_celeba_datasets.py
```

## Train the model
