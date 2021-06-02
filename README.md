# Object-Detection

### Summary:
This project deals with object detection using the You Only Look Once (YOLO) detection model implemented using a type of convolutional neural network called Darknet for image classification.

The code is implemented using Keras on Tensorflow.

***
### Project Structure

- The dataset used for this project will go to the root directory. It can be downloaded from here:
  https://drive.google.com/drive/folders/1bYZqQ4Bpeh-9vOd4zOuM4TGF1ElL3wZl?usp=sharing
  
- The file *full_yolo_backend.h5* contains the weights of the Darknet network and it can be downloaded from here: 
  https://drive.google.com/file/d/17EBpgLF-LpPvchG4CUew1dfd8CSyG4ok/view?usp=sharing
 
***
### About de neuronal network

We trained our neuronal network with 7 epochs and 5 times every image with data augmentation, (a total of 1500 images are processed in each epoch). If you want to try our training, you can try it downloading and putting it to the root directory:
s

***
### Important anotations for the execution:
This project has been implemented in python **3.6**. In any case, the practice can be tested in python **3.6.+**.
It is recommended to create a virtual python environment and install the following tools:
~~~
pip install tensorflow==1.13.2
pip install keras==2.0.8
pip install imgaug==0.2.5
pip install opencv-python
pip install h5py==2.10.0
pip install tqdm
pip install imutils
~~~

***

***IMPORTANT***:
*The versions of tesorflow and keras used are not up to date, but we need them for imgaug and h5py compatibility.*
