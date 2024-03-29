# Object-Detection

### Summary:
This project deals with object detection using the You Only Look Once (YOLO) detection model implemented using a type of convolutional neural network called Darknet for image classification.

The code is implemented using Keras on Tensorflow.

***
### Project Structure:

- The dataset used for this project will go to the root directory. It can be downloaded from [here](https://drive.google.com/drive/folders/1bYZqQ4Bpeh-9vOd4zOuM4TGF1ElL3wZl?usp=sharing).
  
- The file *full_yolo_backend.h5* contains the weights of the Darknet network and it can be downloaded from [here](https://drive.google.com/file/d/17EBpgLF-LpPvchG4CUew1dfd8CSyG4ok/view?usp=sharing).
 
***
### About neuronal network:

We trained our neuronal network with 7 epochs and 5 times every image with data augmentation, (a total of 1500 images are processed in each epoch). We did a second training reducing the dataset to see how the network will work. If you want to try our training, you can try it downloading and putting it to the root directory with the name *red_lego.h5*:
- [First train](https://drive.google.com/file/d/1hK1yzGPQL63YtSub8Y0Hb-OYodp7eKub/view?usp=sharing) 
- [Second Train](https://drive.google.com/file/d/1UIfeY2QB2PjmKDAZM9IEIoidkUm8iSGn/view?usp=sharing)

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
*The versions of tesorflow and keras used are not the latest, but we need them for imgaug and h5py compatibility.*
