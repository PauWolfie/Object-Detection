# Object-Detection

### Summary:
This project deals with object detection using the You Only Look Once (YOLO) detection model implemented using a type of convolutional neural network called Darknet for image classification.

The code is implemented using Keras on Tensorflow.

***
### Project Structure

-The dataset used for this project is in */anotations* and */images*.
- The file *full_yolo_backend.h5* contains the weights of the Darknet network.

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
