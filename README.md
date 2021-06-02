# Object-Detection

### Resum:
Aquest projecte tracta la detecció d'objectes mitjançant el model de detecció You Only Look Once (YOLO) implementat mitjançant un tipus de xarxa neuronal convolucional anomenada Darknet per a la classificació d'imatges.

El codi està implementat utilitzant keras sobre Tensorflow.

***
### Estructura del projecte

- El dataset utilitzat per aquest projecte es troba en */anotations* i */images*.
- L'arxiu *full_yolo_backend.h5* conté els pesos de la xarxa Darknet.

***
### Anotacions importants per a la seva execució:
Aquest projecte s'ha implementat en python **3.6**. En qualsevol cas, la pràctica pot ser testejada en python **3.6.+**.
Es recomanable crear un entorn virtual de python i instalar les seguents eines:
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

***IMPORTANT***
*Les versions emprades de tesorflow i keras no son les actualitzades, però les necessitem per la compatibilitat amb imgaug i h5py.*
