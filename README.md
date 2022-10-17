# Modern-Computer-Vision
PT/BR Notebooks to introduce neural nets as parametric non-linear function approximators in modern computer vision  


## Environment Setup

You can use any Jupyter you want, like from Anaconda, Jupyter Notebook itself, Kaggle, Google Colab.

I use Jupyter notebook extension on vscode.

# Table of contents
1. [Sine approximation](#Sineapproximation)
2. [Softmax Classifier](#Softmaxclassifier)
3. [Two Layer Neural Net](#Twolayerneuralnet)
4. [Airi](#Airi)
5. [Airi ConvNet](#Airiconvnet)
6. [Pytorch Convnet](#Pytorchconvnet)
7. [Deep Convnet (VGG)](#Deepconvnetvgg)
8. [Pytorch Image Classification](#Pytorchimageclassification)

---

## Sine approximation <a name="Sineapproximation"></a>
Although simple and intuitive,the sine function can't be easily formalized with an equation like f(x)= x². The sine values are given by the Y position of a line from the center of a circle to any angle, and by that, we could always check the sine and cossine values with a method as simple as taking a ruler and measuring it yourself in a circle with radius 1. 

In this notebook, we explore a way of describing such simple function but as an optimization problem, and we approximate it using simple and small matrices (1x16, 16x16 and 16x1) using backpropagation and Mean Squared Error.

#### Final result:
![image](https://user-images.githubusercontent.com/56324869/176484505-a99283a1-d645-4937-83ac-9ab96f4fdd7f.png)

------

### 2. Softmax Classifier (Linear classifier)  <a name="Softmaxclassifier"></a>
We'll take the optimization approach for vectors instead of scalars, and approach image classification as a problem of optimization. 

In this notebook we'll learn common procedures and good pratices, such as dataset pre processing, weight initialization. We'll also understand the process of backpropagation as staged computation using the chain rule, that way we'll be in the correct way of thinking deep learning models as DAGs(Directed Acyclic Graph). We'll be using use softmax to create a linear classifier for the CIFAR-10 Dataset and reach close to 40% accuracy on it using Negative Log Likelihood as our loss function! We visulize its neurons at the end of the notebook so we can understand what "learning" means for a Neural Net.

We see how the simplest pipeline for image classification as an optimization problem looks like:

![image](https://user-images.githubusercontent.com/56324869/176485112-ab3f0755-a61a-4870-9e94-cb51aec01f7c.png)

Learn how it looks like a graph and how to calculate its partial derivatives:

![image](https://camo.githubusercontent.com/794575c66a75817db94e08c93f515dd1cfa4a5d94d65155c7a14a60072a4bc4c/68747470733a2f2f757365722d696d616765732e67697468756275736572636f6e74656e742e636f6d2f35363332343836392f3138313934353434392d37363166626631392d393531382d343063382d623066632d3436373363303935663736332e706e67)

And we also visualize what happens inside the neurons when we add the gradient, and what happens when we apply gradient descent, visually and numerically: 

![image](https://user-images.githubusercontent.com/56324869/196222139-8dd23135-fb0d-4d9e-8ea8-f430f9dbd5c7.png)

![image](https://user-images.githubusercontent.com/56324869/196222243-948c9e4e-c8de-4976-85a8-487f8fad73dc.png)

we visually the neurons after the learning process and interpret it:

![image](https://user-images.githubusercontent.com/56324869/176486299-879a7781-487c-4f64-a514-369c7f391ba0.png)


------


### Two Layer Neural Net <a name="Twolayerneuralnet"></a> 
In this notebook we upgrade our Softmax classifier by adding one more layer to it, and also by introducing the non-linear function ReLU. Other common procedures will be explained here, as training in batches, training a bigger model so we can get our feets wet with the Neural net backward pass, and a more pythonic writing for our models, as now everything will be inside a class.

We reach nearly 50% accuracy with this model, here we'll visualize again the neurons, see what got better, and what can't get better with our current approach. At the end of this notebook, we'll be finally getting into the modern ConvNets!

We learn how to make some optimized operations for derivatives:

![image](https://user-images.githubusercontent.com/56324869/176486571-7240653f-05eb-402b-bcc1-8d1f542f8d98.png)


We visualize our new graph for a two layer neural net:

![image](https://camo.githubusercontent.com/640004dc3fc42fdb96ffbbee4c43b81069fa1a291c8a2168ebe9e002be8dc6d7/68747470733a2f2f757365722d696d616765732e67697468756275736572636f6e74656e742e636f6d2f35363332343836392f3138333331353133372d64623235323937302d663561342d343266312d383830612d3963343530386437633132352e706e67)

And we take a look at what a 1000 neurons looks like after training and try to see some templates that were found in cifar-10:

![image](https://user-images.githubusercontent.com/56324869/176486752-fc51d470-f44a-4278-8da9-05dc0f84e803.png)


------


### 4. Airi  <a name="Airi"></a>

In this notebook we implement every layer we learned in a modular way, and also introduce the convolutions. Everything is inside the airi folder in this repo.

A linear layer implemented:

![image](https://user-images.githubusercontent.com/56324869/176487168-050adb9e-36da-4745-bd3e-073ddecaa401.png)

We also take a look on different optimizer on this notebooks, and see why SGD is often not used, but instead RMSProp, Adam, or even SGD + Momentum

![image](https://user-images.githubusercontent.com/56324869/176487346-05c1b1d6-e275-47f8-adc8-c33f81577865.png)


------


### 5. Airi ConvNet <a name="Airiconvnet"></a>

In this notebook we visualize the weights of our implemented Convolutional Neural Network, written with our hand designed machine learning library (airi). Its training script is inside airi directory. 

![image](https://user-images.githubusercontent.com/56324869/176487704-8a2bcb58-df38-403d-8fff-3eb96eaaf2b9.png)

5x5 learned filters

![image](https://user-images.githubusercontent.com/56324869/176487728-0eb90bcc-36c3-4974-be95-17aae66d24c3.png)

feature maps

![image](https://user-images.githubusercontent.com/56324869/176487777-81e298a5-f20e-4ee5-a462-904a83b45b43.png)

In this notebook we also take a better look inside of a much bigger convnet reading [Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901.pdf).

------


### 6. Pytorch ConvNet <a name="Pytorchconvnet"></a>

In this notebook we build the very same convnet we built with Airi but with PyTorch, and learn how simple it its to write and train models in PyTorch. Also, we extend that to a larger CNN to achieve 80% accuracy on cifar-10 validation set. Also, in this notebook we make use of Grad-Cam, to visualize what our model is looking to make its prediction:

![image](https://user-images.githubusercontent.com/56324869/196223294-8c6bde5d-0332-4f5b-9299-0215c93d10d4.png)


------


### 7. Deep ConvNet (VGG) <a name="deepconvnetvgg"></a>

In this notebook we build VGG16 architecture using PyTorch, an old, but very powerful classification model that relies on operations we built at Airi (Conv2d, MaxPool, Relu, Linear, , Flatten, Softmax) and dropout.

![image](https://user-images.githubusercontent.com/56324869/196224556-872dcd68-4d25-4040-9199-03fa68992cc9.png)

----

### 8. Pytorch Image Classification <a name="Pytorchimageclassification"></a>

In this last notebook, we'll take a dataset that consist of 15 cat breeds and classify them with an EfficientNet-B0, provided in the torchvision models package. This notebook is here as a general template for training classification models, here we learn how to split a dataset, create different transforms to the dataset and transform it into a dataloader. Also, we visualize now at big images with Grad-Cam what our model is looking to make its prediciton, see below at the last layer of efficientnet-b0 what its activating the Siamese neuron, and also take a look on its second prediciton (Maine Coon) and what activates that neuron:

Siamese:

![image](https://user-images.githubusercontent.com/56324869/196224039-81d64cf0-277e-442d-bd1b-37831b0dda67.png)

Maine Coon:

![image](https://user-images.githubusercontent.com/56324869/196224393-51093ac5-792f-4045-b246-8f0a337cf94f.png)
