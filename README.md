# Modern-Computer-Vision
PT/BR Notebooks to introduce neural nets as parametric non-linear function approximators in modern computer vision  

## Notebooks

### 1. Sine approximation
Although simple and intuitive,the sine function can't be easily formalized with an equation like f(x)= x². The sine values are given by the Y position of a line from the center of a circle to any angle, and by that, we could always check the sine and cossine values with a method as simple as taking a ruler and measuring it yourself in a circle with radius 1. 

In this notebook, we explore a way of describing such simple function but as an optimization problem, and we approximate it using simple and small matrices (1x16, 16x16 and 16x1) using backpropagation and Mean Squared Error.

#### Final result:
![image](https://user-images.githubusercontent.com/56324869/176484505-a99283a1-d645-4937-83ac-9ab96f4fdd7f.png)

------

### 2. Softmax Classifier (Linear classifier)
We'll take the optimization approach for vectors instead of scalars, and approach image classification as a problem of optimization. 

In this notebook we'll learn common procedures and good pratices, such as dataset pre processing, weight initialization. We'll also understand the process of backpropagation as staged computation using the chain rule, that way we'll be in the correct way of thinking deep learning models as DAGs(Directed Acyclic Graph). We'll be using use softmax to create a linear classifier for the CIFAR-10 Dataset and reach close to 40% accuracy on it using Negative Log Likelihood as our loss function! We visulize its neurons at the end of the notebook so we can understand what "learning" means for a Neural Net.

We see how the simplest pipeline for image classification as an optimization problem looks like:

![image](https://user-images.githubusercontent.com/56324869/176485112-ab3f0755-a61a-4870-9e94-cb51aec01f7c.png)

Learn how it looks like a graph and how to calculate its partial derivatives:

![image](https://user-images.githubusercontent.com/56324869/176485266-73158191-95c0-4ef7-b6cd-4d9b2dfa4075.png)

And we also visualize what happens inside the neurons when we add the gradient, and what happens when we apply gradient descent, visually and numerically: 

![image](https://user-images.githubusercontent.com/56324869/176485376-960af652-719b-49e9-801c-0f093b8dacd3.png)

![image](https://user-images.githubusercontent.com/56324869/176485624-cb55d092-2966-4e85-b358-422645c93bc9.png)

we visually the neurons after the learning process and interpret it:

![image](https://user-images.githubusercontent.com/56324869/176486299-879a7781-487c-4f64-a514-369c7f391ba0.png)


------


### 3. Two Layer Neural Net
In this notebook we upgrade our Softmax classifier by adding one more layer to it, and also by introducing the non-linear function ReLU. Other common procedures will be explained here, as training in batches, training a bigger model so we can get our feets wet with the Neural net backward pass, and a more pythonic writing for our models, as now everything will be inside a class.

We reach nearly 50% accuracy with this model, here we'll visualize again the neurons, see what got better, and what can't get better with our current approach. At the end of this notebook, we'll be finally getting into the modern ConvNets!

We learn how to make some optimized operations for derivatives:

![image](https://user-images.githubusercontent.com/56324869/176486571-7240653f-05eb-402b-bcc1-8d1f542f8d98.png)


We visualize our new graph for a two layer neural net:

![image](https://user-images.githubusercontent.com/56324869/176486659-e72e4815-b913-40a0-ae7f-65f43bd320f8.png)

And we take a look at what a 1000 neurons looks like after training:

![image](https://user-images.githubusercontent.com/56324869/176486752-fc51d470-f44a-4278-8da9-05dc0f84e803.png)


------


### 3.5. Airi

In this notebook we implement every layer we learned in a modular way, and also introduce the convolutions. Everything is inside the airi folder in this repo.

A linear layer implemented:

![image](https://user-images.githubusercontent.com/56324869/176487168-050adb9e-36da-4745-bd3e-073ddecaa401.png)

We also take a look on different optimizer on this notebooks, and see why SGD is often not used, but instead RMSProp, Adam, or even SGD + Momentum

![image](https://user-images.githubusercontent.com/56324869/176487346-05c1b1d6-e275-47f8-adc8-c33f81577865.png)


------


### 4. ConvNet

In this notebook we visualize the weights of our implemented full Convolutional Neural Network, written with our hand designed machine learning library (airi), its training script its inside airi folder. 

![image](https://user-images.githubusercontent.com/56324869/176487704-8a2bcb58-df38-403d-8fff-3eb96eaaf2b9.png)

5x5 learned filters

![image](https://user-images.githubusercontent.com/56324869/176487728-0eb90bcc-36c3-4974-be95-17aae66d24c3.png)

feature maps

![image](https://user-images.githubusercontent.com/56324869/176487777-81e298a5-f20e-4ee5-a462-904a83b45b43.png)

In this notebook we also take a better look inside of a much bigger convnet reading (Visualizing and Understanding Convolutional Networks)[https://arxiv.org/pdf/1311.2901.pdf]

------


### 5. torch convnet

In this notebook we build the very same convnet we built with Airi but with PyTorch, and learn how simple it its to write and train models in PyTorch.


------


### 6. Deep ConvNet

In this notebook we build VGG16 architecture using PyTorch, a old, but very powerful classification model that relies on operations we built at Airi (Conv2d, MaxPool, Relu, Linear, , Flatten, Softmax) and dropout.
