# Modern-Computer-Vision
PT/BR Notebooks to introduce neural nets as parametric non-linear function approximators in modern computer vision  

## Notebooks
### 1. Sine approximation
Although simple and intuitive,the sine function can't be easily formalized with an equation like f(x)= x². The sine values are given by the Y position of a line from the center of a circle to any angle, and by that, we could always check the sine and cossine values with a method as simple as taking a ruler and measuring it yourself in a circle with radius 1. 

In this notebook, we explore a way of describing such simple function but as an optimization problem, and we approximate it using simple and small matrices using backpropagation and Mean Squared Error.

### 2. Softmax Classifier (Linear classifier)
We'll take the optimization approach for vectors instead of scalars, and approach image classification as a problem of optimization. 

In this notebook we'll learn common procedures and good pratices, such as dataset pre processing, weight initialization. We'll also understand the process of backpropagation as staged computation using the chain rule, that way we'll be in the correct way of thinking deep learning models as DAGs(Directed Acyclic Graph). We'll be using use softmax to create a linear classifier for the CIFAR-10 Dataset and reach close to 40% accuracy on it using Negative Log Likelihood as our loss function! We visulize its neurons at the end of the notebook so we can understand what "learning" means for a Neural Net.

### 3. Two Layer Neural Net
In this notebook we upgrade our Softmax classifier by adding one more layer to it, and also by introducing the non-linear function ReLU. Other common procedures will be explained here, as training in batches, training a bigger model so we can get our feets wet with the Neural net backward pass, and a more pythonic writing for our models, as now everything will be inside a class.

We reach nearly 50% accuracy with this model, here we'll visualize again the neurons, see what got better, and what can't get better with our current approach. At the end of this notebook, we'll be finally getting into the modern ConvNets!
