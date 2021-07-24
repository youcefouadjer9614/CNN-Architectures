# CNN-Architectures


'''
Implementing a modified version of the classical Convolutional Neural Network (LeNet-5)

Proposed by Professor LeCun et al
    
We need to modify the architecture (padding) when using it with built-in MNIST Dataset of Pytorch
    
    
MNIST dimension is 28 X 28:

so use padding of p = 2

input dimension: (n=28 + 2*p=2) x (n=28 + 2*p=2) >>>> MNIST digit of 32 x 32 dimensions
    
''' 
