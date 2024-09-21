# Index
|Sl no|Content||
|---|---|---|
|1|[Implementation of PSO on Custom Neural Network](#pso-implementation)||
|2|[Using pyswarm and keras](#using-pyswarm-and-keras-to-optimize-neural-network-models)||

## PSO Implementation
- Implementation of PSO on Custom Neural Network.
- Removed Backpropagation and in-turn Gradient Descent and use Particle Swarm Optimization technique for Neural Network Training.



## Activation Functions

### Relu
```
def relu(x):
    return np.maximum(x, 0)
```

### Leaky_relu
```
def leaky_rely(x, alpha=0.01):
    nonlin = relu(x)
    nonlin[nonlin==0] = alpha * x[nonlin == 0]
    return nonlin
```

### Sigmoid
```
def sigmoid(x):
    return 1 / (1 + np.exp(x))
```

### Softmax
```
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
```

### Tanh
```
def tanh(x):
    return np.tanh(x)


```
## Dataset
[Mnist Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) from Kaggle.


## Training Accuracy
<img src="\PSO\images\Figure_2.png">

## Fitness fuctions

### Loss saved values
[loss](https://github.com/iamrajharshit/OnPSO/blob/main/fitness_values.csv) 
## Refrence 
- [Repo on PSO](https://github.com/piyush2896/PSO-for-Neural-Nets)
- [GFG overview on PSO](https://www.geeksforgeeks.org/particle-swarm-optimization-pso-an-overview/)


## Using pyswarm and keras to optimize neural network models
- The [NoteBook]() containing training NN using brest_cancer dataset and optimized using PSO