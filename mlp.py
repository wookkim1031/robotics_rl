"""
MultiLayer Perceptron (MLP)

1. linear function aggregation z 
2. sigmoid function activation a 
3. cost function (error) calculation e
4. derivative of the error w.r.t the weights w and b in the L layer
5. derivative of the error w.r.t the wieghts w nad b in the (L-1) layer
6. weight and bias update for the (L) layer
7. weight and bias update for the (L - 1) layer
"""

import numpy as np

def init_parameters(n_features, n_neurons, n_output):
    np.random.seed(100)
    W1 = np.random.uniform(size=(n_features, n_neurons))
    W2 = np.random.uniform(size=(n_neurons, n_output))
    b1 = np.random.uniform(size=(1, n_neurons))
    b2 = np.random.uniform(size=(1, n_output))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def linear_function(W, X, b): 
    Z = (X @ W) + b
    return Z 

def sigmoid_function(Z):
    return 1/(1+np.exp(-Z))

def train(X, Y, n_features=2, n_neurons=3, n_output=1, iterations=10, eta=0.001):
    params = init_parameters(n_features=n_features,
                            n_neurons=n_neurons,
                            n_output=n_output)

    errors = []
    # First Layer
    
    for _ in range(iterations):
        Z1 = linear_function(params["W1"], X, params["b1"])
        S1 = sigmoid_function(Z1)
        # Second Layer
        Z2 = linear_function(params["W2"], S1, params["b2"])
        S2 = sigmoid_function(Z2)

        loss = np.mean((S2 - Y) ** 2)
        errors.append(loss)

        # Backpropagation
        """
        How does the error change when we change the weights by a tiny amount?  

        We update the weight and the bias term 
        """
                # update output weights
        delta2 = (S2 - Y)* S2*(1-S2)
        W2_gradients = S1.T @ delta2

        # update hidden weights
        delta1 = (delta2 @ params["W2"].T )* S1*(1-S1)
        W1_gradients = X.T @ delta1 

        params["W1"] = params["W1"] - W1_gradients * eta
        params["W2"] = params["W2"] - W2_gradients * eta
        # update output bias
        params["b2"] = params["b2"] - np.sum(delta2, axis=0, keepdims=True) * eta
        # update hidden bias
        params["b1"] = params["b1"] - np.sum(delta1, axis=0, keepdims=True) * eta
        
    return errors, params


"""
XOR example
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

errors, params = train(X, Y, iterations=10000, eta=0.5)
# error start and end of the training 
print(errors[0], errors[-1])
# Prediction from the input 
print(forward_pass := sigmoid_function(
    sigmoid_function(X @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"]))
"""

