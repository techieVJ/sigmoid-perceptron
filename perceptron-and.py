import numpy as np

nb_epochs=10000

X=np.array([[0,0],
            [0,1],
            [1,0],
            [1,1]])

y=np.array([[0],
            [0],
            [0],
            [1]])

weights=np.random.random((2,1))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def predict(x):
    return sigmoid(np.dot(x,weights))



def sigmoid_derivative(x):
    return x*(1-x)



for i in range(nb_epochs):

    predicted_y=predict(X)
    error=(y-predicted_y)
    adjustment=np.dot(X.transpose()*error.transpose(),sigmoid_derivative(predicted_y))
    weights=weights+adjustment

print(weights)
print(predict(X))
