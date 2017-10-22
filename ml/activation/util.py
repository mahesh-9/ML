from math import exp
import numpy as np
def sigmoid(X):
	"""an activation function which outputs the value between (0,1)"""
	
	if isinstance(X,np.ndarray):
		return 1.0/(1.0+np.exp(-X))
	else:
		X=np.array(X)
		return sigmoid(X)
		#return 1.0/(1.0+np.exp(-X))

def sigmoidDerivative(X):
	"""  input : array of features
	     output : Sigmoid derivative of the input  """

	if isinstance(X,np.ndarray):
		return sigmoid(X)*(1-sigmoid(X))
	else:
		X = np.array(X)
		return sigmoidDerivative(X)

def tanh(X):
	"""an activation function which outputs the value between (-1,1)"""
	if isinstance(X,np.ndarray):
		return (2.0/(1.0+np.exp(-(2*X))))-1
	else:
		X=np.array(X)
		return tanh(X)
def softmax(X):
	"""an activation function which outputs the value between (0,1)"""
	_X=np.full(X.shape,np.inf)
	for i in range(len(X)):
		_X[i]=exp(X[i])/(np.sum(np.exp(X),axis=0))
	return _X

def relu(X):
	if isinstance(X,np.ndarray):
		return np.maximum(X,0)
	else:
		X = np.array(X)
		return relu(X)

def reluDerivative(X,ep=1e-1):
	X = np.array(X)
	grads= 1.0*(X>0)
	grads[grads==0]=1e-1
	return grads
		
		
"""for testing """	
if __name__ == "__main__":
	a = relu([-1,-2,0,1,3,4])
	print(a)	
