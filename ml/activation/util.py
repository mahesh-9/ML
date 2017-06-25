import numpy as np
def sigmoid(X):
	"""an activation function which outputs the value between (0,1)"""
	if isinstance(X,np.ndarray):
		return 1.0/(1.0+np.exp(-X))
	else:
		X=np.array(X)
		return sigmoid(X)
		#return 1.0/(1.0+np.exp(-X))
def tanh(X):
	"""an activation function which outputs the value between (-1,1)"""
	if isinstance(X,np.ndarray):
		return (2.0/(1.0+np.exp(-2X)))-1
	else:
		X=np.array(X)
		return tanh(X)
def softmax(c,i):
	"""an activation function which outputs the value between (0,1)"""
	n=np.exp(c[i])
	s=0
	for j in range(len(c)):
		s+=np.exp(c[j])
	return n/(s)	
	
			
	
