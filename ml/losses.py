#TODO Not completed yet. 
import numpy as np
from math import log,e,exp
from ml.activation.activations import softmax,sigmoid
#class LOSS:

def sigmoidal(X):
	return sigmoid(normal(self,X))


def soft_max(X):
	"""an activation function which outputs the value between (0,1)"""
	_X=np.full(X.shape,np.inf)
	#for i in range(len(X)):
	_X=np.exp(X)/(np.sum(np.exp(X),axis=0))
	return _X
	#@staticmethod
def mean_squared_loss(features,targets,hyp="Regression"):
	self.f=features
	self.hyp=HYP[hyp]
	self.target=targets
	self.no_train=len(self.feat)
	su=0
	for i in range(self.no_train):
		res=(self.hyp(self.feat[i])-self.target[i])**2
		su+=res
	return np.mean(su)

	#@staticmethod
def cross_entropy(self,targets,features,hyp="binary_class"):
	self.feat=features
	self.target=targets
	self.hyp=HYP[hyp]
	self.no_train=len(self.feat)
	s=0
	for _ in range(self.no_train):
		s+=self.target[_]*log(self.hyp(self.feat[_]))+(1-self.target[_])*log(1-self.hypo(self.feat[_]))
	r=(1/self.no_train)*s
	return r

	#@staticmethod
def categorical_cross_entropy(feat,target,hyp="multiclass",model="nn"):
	r = np.sum(np.dot(target,np.log(sigmoid(feat))))
	return -r

def normal(X):
	return np.matmul(obj.weights,X)
#HYP={"Regression":LOSS.normal,"binary_class":LOSS.sigmoidal,"multiclass":LOSS.soft_max}
			
			
def cost(AL,Y):
	"""
	Args:
	AL -- label predictions vector
	Y -- true "label" vector (feat)
	"""
	m = Y.shape[0]
	res=np.sum(AL-Y)
	return res		
			
			
		
		
		
		
		
		
	
	
