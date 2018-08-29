import numpy as np
from activation import activations 

class Layer:
	def __init__(self,units=None,activation=None):
		self.units=units
		try:
			self.activation=activations.__dict__[activation]
		except :raise("no such activation function")
	@property
	def get_weights(self):
		return self.weights
	@property
	def get_units(self):
		return self.units
	@property
	def get_activation(self):return self.activation
		
	def set_weights(self,prev_l_u=1):
		self.weights=np.random.rand(self.units,prev_l_u)
	def feed_forward(self,X):
		self.layer_activations=X
	@property
	def get_activations(self):
		return (self.weighted_sum,self.layer_activations)
class DNN(Layer):
	def __init__(self,units=None,activation=None):
		Layer.__init__(self,units=units,activation=activation)
		#self.weights=self.set_weights
	def feed_forward(self,X):
		self.weighted_sum=np.asarray(np.matmul(X,self.weights))
		self.layer_activations=np.asarray(self.activation(self.weighted_sum))
class Input(Layer):
	def __init__(self,units=None,activation='plain'):
		Layer.__init__(self,units=units,activation=activation)
		
