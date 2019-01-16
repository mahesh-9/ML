import numpy as np
from ..activation import activations 
from ml import losses
class Layer:
	def __init__(self,units=None,activation=None):
		self.units=units
		try:
			self.activation=activations.__dict__[activation]
		except :raise ValueError("no such activation function")
	@property
	def get_weights(self):
		return self.weights
	@property
	def get_biases(self):
		return self.biases
	@property
	def get_units(self):
		return self.units
	@property
	def get_weighted_sum(self):return self.weighted_sum
	@property
	def get_activation(self):return self.activation
		
	def set_weights(self,prev_l_u=1):
		self.weights=np.random.rand(self.units,prev_l_u)
		#self.weights.dtype="float32"
		self.biases=np.random.rand(self.units,1)
	def _feed_forward(self,X):
		self.layer_activations=X
	@property
	def get_activations(self):
		return self.layer_activations
	def __call__(self,X):
		self._feed_forward(X)
		return self.layer_activations
	@property
	def loss(self):return self._loss
	@loss.setter
	def loss(self,l):
		self._loss=l
class DNN(Layer):
	def __init__(self,units=None,activation=None):
		Layer.__init__(self,units=units,activation=activation)
	def _feed_forward(self,X):
		self.weighted_sum=np.dot(self.weights,X)+self.biases
		self.layer_activations=np.asarray(self.activation(self.weighted_sum))
	def _backprop(self):pass
		
class Input(Layer):
	def __init__(self,units=None):
		Layer.__init__(self,units=units,activation="identity")
		
