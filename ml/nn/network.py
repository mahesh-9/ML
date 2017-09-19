import numpy as np
from math import log
from ..activation.util import sigmoid
from ..numc import *
from ..GLM.lr import checkfit,check_labels
class brain:
	def __init__(self,hidden_l=1,ep=10e-4):
		self.hidden_layers=hidden_l
		self.e=ep
		self.layers=[]
		self.wpl=[]
	def fit(self,X,Y,layers=None,neurons=None,one_hot=True):
		""" Fits the training data:
		Parameters:
			X=training features(array like object)
			Y=training labels(array like object)
			layers=number of hidden layers(default=None)
			neurons=number of neurons for each layer (list),default=None
			one_hot=one hot vectors of the target values(training labels)
		"""
		if checkfit(X,Y):
			self.feat=X
			self.m=len(X)
			self.class_set=set(Y)
			if one_hot:
				Y=check_labels(Y)
				self.target=Y
			if layers != None:
				if neurons==None:
					raise ValueError("Did not provide neurons to the hidden layers")
				if not isinstance(neurons,list):
					raise ValueError("parameter neurons should be of class list but given %s"%(type(neurons)))
				if not isinstance(layers,int):
					raise ValueError(" parameter layers should be of class int but given %s"%(type(layers))) 
				self.totl=layers
				self.layers.append(X)
				for i in range(self.totl):
					l=np.full([neurons[i],],0.0)
					self.layers.append(l)
					if i==self.totl-1:
						out_l=np.full([len(list(self.class_set)),],0.0)
						self.layers.append(out_l)
				for j in range(1,len(self.layers)):
					w=np.random.random_sample((len(self.layers[j]),len(self.layers[j-1])+1))
					self.wpl.append(w)
					

