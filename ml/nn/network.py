import numpy as np
from math import log
from ..activation.util import sigmoid,sigmoidDerivative
from ..preprocess.util import *
from ..losses import LOSS
from ..optimizers.SGD import *
class brain:
	def __init__(self,ep=10e-4):
		self.e=ep
		self.layers=[]
		self.wpl=[]#weights per layer
		self.bpl=[]
	def fit(self,X,Y,layers=None,neurons=None,one_hot=True):
		""" Fits the training data:
		Parameters:
			X=training features(array like object)
			Y=training labels(array like object)
			layers=number of hidden layers(default=None)
			neurons=number of neurons for each layer (list),default=None
			one_hot=one hot vectors of the target values(training labels)
		"""
		self.class_set=set([_[0] for _ in Y if isinstance(_,(list,np.ndarray))])
		if not self.class_set:self.class_set=set(Y)
		X,Y=checkfit(X,Y)
		self.feat=X
		self.m=len(X)
		if one_hot:
			Y=one_hot_encoding(Y)
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
			#for j in range(1,len(self.layers)):
			#	w=np.random.random_sample((len(self.layers[j]),len(self.layers[j-1])+1))
			for j in range(1,len(neurons)):
				w=np.random.random_sample((neurons[j],neurons[j-1]+1))
				self.wpl.append(w)
			for k in range(1,len(neurons)):
				b=np.random.random_sample((1,neurons[k]))
				self.bpl.append(b)
	def train(self,optimizer="SGD",epoch=30):
		self.opt=SGD(self.feat,self.target,epoch,self.e)
		self.f_w,self.f_b=self.opt.optimize(self.wpl,self,bpl)	
					

