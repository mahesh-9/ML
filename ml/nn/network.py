import numpy as np
from ..activation import sigmoid
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
			class_set=set(Y)
			if one_hot:
				Y=checklabels(Y)
				self.target=Y
			if layers != None:
				if neurons==None:
					raise ValueError("Did not provide neurons to the hidden layers")
				if not isinstance(neurons,list):
					raise ValueError("parameter neurons should be of class list but given %s"%(type(neurons)))
				if not isinstance(layers,int):
					raise ValueError(" parameter layers should be of class int but given %s"%(type(layers))) 
				self.totl=len(layers)
				self.layers.append(X)
				for i in range(self.totl):
					l=[np.full([neurons[i],],0.0)]
					self.layers.append(l)
					if i==self.totl-1:
						out_l=[np.full([len(list(class_set)),],0.0)]
				for i in range(1,(len(self.layers))):
					w=np.random.random_sample((len(self.layers[i]),len(self.layers[i-1])+1))
					self.wpl.append(w)
	def _feed_forward(self,it,prime=False):
		""" calculates the activations of neurons which are inturn 
		inputs to the other hidden layers
		Parameters:
		it=layer number
		prime:if True calculates the derivative (default:False)
		"""
		temp_f=self.feat
		i=nx(temp_f)
		temp_f=i.add_col(col_no=0,val=1)
		activations=[]
		for i in range(len(self.layers[it])):
			s=0
			for j in self.wpl[it]:
				for k in range(len(j)):
					s+=j[k]*temp_f[k]
			if prime:
				activations.append(sigmoid(s,prime=True))
			else:activations.append(sigmoid(s))
		for i in range(len(self.layers[it])):
			self.layers[it][i]=activations[i]
	def cost(self):pass
		
		
		
		
		
		
			
					
				
			
	
			
		
		
		
		

		
		
		
					
					
				
					
				
					
					
					

		
