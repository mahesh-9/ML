import numpy as np
from ..preprocess.util import normalize,one_hot
from ..preprocess.util import checkfit,nx
from ..losses import * 
from abc import ABCMeta,abstractmethod
class Base(metaclass=ABCMeta):
	def __init__(self,add_bias=True,shuffle=False,one_hot=False,norm=False,no_iter=1000):
		self.ad_bias=add_bias
		self.shuffle=shuffle
		self.one_hot=one_hot
		self.norm=norm
		self.it_no=no_iter
	@abstractmethod
	def fit(self,X,Y):
		"""fits features and targets for the model"""
	def predict(self,X):
		if not self.ad_bias:
			return np.matmul(X,self.weights[1:])+self.weights[0]
		else:
			return np.matmul(self.weights,X)
	@property
	def params(self):
		if self.ad_bias:
			return self.weights,self.bias
		else:
			return self.weights[1:],self.weights[0]
class Regressor(Base):
	def __init__(self,add_bias=False,shuffle=False,one_hot=True,norm=False,no_iter=1000):
		Base.__init__(self,add_bias,shuffle,one_hot,norm,no_iter)
		if self.it_no<0:raise ValueError("The number of iterations should be greater than or equal to zero")
	def fit(self,X,Y):
		if checkfit(X,Y):
			self.n_classes=len(set(Y))
			self.feat=X
			self.target=Y
			if self.ad_bias:
				if self.one_hot:
					self.target=one_hot(Y)
					self.loss="categorical_cross_entropy"
					self.weights=np.random.random_sample((self.feat.shape[1],self.n_classes))
					self.bias=np.random.random_sample((self.n_classes,))
				else:
					self.weights=np.zeros(X.shape[1])
					self.bias=np.zeros(1)
			else:
				temp=nx(X)
				self.feat=temp.add_col(0,val=1)
				if self.one_hot:
					self.target=one_hot(Y)
					self.loss="categorical_cross_entropy"
					self.weights=np.random.random_sample((X.shape[1],self.n_classes))
				else:self.weights=np.zeros(self.feat.shape[1])
			self.tr_ex=len(X)
			if self.tr_ex<20000:
				self.opt="normal"
			else:
				self.opt="SGD"
			self._run()
	def _run(self):
		if self.opt=="normal":
			Xt_X = self.feat.T.dot(self.feat)
			Xt_Y = self.feat.T.dot(self.target)
			self.weights = np.linalg.solve(Xt_X,Xt_Y)
			return self.weights
		elif self.opt=="SGD":pass
	
			
			
			
			
				
			

			
		
	
		
