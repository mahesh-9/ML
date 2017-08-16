import numpy as np
from preprocess.util import normalize,one_hot
from preprocess.util import nx,checkfit
from losses import * 
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
		if self.ad_bias:
			return np.matmul(self.weights[1:],X)+self.weights[0]
		else:
			return np.matmul(self.weights,X)
	@property
	def params(self):
		if self.ad_bias:
			return self.weights,self.bias
		else:
			return self.weights[1:],self.weights[0]
class Regressor(Base):
	def __init__(self,add_bias=False,shuffle=False,one_hot=False,norm=False,no_iter=1000):
		super(Base,self).__init__(add_bias,shuffle,one_hot,norm)
		if self.it_no<0:raise ValueError("The number of iterations should be greater than or equal to zero")
	def fit(self,X,Y):
		if checkfit(X,Y):
			self.feat=X
			self.target=Y
			if ad_bias:
				self.weights=np.zeros(X.shape[1])
				self.bias=np.zeros(1)
			else:
				self.feat=nx(self.feat)
				self.feat.add_col(0,val=1)
				self.weights=np.zeros(self.feat.shape[1])
			if self.one_hot:
				self.target=one_hot(Y)
				self.loss="categorical_cross_entropy"
			else:self.loss="cross_entropy"
			self.tr_ex=len(self.feat)
			if self.tr<20000:
				self.opt="normal"
			else:
				self.opt="SGD"
			self._run()
	def _run(self):
		if self.opt="normal":
			Xt_X = self.feat.T.dot(self.feat)
			Xt_Y = self.feat.T.dot(self.target)
			self.weights = np.linalg.solve(Xt_X,Xt_Y)
			return self.weights
		else:pass
			
				
			

			
		
	
		
