from math import log,e
from ..numc import *
from .lr import LR

""" trying to implement kernel using guassian function """

class SVC():
	def fit(self,X,Y):
		"""Fits the training data(X,Y).
		n_eq=if true solves for theta using normal equations method.(default=False,solves using gradient descent.)
		"""
		if checkfit(X,Y):
			if not isinstance(X,np.ndarray):self.a=np.array(X)
			else:self.a=X
			inst=nx(self.a)
			self.feat=inst.add_col(col_no=0,val=1)
			self.l =normalize(X)
			self.target=Y
			self.theta=np.zeros(self.feat.shape[1])
			self.m=self.a.shape[0]
			self.gd()
	def f(self,x):
		""" f is a function of features X and l are the landmarks which are equal to training set features """
		x= normalize(x)
		for i in range(len(self.m)):
			f[i] = sim(self.x,self.l[i])
	
	def sim(self,sigma=None):
		S = np.exp(-(scipy.linalg.norm(self.x-self.l[i]))/2*(sigma**2))
		return S

	def hypo(self,it):
		return self.theta.dot(self.feat[it])

	def cost(self,C=None):
		s =0 
		for _ in range(self.m):
			s+= C*((self.target[_])*log(self.hypo(_))+ (1-self.target[_])*log(1-self.hypo(_)))
		return s

	def gd(self,rate=0.001,loops=100):
		for k in range(loops):
			for i in range(len(self.theta)):
				ts=0
				for j in range(self.m):
					res = (self.hypo(it=j)-self.target[j])*self.feat[j][i]
					ts+=res
				self.theta[i]-=rate*(1/(self.m))*(ts)
			t=self.cost()

	def predict(self,x):
		self.gd()
		x = np.array(x)
		if self.theta.T.dot(f()) >= 0:return 1
		else:return 0

	def checkfit(X,Y):
		if len(X)!=len(Y):
			raise ValueError("The length of the features vector and target vector does not match")
		elif len(X.shape)!=2:
			raise ValueError("all the values in the feature vector do not have same dimensions")
		else:return True
	
	def normalize(X):
		"""This function normalizes the feature values:
		new_x=(x-min(x))/(max(x)-min(x))"""
		mi=np.min(X,axis=0)
		ma=np.max(X,axis=0)
		r=np.full(X.shape,np.inf)
		for i in range(len(X)):
			q=(X[i]-mi)/(ma-mi)
			r[i]=q
		return r	
