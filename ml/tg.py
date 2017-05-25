"""Implementing Linear regression for multiple features"""
import numpy as np
from .numc import *
from sklearn.datasets import load_iris
class LR():
	#def __init__(self):
		#self.get_data()
		#self.count=0
	def fit(self,X,Y):
		"""extracting data from sklearn.datasets.load_iris() object"""
		self.a=X
		inst=nx(self.a)
		self.feat=inst.add_col(col_no=0,val=1)
		self.target=Y
		self.theta=np.zeros(self.feat.shape[1])
		self.m=self.a.shape[0]
	def hyp(self,it=None):
		"""returns the hypothesis equation of the pirticular feature"""
		return self.theta.dot(self.feat[it])				
	def cost(self):
		"""computes the cost"""
		#self.count+=1
		#self.m=self.a.shape[0]
		sum=0
		for i in range(self.m):
			res=(self.hyp(it=i)-self.target[i])**2
			sum+=res
		return (1/2*(self.m))*(sum)	
	def gd(self,rate=0.001,loops=700):
		"""implementing the gradient descent"""
		for k in range(loops):
			for i in range(len(self.theta)):
				ts=0
				for j in range(self.m):
					res=(self.hyp(it=j)-self.target[j])*self.feat[j][i]
					ts+=res
				self.theta[i]-=rate*(1/(self.m))*(ts)
				#self.theta[i]-=rate*(ts)
			t=self.cost()
		#return self.theta
	def predict(self,x):
		"""predicting the values:"""
		self.gd()
		return x.dot(self.theta[1:])
	@property
	def intercept_coef(self):
		return "intercept:{} coef:{}".format(self.theta[0],self.theta[1:])
		

				
			
		
		
