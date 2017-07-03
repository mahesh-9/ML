from math import log,e
from ..numc import *
from .lr import LR
from .LogisticRegr import LogisticRegression

""" trying to implement linear kernel using SVC """

class SVM():
	def fit(self,X,Y):
		LR.fit(self,X,Y)
		return self
	def hypo(self,it):
		res= 1/(1+e**(-LR.hyp(self,it)))
		return res
	def cost1(self):
		cos1 = -log(self.hypo(_))
		return cos1
	def cost2(self):
		cos2 = -log(1-(self.hypo(_)))
		return cos2
	def cost(self):
		s =0 
		for _ in range(self.m):
			s+= (self.target[_])*self.cost1(_)+ (1-self.target[_])*self.cost2(_)
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
		return x.dot(self.theta[1:])+self.theta[0]

		
