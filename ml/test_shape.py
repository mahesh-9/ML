import numpy as np
class wrapper:
	def __init__(self,f,include_shape=False):
		self.f=f
		self.flag=False
	def __call__(self,*args,**kwargs):
		inp=np.asarray(args[0]).shape
		out=self.f(*args,**kwargs)
		outs=out.shape
		state="ok" if inp==outs else "not ok"
		print(state)
		return out
	def __getattr__(self,attr):
		return getattr(self.f,attr)

