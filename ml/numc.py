import numpy as np
class nx:
	"""a class for custom methods in numpy"""
	def __init__(self,ob):
		if not isinstance(ob,np.ndarray):
			raise("expected np.array object")
		else:
			self.ob=ob
	def add_col(self,col_no,val=np.inf):
		"""this method returns an numpy.ndarray
		which consists of an extra column (with a same value) to the original array object passed in __init__.
		col_no=column number to add
		val=value to be passed""" 
		new_shape=self.ob.shape[0],self.ob.shape[1]+1
		b=np.full(new_shape,np.inf)
		for i in range(len(self.ob)):
			temp=np.insert(self.ob[i],col_no,val)
			b[i]=temp
		return b
	
			
		
