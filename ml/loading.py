import sys
import time
def a(n):
	s=""
	r=chr(n)*1
	for j in range(5):
		s=""
		for i in range(20):
			s+=r
			sys.stdout.write("\r"+s)
			sys.stdout.flush()
			time.sleep(0.5)
		sys.stdout.write("\n")

if __name__=="__main__":
	#a(128187)
	a(9209)
	sys.stdout.write("\n")
		
