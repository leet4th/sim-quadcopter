import numpy as np
import sympy as sym
from sympy.physics.mechanics import *

sym.init_printing(use_unicode=False)

def calcBfromQ(w,x,y,z):
	"""
	dq/dt = 1/2 [B(q)] w
	"""

	B = sym.zeros(4,3)
	B[0,0] = -x
	B[0,1] = -y
	B[0,2] = -z
	B[1,0] =  w
	B[1,1] = -z
	B[1,2] =  y
	B[2,0] =  z
	B[2,1] =  w
	B[2,2] = -x
	B[3,0] = -y
	B[3,1] =  x
	B[3,2] =  w

	return B

# Variables
qw, qx, qy, qz = sym.symbols('qw, qx, qy, qz')


B = calcBfromQ(qw,qx,qy,qz)

