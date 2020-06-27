import numpy as np
import os
import h5py

import IPython


def updateModels(models,dt):
	"""
	Calls "update" method of each model to integrate for next time step
	
	each model must have the following method:
	update(dt, input1, input2, ... )
	
	models must be a dict with the following fields:
		'model' : pointer to model
		'inputs': list that contains model specific inputs in order expected
	"""
	for mName in models.keys():
		models[mName]['model'].update( dt,*models[mName]['input'] )

def getModelOutput(models, time):
	output = {}
	output['time'] = time
	for name in models.keys():
		output[name] = models[name]['model'].getOutput()
	return output


# 4th Order Runge Kutta Calculation
def RK4(f,x,u,dt,**kwargs):
	# Inputs: x[k], u[k], dt (time step, seconds)
	# Returns: x[k+1]

	#import IPython; IPython.embed()

	# Calculate slope estimates
	K1 = f(x, u, **kwargs)
	K2 = f(x + K1 * dt / 2, u, **kwargs)
	K3 = f(x + K2 * dt / 2, u, **kwargs)
	K4 = f(x + K3 * dt, u, **kwargs)

	# Calculate x[k+1] estimate using combination of slope estimates
	x_next = x + 1/6 * (K1 + 2*K2 + 2*K3 + K4) * dt


	return x_next,K1
	
def setupTime(tStart,tEnd,dt):
	
	N = int((tEnd-tStart)/dt+1)
	time,dt = np.linspace(tStart,tEnd,N,retstep=True)
	
	return time,dt,N
	
def saveData(filename,myData):
	# get full path for filename
	savePath = os.path.dirname(os.path.realpath(__file__))
	filepath = f'{savePath}\{filename}'
	
	print("Saving to...")
	print(f"\t{filepath}")
	with h5py.File(filepath,'w') as myFile:
		dList = []
		for myKey in myData.keys():
			dList.append(myFile.create_dataset(myKey,data=myData[myKey]))
	print('Done')