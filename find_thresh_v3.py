# This is a script that tries to find the value of the threshold.
# NOTES on transitioning objectives:
# One of the main changes you will notice is that many variables that were named
# "self.whatever" had the "self." part removed.
# The other thing is that the self.parameterizeCell commands were commented out.
# The last thing I guess is that plotting the results will change depending on the objective

from neuron import h
import cell
import time as cookie
import numpy as np
import pickle
import pylab as plt 
import scipy.stats as stats
start=cookie.time()

def func(x,a,b,c):
	if b > 0:
		return 1e9*np.ones(len(x))
	else:
		return a*np.exp(b*x)+c

#######################
# Threshold Amplitude #
#######################
# Obtained by finding an amplitude that elicits a spike
# and decreasing the duration of the pulse until a spike
# is no longer generated.

# Simulation parameters
synvars = {}
synvars['type']='E2'
h.load_file('stdrun.hoc')
h.load_file('negative_init.hoc')
tstop = 200
dt = 0.025
h.stdinit()
h.celsius = 37.0
h.tstop = tstop
h.v_init = -65

# Need to make sure the amplitude elicits a spike
flag = 1
addon = 0

# Using count to find cases where the cell is
# unable to fire an action potential
count = 0
stop = 0

# Define a factor to scale threshold amplitude if things are going too slow... or too fast...
factor = 1

while flag:
	# Instantiate and parameterize cells
	testCell = cell.Cell(0,(0,0),synvars,'granulecell','output0_updated.swc')
	#self.parameterizeCell(testCell,parameters)
	
	# Create inputs for experiments
	stim = h.IClamp(0.5,sec=testCell.c.soma[0])
	stim.amp = factor*(0.45 + addon)
	stim.dur = 200
	stim.delay = 0

	# Instrument cells
	threshV = h.Vector()
	threshV.record(testCell.c.soma[0](0.5)._ref_v)
	
	tvec = h.Vector()
	nc = testCell.connect_pre(None,0,0)
	nc.record(tvec)

	# Run simulation
	h.run()

	if len(tvec) > 0:
		spiketime = tvec[0]
		flag = 0
		threshAmp = factor*(0.45+addon)
	else:
		addon += 0.1
	
	count += 1
	if count > 10:
		flag = 0
		stop = 1
		thresh = 1000

######################
# Threshold Duration #
######################
if not stop:
	threshVDur = []
	for ii in range(1,20):
		# Simulation parameters
		h.load_file('stdrun.hoc')
		h.load_file('negative_init.hoc')
		tstop = 100
		dt = 0.025
		h.stdinit()
		h.celsius = 37.0
		h.tstop = spiketime + 10
		h.v_init = -65
		
		# Instantiate and parameterize cells
		testCell = cell.Cell(0,(0,0),synvars,'granulecell','output0_updated.swc')
		#self.parameterizeCell(testCell,parameters)
		
		# Create inputs for experiments
		stim = h.IClamp(0.5,sec=testCell.c.soma[0])
		stim.amp = threshAmp
		stim.dur = spiketime-ii
		stim.delay = 0
		
		# Instrument cells
		threshV = h.Vector()
		threshV.record(testCell.c.soma[0](0.5)._ref_v)
		
		t = h.Vector()
		t.record(h._ref_t)
		
		tvec = h.Vector()
		nc = testCell.connect_pre(None,0,0)
		nc.record(tvec)
		
		# Run simulation
		h.run()
		
		# Calculate threshold
		threshV = np.array(threshV)
		threshVDur.append(threshV)
		if len(tvec) == 0:
			thresh = np.max(threshVDur[-1])
			subthresh_dur = spiketime-ii
			break

print("Took {0:.2f} seconds".format(cookie.time()-start))

_=plt.figure()
_=plt.title('Threshold Duration')
for v in threshVDur:
	_=plt.plot(t,v)

plt.show()

