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
tstop = 600
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

# Instantiate and parameterize cells and inputs for experiment
cells = []
stim_list = []
threshV_list = []
t_list = []
amps_arr = []
for ii in range(5):
	# create 5 instrument cells and organize them in a list
	testCell = cell.Cell(0,(0,0),synvars,'granulecell','output0_updated.swc')
	cells.append(testCell)

	# store the stimulus amplitudes in a list for each cell
	stim = h.IClamp(0.5,sec=cells[ii].c.soma[0])
	stim_list.append(stim)
	stim.dur = 600
	stim.delay = 0

	# record and store the thresholds for each of these cells
	threshV = h.Vector()
	threshV.record(cells[ii].c.soma[0][0.5]._ref_v)
	threshV_list.append(threshV)

	# a list for time, to make checking easier
	t = h.Vector
	t.record(h._ref_t)
	t_list.append(t)


# Conversions
amps_arr = np.array(amps_arr/(10**15)) #nanoamps to megaamps
steadystate_arr = np.array(steadystate_arr/10**9) #millivolts to megavolts

#input resistance
inputresistance_arr = np.array(amps_arr/steadystate_arr)
inputresistance = len(inputresistance_arr) #average input resistance (avg slope) 



#tvec = h.Vector() #spike time
#nc = testCell.connect_pre(None,0,0)
#nc.record(tvec)

# Run simulation
h.run()

#saving threshV
steadystate_arr.append = threshV[-1] #trying to make an array from last values of threshV array?

threshV = np.array(threshV)

print("Took {0:.2f} seconds".format(cookie.time()-start))
print("steady state value:", threshV[-1])

_=plt.figure()
_=plt.title('Threshold Duration')

_=plt.plot(t,threshV)

plt.show()

