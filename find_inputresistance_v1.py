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

inputresistance_arr = []
voltage_master = []
amps_master = []
steadystate_master = []
x_list = []
y_list = []
r_value_list = []
p_value_list = []
std_err_list = []
na_factor = np.linspace(0,10.0,11)
# for loop to change Na values
for na in na_factor:
	# Instantiate and parameterize cells and inputs for experiment
	cells = []
	stim_list = []
	threshV_list = []
	t_list = []
	amps_list = np.linspace(-0.05, 0.05, 11)
	steadystate_arr = []

	gnatbar_default = 0.84 #default na value
	for ii in range(len(amps_list)):
		# create 5 instrument cells and organize them in a list
		testCell = cell.Cell(0,(0,0),synvars,'granulecell','output0_updated.swc')
		testCell.c.soma[0].gnatbar_ichan2 = na*gnatbar_default
		cells.append(testCell)

		# store the stimulus amplitudes in a list for each cell
		stim = h.IClamp(0.5,sec=cells[ii].c.soma[0])
		stim_list.append(stim)
		stim_list[ii].dur = 600
		stim_list[ii].delay = 0
		stim_list[ii].amp = amps_list[ii]


		# record and store the thresholds for each of these cells
		threshV = h.Vector()
		threshV.record(cells[ii].c.soma[0](0.5)._ref_v)
		threshV_list.append(threshV)

		# a list for time, to make checking easier
	t = h.Vector()
	t.record(h._ref_t)
	t_list.append(t)

	h.celsius = 35
	# Run simulation
	h.run()

	#saving threshV
	for jj in range (len(amps_list)):
		steadystate_arr.append(threshV_list[jj][-1])

	# Conversions
	amps_list = np.array(amps_list)/(10**9) #nanoamps to amps
	steadystate_arr = np.array(steadystate_arr)/(10**3) #millivolts to volts
	# np.array converts list to a numerical array

	#input resistance = slope
	#append an input resistance list for these
	inputresistance, intercept, r_value, p_value, std_err = stats.linregress(amps_list, steadystate_arr)
	#create lists that grab r_value, p_value, std_err (tell how good the fit was)
	r_value_list.append(r_value)
	p_value_list.append(p_value)
	std_err_list.append(std_err)
	inputresistance = inputresistance/(10**6)
	inputresistance_arr.append(inputresistance)
	x = np.linspace(min(amps_list),max(amps_list), 10) #makes a list b/w min and max ampslist with 10 values equally apart
	y = inputresistance * x * (10**6) + intercept

	# appending master lists with the lists for each Na
	voltage_master.append(threshV_list)
	amps_master.append(amps_list)
	steadystate_master.append(steadystate_arr)
	x_list.append(x)
	y_list.append(y)
np.save('combo_inputresistance{}.npy',np.array(inputresistance_arr))
np.save('combo_rvalue{}.npy',np.array(r_value_list))
np.save('combo_pvalue{}.npy',np.array(p_value_list))
np.save('combo_stderr{}.npy',np.array(std_err_list))

print("r-values:" , r_value_list)
print("p-values:" , p_value_list)
print("standard deviations:", std_err_list)

#save stats lists out as numpy arrays after all simulations 

# end of Na values for loop here
print("Took {0:.2f} seconds".format(cookie.time()-start))
print("input resistance:", inputresistance_arr)
for mm in range(len(na_factor)):
	_ = plt.figure()
	_ = plt.title('Input Resistance{}Na'.format(na_factor[mm]))
	#print(x_list[mm])
	#print(y_list[mm])


	_ = plt.plot(x_list[mm], y_list[mm])  # access lists
	plt.scatter(amps_master[mm], steadystate_master[mm])
	plt.savefig('Combo_InputResistance_Na{}.png'.format(na_factor[mm]))

	# 2nd figure
	plt.figure()
	for kk in range(11):
		plt.title('Voltage{}Na'.format(na_factor[mm]))
		plt.plot(voltage_master[mm][kk])
	plt.savefig('Combo_Voltage{}Na.png'.format(na_factor[mm]))
plt.show()


