# This is a script that determines threshold using the bisection method
# Need to write an algorithm that evaluates the high, middle, and low values for action potential

import neuron
from neuron import h
import cell
import time as cookie
import numpy as np
import pickle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import scipy.stats as s

start = cookie.time()


def func(x, a, b, c):
    if b > 0:
        return 1e9 * np.ones(len(x))
    else:
        return a * np.exp(b * x) + c

#######################
# Threshold Amplitude #
#######################
# Obtained by finding an amplitude that elicits a spike
# and decreasing the duration of the pulse until a spike
# is no longer generated.

# Simulation parameters
synvars = {}
synvars['type'] = 'E2'
h.load_file('stdrun.hoc')
h.load_file('negative_init.hoc')
tstop = 200
dt = 0.025
h.stdinit()
h.celsius = 37.0
h.tstop = tstop
h.v_init = -65
gnatbar_default = 0.84

amp_hi = 1.0
amp_mid = 0.5
amp_lo = 0

# Instantiate and parameterize cells
cell_hi = cell.Cell(0, (0, 0), synvars, 'granulecell', 'output0_updated.swc')
cell_mid = cell.Cell(0, (0, 0), synvars, 'granulecell', 'output0_updated.swc')
cell_lo = cell.Cell(0, (0, 0), synvars, 'granulecell', 'output0_updated.swc')
# self.parameterizeCell(testCell,parameters)

# Create inputs for experiments
stim_hi = h.IClamp(0.5, sec=cell_hi.c.soma[0])
stim_hi.dur = 600
stim_hi.delay = 0
stim_hi.amp = amp_hi

stim_mid = h.IClamp(0.5, sec=cell_mid.c.soma[0])
stim_mid.dur = 600
stim_mid.delay = 0
stim_mid.amp = amp_mid

stim_lo = h.IClamp(0.5, sec=cell_lo.c.soma[0])
stim_lo.dur = 600
stim_lo.delay = 0
stim_lo.amp = amp_lo

# Instrument cells
somaticV_hi = h.Vector()
somaticV_hi.record(cell_hi.c.soma[0](0.5)._ref_v)
somaticV_mid = h.Vector()
somaticV_mid.record(cell_mid.c.soma[0](0.5)._ref_v)
somaticV_lo = h.Vector()
somaticV_lo.record(cell_lo.c.soma[0](0.5)._ref_v)

t = h.Vector()
t.record(h._ref_t)

# record action potentials
tvec_hi = h.Vector()
nc_hi = cell_hi.connect_pre(None, 0, 0)
nc_hi.record(tvec_hi)

tvec_mid = h.Vector()
nc_mid = cell_mid.connect_pre(None, 0, 0)
nc_mid.record(tvec_mid)

tvec_lo = h.Vector()
nc_lo = cell_lo.connect_pre(None, 0, 0)
nc_lo.record(tvec_lo)

# Run simulation
h.run()

# determine which cells fired an action potential
if (len(tvec_hi) > 0) and (len(tvec_mid) == 0):
    amp_hi = amp_hi
    amp_lo = amp_mid
if (len(tvec_mid) > 0) and (len(tvec_lo) == 0):
    amp_hi = amp_mid
    amp_lo = amp_lo
if (len(tvec_hi) > 0) and (len(tvec_lo) > 0) and (len(tvec_mid) > 0):
    print('Error with Parameters')

# new middle amplitude
amp_mid = 0.5*(amp_hi + amp_lo)
e_rel = 1e9
e_lim = 0.001

# relative diff b/w old and new midpoint
numloops = 0
while e_rel > e_lim:
    numloops = numloops + 1
    if len(tvec_mid) > 0:
        somaticV_hi = somaticV_mid
    else:
        somaticV_lo = somaticV_mid
    print('current amp_hi:', amp_hi)
    print ('current amp_mid:', amp_mid)
    print('current amp_lo:', amp_lo)
    amp_mid_old = amp_mid
    # create cell_mid
    cell_mid = cell.Cell(0, (0, 0), synvars, 'granulecell', 'output0_updated.swc')

    #attach current clamp
    stim_mid = h.IClamp(0.5, sec=cell_mid.c.soma[0])
    stim_mid.dur = 600
    stim_mid.delay = 0
    stim_mid.amp = amp_mid

    somaticV_mid = h.Vector()
    somaticV_mid.record(cell_mid.c.soma[0](0.5)._ref_v)

    t = h.Vector()
    t.record(h._ref_t)

    # set up variables to record action potential
    tvec_mid = h.Vector()
    nc_mid = cell_mid.connect_pre(None, 0, 0)
    nc_mid.record(tvec_mid)

    # Run simulation
    h.run()

    if len(tvec_mid) > 0:
        amp_hi = amp_mid
        amp_lo = amp_lo

    else:
        amp_hi = amp_hi
        amp_lo = amp_mid

    amp_mid = 0.5*(amp_hi + amp_lo)
    e_rel = np.abs(amp_mid - amp_mid_old)/amp_mid

#output the amp needed for AP, somatic voltage threshold, and plot
print("Took {0:.2f} seconds".format(cookie.time()-start))

print('amplitude that triggers action potential: ', amp_mid)
somaticvolt_thresh = max(somaticV_lo)
print('Somatic Voltage Threshold: ', somaticvolt_thresh)
print('Number of Loops: ', numloops)
Error = np.abs(somaticV_mid - somaticV_lo)/np.abs(somaticV_lo)
print('Error Estimate: ', np.mean(Error))

_=plt.figure()
_=plt.title('Final Somatic Membrane Voltages')
_=plt.plot(t,somaticV_lo, 'r-',t, somaticV_mid, 'b-', t, somaticV_hi, 'g')
plt.xlabel('time(s)')
plt.ylabel('Voltage(mV)')
one = mpatches.Patch(color='red', label='somaticV_lo', linewidth = 0.5, edgecolor = 'black')
two = mpatches.Patch(color='blue', label = 'somaticV_mid', linewidth = 0.5, edgecolor = 'black')
three = mpatches.Patch(color='green', label = 'somaticV_hi', linewidth = 0.5, edgecolor = 'black')

legend = plt.legend(handles=[one, two, three], loc = 4, fontsize = 'small', fancybox = True)

frame = legend.get_frame() #sets up for color, edge, and transparency
frame.set_facecolor('#b4aeae') #color of legend
frame.set_edgecolor('black') #edge color of legend
frame.set_alpha(1) #deals with transparency
plt.show()
plt.legend([somaticV_lo, somaticV_mid, somaticV_hi])
plt.show()




