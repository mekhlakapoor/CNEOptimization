# Script to test single target_obj
from optFunc_granuleCell_v2 import Objectives
import numpy as np
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

target_obj = {}
target_obj['thresh'] = -48.70   #mV
target_obj['Rin'] = 183         #MOhms
target_obj['RMP'] = -70.4       #mV
target_obj['tau'] = 30          #ms
target_obj['sfa'] = 0.31        #np.array([0.31205673758805536])
target_obj['fAHP'] = -11.7      #incorrect value
target_obj['mAHP'] = 4.25      #incorrect value

# Weighting for objectives
weight_obj = {}
weight_obj['thresh'] = np.array([1,0.1])
weight_obj['Rin'] = np.array([1])
weight_obj['RMP'] = np.array([1])
weight_obj['tau'] = np.array([1])
weight_obj['sfa'] = np.array([1])
weight_obj['fAHP'] = np.array([1])
weight_obj['mAHP'] = np.array([1])

# Parameters for neuron
params = {}
params['gkfbar_ichan2'] = 0.036
params['gksbar_ichan2'] = 0.006
params['gkabar_borgka'] = 0.009
params['gl_ichan2'] = 0.000290152
params['gncabar_nca'] = 0.0007353
params['glcabar_lca'] = 0.0025
params['gcatbar_cat'] = 0.000074
params['gskbar_gskch'] = 0.001
params['gkbar_cagk'] = 0.00012

# Instantiating Objective class
Obj = Objectives(target_obj,weight_obj)

na_factor = np.linspace(0,10.0,11)
hi_list = []
mid_list = []
lo_list = []
thresh_list = []
# for loop to change Na values
for na in na_factor:
    params['gnatbar_ichan2'] = 0.84*na
    hi, mid, lo, t, thresh = Obj.test_objective('thresh', params)
    hi_list.append(hi)
    mid_list.append(mid)
    lo_list.append(lo)
    thresh_list.append(thresh)

#set 3 different voltages
amp_hi = 1.0
amp_mid = 0.5
amp_lo = 0

print('Threshlist:', thresh_list)
for mm in range(len(na_factor)):
    # Plot data
    _ = plt.figure()
    _ = plt.title('Somatic Membrane Voltages{}'.format(na_factor[mm]))
    _ = plt.plot(t, lo_list[mm], 'r-', t, mid_list[mm], 'b-', t, hi_list[mm], 'g')
    plt.xlabel('time(s)')
    plt.ylabel('Voltage(mV)')
    one = mpatches.Patch(color='red', label='somaticV_lo', linewidth=0.5, edgecolor='black')
    two = mpatches.Patch(color='blue', label='somaticV_mid', linewidth=0.5, edgecolor='black')
    three = mpatches.Patch(color='green', label='somaticV_hi', linewidth=0.5, edgecolor='black')

    legend = plt.legend(handles=[one, two, three], loc=4, fontsize='small', fancybox=True)

    frame = legend.get_frame()  # sets up for color, edge, and transparency
    frame.set_facecolor('#b4aeae')  # color of legend
    frame.set_edgecolor('black')  # edge color of legend
    frame.set_alpha(1)  # deals with transparency
plt.show()
    #plt.legend([lo_list[mm], mid_list[mm], hi_list[mm]])


