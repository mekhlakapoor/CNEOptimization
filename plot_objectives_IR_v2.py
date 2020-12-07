# Script to test Input resistance, with different Na
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
master_threshV_list = []
amps_list = []
RinVSS_list = []
t_list = []
IR_list = []

# for loop to change Na values
for na in na_factor:
    params['gnatbar_ichan2'] = 0.84*na
    hi, mid, lo, t, thresh = Obj.test_objective('thresh', params)
    hi_list.append(hi)
    mid_list.append(mid)
    lo_list.append(lo)

    IR, amps, RinVSS, threshV_list, t = Obj.test_objective('Rin', params)
    IR_list.append(IR)
    amps_list.append(amps)
    RinVSS_list.append(RinVSS)
    master_threshV_list.append(threshV_list)

#print("input resistance:", self.Rin)

_= plt.figure()
_ = plt.title('Input Resistance with Varying Na')
_=plt.plot(na_factor, IR_list)
plt.scatter(na_factor, IR_list)
plt.xlabel('Na Factor')
plt.ylabel('Input Resistance')
plt.show()

