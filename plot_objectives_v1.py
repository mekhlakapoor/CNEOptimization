# Script to test single target_obj
from optFunc_granuleCell_v1 import Objectives
import numpy as np
import pickle
import matplotlib.pyplot as plt

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
params['gnatbar_ichan2'] = 0.84
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

# Run an objective
vs, t = Obj.test_objective('thresh', params)
#set 3 different voltages

# Plot data
for v in vs:
    plt.plot(t,v)

plt.show()

