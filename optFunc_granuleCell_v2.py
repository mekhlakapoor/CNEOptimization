# Latest optFunc_granuleCell_v1.py - 2020-05-04
# Collection of functions outlining biophysical characteristics

# =====================================
# Importing Elements from Other Files
# =====================================
from neuron import h
import cell
import time as cookie
import numpy as np
import pickle
# import pylab as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
start = cookie.time()


# ==============
# Time Function
# ==============
def func(x, a, b, c, ):
    if b > 0:
        return 1e9 * np.ones(len(x))
    else:
        return a * np.exp(b * x) + c


# ======================================
# Main Class to Obtain Objective Values
# ======================================
class Objectives:
    # =======================
    # Initializing Variables
    # =======================
    def __init__(self, objectives,
                 weights):  # Initialization takes the dictionaries for default values of the objectives and values for the weights of those objectives
        self.synvars = {}
        self.synvars['type'] = 'E2'
        self.membrane_var = ['cm', 'Ra']
        self.objectives = objectives  # Dictionary with default values of objectives
        self.weights = weights  # Dictionary with default values of objectives
        self.createFuncDict()
        self.factor = 1
        self.spontaneous_flag = 0
        self.stop = 0

    # =============================
    # Objective Errors Calculation
    # =============================
    # Using normalized error to be able to compare multiple objectives
    def optimize(self, parameters):
        self.parameters = parameters
        obj_order = ['thresh', 'Rin', 'RMP', 'tau']  # ,'sfa','fAHP','sAHP','mAHP']

        # Output from Objective Functions
        self.props = {}

        # Check to see if neuron spontaneously generates action potential
        self.checkSpontaneous(self.parameters)
        objective_value = []
        self.all_returned_values = {}
        self.no_returned = {}

        # For each objective
        for prop in obj_order:
            # If the neuron does not spontaneously fire, run the objectives
            if not self.spontaneous_flag:
                objective_value = self.funcs[prop](self.parameters)
                self.props[prop] = objective_value[0]
                self.all_returned_values[prop] = objective_value

                # Number of values returned per objective
                self.no_returned[prop] = len(self.all_returned_values)

            # If the neuron spontaneously fires, return high errors for all objectives
            else:
                self.props[prop] = 1e9

        # Calculate errors between objectives' desired values (objectives) and values from optFunc (props)
        self.output = {}
        for prop in self.props:
            self.output[prop] = np.sum(
                self.weights[prop] * np.abs((self.props[prop] - self.objectives[prop]) / self.objectives[prop]))

        return [self.output, self.all_returned_values['thresh'], self.all_returned_values['Rin'],
                self.all_returned_values['RMP'], self.all_returned_values['tau']]  # self.no_returned,

    def test_objective(self, objective, parameters):
        self.parameters = parameters
        data = self.funcs[objective](parameters, debug=True)
        return data

    # Creates dictionary to obtain values from error functions
    def createFuncDict(self):
        self.funcs = {}
        self.funcs['thresh'] = self.getThreshold
        self.funcs['Rin'] = self.getRin
        self.funcs['RMP'] = self.getRMP
        self.funcs['tau'] = self.getMembraneTimeConstant
        # self.funcs['sfa'] = self.spikefreq
        # self.funcs['fAHP'] = self.getfAHP
        # self.funcs['mAHP'] = self.getmAHP
        # self.funcs['sAHP'] = self.getsAHP

    # Function that parameterizes the cell
    def parameterizeCell(self, cell, parameters):
        for par in parameters:
            '''
            # Original code to set slope values for dendritic parameters
            if par not in self.membrane_var:
                if 'slope' not in par:
                    setattr(cell.c.soma[0],par,parameters[par])
                    if 'gkabar' not in par:
                        secval = parameters[par]*(1-parameters[par+'_slope'])
                        if secval < 0:
                            secval = 0
                        for sec in cell.granuleCellLayer:
                            for seg in cell.granuleCellLayer[sec]:
                                    setattr(sec(seg),par,secval)

                        secval = parameters[par]*(1-2*parameters[par+'_slope'])
                        if secval < 0:
                            secval = 0
                        for sec in cell.innerThird:
                            for seg in cell.innerThird[sec]:
                                setattr(sec(seg),par,secval)

                        secval = parameters[par]*(1-3*parameters[par+'_slope'])
                        if secval < 0:
                            secval = 0
                        for sec in cell.middleThird:
                            for seg in cell.middleThird[sec]:
                                setattr(sec(seg),par,secval)

                        secval = parameters[par]*(1-4*parameters[par+'_slope'])
                        if secval < 0:
                            secval = 0
                        for sec in cell.outerThird:
                            for seg in cell.outerThird[sec]:
                                setattr(sec(seg),par,secval)
            '''

            # Testing code to modulate only somatic values
            if par == 'gnatbar_ichan2':
                # for dend in cell.dendTypeList:
                #    for sec in cell.dendTypeList[dend]:
                #        sec.gnatbar_ichan2 = parameters[par]
                cell.c.soma[0].gnatbar_ichan2 = parameters[par]

            elif par == 'glcabar_lca':
                # for dend in cell.dendTypeList:
                #    for sec in cell.dendTypeList[dend]:
                #        sec.glcabar_lca = parameters[par]
                cell.c.soma[0].glcabar_lca = parameters[par]

            elif par == 'gkbar_cagk':
                # for dend in cell.dendTypeList:
                #    for sec in cell.dendTypeList[dend]:
                #        sec.gkbar_cagk = parameters[par]
                cell.c.soma[0].gkbar_cagk = parameters[par]

            elif par == 'gkfbar_ichan2':
                # for dend in cell.dendTypeList:
                #    for sec in cell.dendTypeList[dend]:
                #        sec.gkfbar_ichan2 = parameters[par]
                cell.c.soma[0].gkfbar_ichan2 = parameters[par]

            elif par == 'gksbar_ichan2':
                # for dend in cell.dendTypeList:
                #    for sec in cell.dendTypeList[dend]:
                #        sec.gksbar_ichan2 = parameters[par]
                cell.c.soma[0].gksbar_ichan2 = parameters[par]

            elif par == 'gkabar_borgka':
                # for dend in cell.dendTypeList:
                #    for sec in cell.dendTypeList[dend]:
                #        sec.gkabar_borgka = parameters[par]
                cell.c.soma[0].gkabar_borgka = parameters[par]

            elif par == 'gl_ichan2':
                # for dend in cell.dendTypeList:
                #    for sec in cell.dendTypeList[dend]:
                #        sec.gl_ichan2 = parameters[par]
                cell.c.soma[0].gl_ichan2 = parameters[par]

            elif par == 'gcatbar_cat':
                # for dend in cell.dendTypeList:
                #    for sec in cell.dendTypeList[dend]:
                #        sec.gcatbar_cat = parameters[par]
                cell.c.soma[0].gcatbar_cat = parameters[par]

            elif par == 'gncabar_nca':
                # for dend in cell.dendTypeList:
                #    for sec in cell.dendTypeList[dend]:
                #        sec.gncabar_nca = parameters[par]
                cell.c.soma[0].gncabar_nca = parameters[par]

            elif par == 'gskbar_gskch':
                # for dend in cell.dendTypeList:
                #    for sec in cell.dendTypeList[dend]:
                #        sec.gskbar_gskch = parameters[par]
                cell.c.soma[0].gskbar_gskch = parameters[par]

            elif par == 'cm':
                # for dend in cell.dendTypeList:
                #    for sec in cell.dendTypeList[dend]:
                #        sec.cm = parameters[par]
                cell.c.soma[0].cm = parameters[par]

            elif par == 'Ra':
                # for dend in cell.dendTypeList:
                #    for sec in cell.dendTypeList[dend]:
                #        sec.Ra = parameters[par]
                cell.c.soma[0].Ra = parameters[par]

    # Exponential function for fitting
    def expFunc(self, x, a, b, c):
        if b > 0:
            return 1e9 * np.ones(len(x))
        else:
            return a * np.exp(b * x) + c

    # Function to see if neuron spontaneously generates an action potential
    def checkSpontaneous(self, parameters):
        h.load_file('stdrun.hoc')
        h.load_file('negative_init.hoc')
        tstop = 500
        dt = 0.025
        h.stdinit()
        h.celsius = 37.0
        h.tstop = tstop
        h.v_init = -65

        # Instantiate and parameterize cells
        testCell = cell.Cell(0, (0, 0), self.synvars, 'granulecell', 'output0_updated.swc')
        self.parameterizeCell(testCell, parameters)

        # Instrument cells
        tvec = h.Vector()
        nc = testCell.connect_pre(None, 0, 0)
        nc.record(tvec)

        # Run simulation
        h.run()
        if len(tvec) > 0:
            self.spontaneous_flag = 1

    # ===================
    # Threshold Function
    # ===================
    # Finds an amplitude that elicits a spike and decreases the duration of the pulse until a spike is no longer generated.
    # Simulation parameters
    def getThreshold(self, parameters, debug = False):
        synvars = {}
        synvars['type'] = 'E2'
        h.load_file('stdrun.hoc')
        h.load_file('negative_init.hoc')
        tstop = 600
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
        self.parameterizeCell(cell_hi,parameters)
        self.parameterizeCell(cell_mid, parameters)
        self.parameterizeCell(cell_lo, parameters)

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
        amp_mid = 0.5 * (amp_hi + amp_lo)
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
            #print('current amp_hi:', amp_hi)
            #print('current amp_mid:', amp_mid)
            #print('current amp_lo:', amp_lo)
            amp_mid_old = amp_mid
            # create cell_mid
            cell_mid = cell.Cell(0, (0, 0), synvars, 'granulecell', 'output0_updated.swc')
            self.parameterizeCell(cell_mid, parameters)

            # attach current clamp
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

            amp_mid = 0.5 * (amp_hi + amp_lo)
            e_rel = np.abs(amp_mid - amp_mid_old) / amp_mid

        self.thresh = max(somaticV_lo)
        self.APrepolar = somaticV_mid[numloops]
        self.threshAmp = amp_mid
        if debug:
             return np.array(somaticV_hi), np.array(somaticV_mid) ,np.array(somaticV_lo), np.array(t), amp_mid
        else:
            return np.array([self.thresh, self.APrepolar])

    # ==========================
    # Input Resistance Function
    # ==========================
    # Calculates input resistance of soma
    def getRin(self, parameters, debug=False):
        if not self.stop:
            ###################
            # Rin Experiments #
            ###################
            amps_list = np.linspace(-0.05, 0.05, 11)
            amps = self.threshAmp * amps_list #self.threshAmp = amp_mid from thresh objective

            # Simulation parameters
            h.load_file('stdrun.hoc')
            h.load_file('negative_init.hoc')
            tstop = 600
            dt = 0.025
            h.stdinit()
            h.celsius = 37.0
            h.tstop = tstop
            h.v_init = -65

            # Instantiate and parameterize cells and inputs for experiment
            cells = []
            stim_list = []
            threshV_list = []

            for ii in range(len(amps)):
                # create 5 instrument cells and organize them in a list
                testCell = cell.Cell(0, (0, 0), self.synvars, 'granulecell', 'output0_updated.swc')
                self.parameterizeCell(testCell,parameters)
                cells.append(testCell)

                # store the stimulus amplitudes in a list for each cell
                stim = h.IClamp(0.5, sec=cells[ii].c.soma[0])
                stim_list.append(stim)
                stim_list[ii].dur = 600
                stim_list[ii].delay = 0
                stim_list[ii].amp = amps_list[ii]

                # record and store the thresholds for each of these cells
                RinVs = h.Vector()
                RinVs.record(cells[ii].c.soma[0](0.5)._ref_v)
                threshV_list.append(RinVs)

            # a list for time, to make checking easier
            t = h.Vector()
            t.record(h._ref_t)

            # Run simulation
            h.run()

            # saving threshV
            RinVSS = [] #voltage steady state
            for jj in range(len(amps)):
                RinVSS.append(threshV_list[jj][-2]) #steadystate_arr

            # Conversions
            amps = np.array(amps) / (10 ** 9)  # nanoamps to amps
            RinVSS = np.array(RinVSS) / (10 ** 3)  # millivolts to volts

            # input resistance = slope
            # append an input resistance list for these
            self.Rin, intercept, r_value, p_value, std_err = stats.linregress(amps, RinVSS)
            self.Rin = self.Rin / (10 ** 6)
            if debug:
                return [self.Rin, amps, RinVSS, threshV_list, t]
            else:
                return np.array([self.Rin])
        else:
            return 1e9

    # ====================================
    # Resting Membrane Potential Function
    # ====================================
    def getRMP(self, parameters, debug=False):
        if not self.stop:
            # Simulation parameters
            h.load_file('stdrun.hoc')
            h.load_file('negative_init.hoc')
            tstop = 100
            dt = 0.025
            h.stdinit()
            h.celsius = 37.0
            h.tstop = tstop
            h.v_init = -65

            # Instantiate and parameterize cells
            testCell = cell.Cell(0, (0, 0), self.synvars, 'granulecell', 'output0_updated.swc')
            self.parameterizeCell(testCell, parameters)

            # Instrument cells
            self.RMPv = h.Vector()
            self.RMPv.record(testCell.c.soma[0](0.5)._ref_v)

            # Run simulation
            h.run()

            # Get RMP
            self.RMP = self.RMPv[-1]
            if debug:
                return [np.array([self.RMP]), np.array([self.RMPv])]
            else:
                return np.array([self.RMP])
        else:
            return 1e9

    # ========================================
    # Resting Membrane Time Constant Function
    # ========================================
    # Calculates somatic membrane time constant
    def getMembraneTimeConstant(self, parameters, debug=False):
        if not self.stop:
            ######################################
            # Membrane Time Constant Experiments #
            ######################################
            # Simulation parameters
            h.load_file('stdrun.hoc')
            h.load_file('negative_init.hoc')
            tstop = self.subthresh_dur + 300
            dt = 0.025
            h.stdinit()
            h.celsius = 37.0
            h.tstop = tstop
            h.v_init = -65

            # Instantiate and parameterize cells
            testCell = cell.Cell(0, (0, 0), self.synvars, 'granulecell', 'output0_updated.swc')
            self.parameterizeCell(testCell, parameters)

            # Make sure the cell doesn't spike
            # Create inputs for experiments
            stim = h.IClamp(0.5, sec=testCell.c.soma[0])
            stim.amp = self.threshAmp  # amp at which neuron no longer spikes
            stim.dur = self.subthresh_dur  # subthresh is duration where neuron no longer spikes
            stim.delay = 0

            # Instrument cells
            self.tauV = h.Vector()
            self.tauV.record(testCell.c.soma[0](0.5)._ref_v)
            t = h.Vector()
            t.record(h._ref_t)
            tvec = h.Vector()
            nc = testCell.connect_pre(None, 0, 0)
            nc.record(tvec)

            # Run simulation
            h.run()

            # Calculate membrane time constant
            self.tauV = np.array(self.tauV)
            t = np.array(t)
            t_idx = t >= self.subthresh_dur
            t = t[t_idx] - self.subthresh_dur
            v = self.tauV[t_idx]
            try:
                popt, pcov = curve_fit(self.expFunc, t, v, p0=(1, -0.1, -60))
                self.tau = -1 / popt[1]
            except RuntimeError:
                self.tau = 1000

            if debug:
                return [np.array([self.tau]), v, t, popt]
            else:
                return np.array([self.tau])

        else:
            return 1e9

    # ====================================
    # Spike Frequency Adaptation Function
    # ====================================
    def spikefreq(self, parameters, debug=False):
        if not self.stop:
            def func(x, a, b, c):  # for the time function
                if b > 0:
                    return 1e9 * np.ones(len(x))
                else:
                    return a * np.exp(b * x) + c

            # Simulation parameters
            synvars = {}
            synvars['type'] = 'E2'
            h.load_file('stdrun.hoc')
            h.load_file('negative_init.hoc')
            tstop = 500  # ms
            dt = 0.025
            h.stdinit()
            h.celsius = 37.0
            h.tstop = tstop
            h.v_init = -65

            # Instantiate and parameterize cells
            testCell = cell.Cell(0, (0, 0), synvars, 'granulecell', 'output0_updated.swc')
            self.parameterizeCell(testCell, parameters)

            # Create inputs for experiments
            stim = h.IClamp(0.5, sec=testCell.c.soma[0])

            stim.dur = tstop
            stim.delay = 0  # When current clamp is applied

            ratio = []  # Initializing ratio between initial frequency and steady state frequency

            stim.amp = ((5.61115145735) * (self.thresh - self.RMP)) / (self.Rin)  # 0.783 nA yields 0.31 ratio
            # Instrument cells -> at soma
            spikes_sfa = h.Vector()  # Neuron HOC vector data type
            spikes_sfa.record(testCell.c.soma[0](0.5)._ref_v)  # Record voltage at soma
            t = h.Vector()
            t.record(h._ref_t)  # Records the time
            tvec = h.Vector()  # Instantiates tvec vector

            ncThreshold = 0.0  # The threshold NEURON uses to determine if/when the pre-synaptic cell generates an action potential. Units: mV

            nc = testCell.connect_pre(None, 0,
                                      0)  # Connect netcon to a vector to record from (No synapses, weight of 0, delay of 0)
            spike_times_sfa = nc.record(tvec)  # Records spike times

            # print(tvec)
            # spike_freq = h.Vector()
            # spike_freq[:] = np.divide(1, spike_time)
            # spike_freq[:] = [x / spike_time for x in spike_freq]

            # Run simulation
            h.run()

            # Obtaining the interspike interval
            isi = []
            for s in range(len(tvec) - 1):
                isi.append(tvec[s + 1] - tvec[s])

            f = []
            # for i in range(len(isi)):
            #    f.append(1000/isi[i])

            # f = [1000/isi[i] for i in range(len(isi))]

            f = 1000 / np.array(isi)
            # print("Took {0:.2f} seconds".format(cookie.time()-start))
            # print("Amp =", stim.amp)
            # if isi != []:
            # print("ISI = ", isi[0])
            if f != []:  # Following commented out here but utilized in individual objective
                # SS_ratio = ((f[-2]-f[-1])/(f[-1]))*100 # Find difference between last two frequencies
                # if SS_ratio > 1: # If steady state difference is too large
                # print("Steady state frequency not reached.")
                #    tstop = tstop + 1000 # Extend time to allow steady state frequency to be reached
                # else:
                # print("f_0 = ", f[0])
                # print("f_ss = ", f[-1])
                # print("ratio = ", (f[0]-f[-1])/f[0])
                ratio = (f[0] - f[-1]) / f[
                    0]  # Appends ratio between steady state and initial frequencies to ratio vector
                # ratio2.append(f[-1]/f[0])
                # amps_x.append(stim.amp)
                # f_0.append(f[0])
                # f_ss.append(f[-1])
                # volts.append(list(spikes))
                return np.array([ratio])
            else:
                return 1e9
                print(ratio)

        else:
            return 1e9

    # =====================================
    # Fast Afterhyperpolarization Function
    # =====================================
    def getfAHP(self, parameters, debug=False):
        if not self.stop:
            def func(x, a, b, c):  # for the time function
                if b > 0:
                    return 1e9 * np.ones(len(x))
                else:
                    return a * np.exp(b * x) + c

            # Simulation parameters
            synvars = {}
            synvars['type'] = 'E2'
            h.load_file('stdrun.hoc')
            h.load_file('negative_init.hoc')
            tstop = 1000  # 100ms for one AP
            dt = 0.025
            h.stdinit()
            h.celsius = 37.0
            h.tstop = tstop
            h.v_init = -65

            # Instantiate and parameterize cells
            testCell = cell.Cell(0, (0, 0), synvars, 'granulecell', 'output0_updated.swc')
            self.parameterizeCell(testCell, parameters)

            # Create inputs for experiments
            stim = h.IClamp(0.5, sec=testCell.dendTypeList['Apical'][1])
            stim.amp = self.threshAmp  # from find_thresh
            stim.dur = self.subthresh_dur + 1  # from find_thresh
            stim.delay = 0  # When current clamp is applied

            # Instrument cells -> at soma
            spikes_fahp = h.Vector()  # Neuron HOC vector data type
            spikes_fahp.record(testCell.c.soma[0](0.5)._ref_v)  # Record voltage at soma
            t = h.Vector()

            t.record(h._ref_t)  # Records the time
            tvec = h.Vector()  # Instantiates tvec vector

            ncThreshold = 0.0  # The threshold NEURON uses to determine if/when the pre-synaptic cell generates an action potential. Units: mV

            nc = testCell.connect_pre(None, 0,
                                      0)  # Connect netcon to a vector to record from (No synapses, weight of 0, delay of 0)
            spike_time_fahp = nc.record(tvec)  # Records spike times

            # Finding AHP (global minimum after single AP)
            i = 0
            j = 1

            # Run simulation
            h.run()

            t_max = np.argmax(spikes_fahp)
            t_max_time = t_max * dt - stim.delay
            # print("Time and amplitude of action potential: ", t_max_time, ",", spikes[t_max])

            t_min = np.argmin(spikes_fahp)
            t_min_time = t_min * dt - stim.delay
            # print("Time and amplitude of minimum: ", t_min_time, ",", spikes[t_min])
            fAHP_thresh = (spikes_fahp[t_min]) - (self.threshAmp)
            fAHP_RMP = (spikes_fahp[t_min]) - (self.RMP)
            return fAHP_RMP
        else:
            return 1e9

    # =======================================
    # Medium Afterhyperpolarization Function
    # =======================================
    def getmAHP(self, parameters, debug=False):
        if not self.stop:
            def func(x, a, b, c):  # for the time function
                if b > 0:
                    return 1e9 * np.ones(len(x))
                else:
                    return a * np.exp(b * x) + c

            # Simulation parameters
            synvars = {}
            synvars['type'] = 'E2'
            h.load_file('stdrun.hoc')
            # h.load_file('negative_init.hoc')
            tstop = 1500  # 100ms for one AP
            dt = 0.025
            h.stdinit()
            h.celsius = 37.0
            h.tstop = tstop
            h.v_init = -65

            # Need to make sure the amplitude elicits a spike
            addon = 0

            # Using count to find cases where the cell is
            # unable to fire an action potential
            count = 0
            spikes_mahp = []

            while spikes_mahp == [] or (spikes_mahp[-1] < -62.5) or (spikes_mahp[-1] >= -61.5):
                # Instantiate and parameterize cells
                testCell = cell.Cell(0, (0, 0), synvars, 'granulecell', 'output0_updated.swc')
                # self.parameterizeCell(testCell,parameters)

                # Create second IClamp to hold membrane potential at -62mV
                stim2 = h.IClamp(0.5, sec=testCell.dendTypeList['Apical'][1])
                stim2.amp = (-50 - self.RMP) / (self.Rin) + addon  # (62 - RMP)/Rin nA
                stim2.dur = tstop  # Holds MP at -62 for entire duration of experiment
                stim2.delay = 0  # When current clamp is applied

                # Instrument cells -> at soma
                spikes_mahp = h.Vector()  # Neuron HOC vector data type
                spikes_mahp.record(testCell.c.soma[0](0.5)._ref_v)  # Record voltage at soma
                t = h.Vector()

                t.record(h._ref_t)  # Records the time
                tvec = h.Vector()  # Instantiates tvec vector

                ncThreshold = 0.0  # The threshold NEURON uses to determine if/when the pre-synaptic cell generates an action potential. Units: mV

                nc = testCell.connect_pre(None, 0,
                                          0)  # Connect netcon to a vector to record from (No synapses, weight of 0, delay of 0)

                # Run simulation
                h.run()

                if (spikes_mahp[-1] < -62.5) or (spikes_mahp[-1] >= -61.5):
                    addon += 0.001  # Increases amplitude by 0.1 if holding potential of ~-62mV is not reached

                count += 1
                # if count > 10:
                #    break

            # Holding voltage
            holding_voltage = list(spikes_mahp)

            # Create inputs for experiments
            stim = h.IClamp(0.5, sec=testCell.dendTypeList['Apical'][1])
            stim.amp = 0.85  # 1.55 #0.26-0.3 ->  range to elicit >2 spikes in train
            stim.dur = 500  # 106.00000000011622 from find thresh, tstop for multiple
            stim.delay = 100  # When current clamp is applied

            # Create second IClamp to hold membrane potential at -62mV
            stim2 = h.IClamp(0.5, sec=testCell.dendTypeList['Apical'][1])
            stim2.amp = (-50 - (-73.4818762779)) / (169.414948831) + addon  # (62 - RMP)/Rin nA
            stim2.dur = tstop  # Holds MP at -62 for entire duration of experiment
            stim2.delay = 0  # When current clamp is applied

            # Instrument cells -> at soma
            spikes_mahp2 = h.Vector()  # Neuron HOC vector data type
            spikes_mahp2.record(testCell.c.soma[0](0.5)._ref_v)  # Record voltage at soma
            t2 = h.Vector()

            t2.record(h._ref_t)  # Records the time
            tvec2 = h.Vector()  # Instantiates tvec vector

            ncThreshold = 0.0  # The threshold NEURON uses to determine if/when the pre-synaptic cell generates an action potential. Units: mV

            nc = testCell.connect_pre(None, 0,
                                      0)  # Connect netcon to a vector to record from (No synapses, weight of 0, delay of 0)
            spike_time_mahp = nc.record(tvec)  # Records spike times

            # Run simulation
            h.run()

            t_max = np.argmax(spikes_mahp2)
            t_max_time = t_max * dt - stim.delay
            # print("Time and amplitude of action potential: ", t_max_time, ",",) spikes_mahp2[t_max]

            t_min = np.argmin(spikes_mahp2)
            t_min_time = t_min * dt - stim.delay

            mAHP_thresh = (spikes_mahp2[t_min]) - (self.threshAmp)
            mAHP_RMP = (spikes_mahp2[t_min]) - (self.RMP)
            return mAHP_RMP
        else:
            return 1e9

# End of file