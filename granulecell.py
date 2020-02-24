# Granule cell class
from neuron import h
import numpy as np
h.load_file("importCell.hoc")

# The following layers are dictionaries whose keys are NEURON sections.
# Each entry of the dictionaries contains a list. The list contains
# the normalized distances along the section that lie within a layer.
def makeSecDict():
	SecList = {}
	SecList['soma'] = {}
	SecList['granuleCellLayer'] = {}
	SecList['innerThird'] = {}
	SecList['middleThird'] = {}
	SecList['outerThird'] = {}
	return SecList

def makeLayerDict(cell):
	LayerDict = {}
	LayerDict['Apical'] = {}
	LayerDict['Apical']['soma'] = cell.soma
	LayerDict['Apical']['granuleCellLayer'] = cell.granuleCellLayer
	LayerDict['Apical']['innerThird'] = cell.innerThird
	LayerDict['Apical']['middleThird'] = cell.middleThird
	LayerDict['Apical']['outerThird'] = cell.outerThird
	
	return LayerDict

# Since all the morphology is defined by HOC code, we need a pointer to the HOC object.
def loadMorph(morphFileName):
	param = {}
	param['c'] = h.mkcell(morphFileName)
	return param

# Initilize the list of synapses for the cell
def makeSynGroups(cell):
	SynGroups = {}
	SynGroups['AMPA'] = {}
	SynGroups['AMPA']['soma'] = []
	SynGroups['AMPA']['granuleCellLayer'] = []
	SynGroups['AMPA']['innerThird'] = []
	SynGroups['AMPA']['middleThird'] = []
	SynGroups['AMPA']['outerThird'] = []
	
	SynGroups['NMDA'] = {}
	SynGroups['NMDA']['soma'] = []
	SynGroups['NMDA']['granuleCellLayer'] = []
	SynGroups['NMDA']['innerThird'] = []
	SynGroups['NMDA']['middleThird'] = []
	SynGroups['NMDA']['outerThird'] = []
	
	SynGroups['GABA'] = {}
	SynGroups['GABA']['soma'] = []
	SynGroups['GABA']['granuleCellLayer'] = []
	SynGroups['GABA']['innerThird'] = []
	SynGroups['GABA']['middleThird'] = []
	SynGroups['GABA']['outerThird'] = []
	
	return SynGroups

# Defines the major axis of the morphology
def getNewAxis():
	new_axis = {}
	new_axis['new_axis'] = [np.cos(np.pi/2),0,np.sin(np.pi/2)]
	return new_axis

# Function to return the nseg resolution
def getNsegRes():
	return 20

# Function to return soma
def getSoma(cell):
	return cell.c.soma[0]

# Function to return the "center" of the morphology
# The reference point is set to the somatic location
def getCenter(soma):
	soma.push()
	center = np.array((h.x3d(0),h.y3d(0),h.z3d(0)))
	h.pop_section()
	return center

# Function to return to a dendrite lists organized by type (apical or basal)
def getDendTypeList(cell):
	dendTypeList = {}
	dendTypeList['Apical'] = getApicDend(cell)
	
	return dendTypeList

# Function to return apical dendrites
def getApicDend(cell):
	return cell.c.dend

# Function to return bounds of the layers
def getBounds(maxExtent):
	bounds = {}
	bounds['Apical'] = {}
	bounds['Apical']['soma'] = (0,0)
	bounds['Apical']['granuleCellLayer'] = (0,0.1*maxExtent['Apical'])
	bounds['Apical']['innerThird'] = (0.1*maxExtent['Apical'],0.3*maxExtent['Apical'])
	bounds['Apical']['middleThird'] = (0.3*maxExtent['Apical'],0.6*maxExtent['Apical'])
	bounds['Apical']['outerThird'] = (0.6*maxExtent['Apical'],maxExtent['Apical'])
	
	return bounds

# Function to make the lists containing the locations of the segments
# The list is organized as [ [x], [y], [z] ]
def makeSegLocDict(cell):
	SegLocDict = {}
	SegLocDict['Apical'] = {}
	SegLocDict['Apical']['soma'] = [ [], [], [] ]
	SegLocDict['Apical']['granuleCellLayer'] = [ [], [], [] ]
	SegLocDict['Apical']['innerThird'] = [ [], [], [] ]
	SegLocDict['Apical']['middleThird'] = [ [], [], [] ]
	SegLocDict['Apical']['outerThird'] = [ [], [], [] ]
	return SegLocDict

# Eliminate "double-booking" of segments to layers
# Layers closest to soma get priority for segment allocation
def separateSeg(cell):
	for sec in cell.soma:
		# GCL to inner third intersections
		intersect = list(set(cell.layerDict['Apical']['granuleCellLayer'][sec]).intersection(cell.layerDict['Apical']['innerThird'][sec]))
		for seg_pos in intersect:
			_=cell.innerThird[sec].pop(cell.innerThird[sec].index(seg_pos))
	
		# Inner third to middle third intersections
		intersect = list(set(cell.layerDict['Apical']['innerThird'][sec]).intersection(cell.layerDict['Apical']['middleThird'][sec]))
		for seg_pos in intersect:
			_=cell.layerDict['Apical']['middleThird'][sec].pop(cell.layerDict['Apical']['middleThird'][sec].index(seg_pos))
	
		# Middle third to outer third intersections
		intersect = list(set(cell.layerDict['Apical']['middleThird'][sec]).intersection(cell.layerDict['Apical']['outerThird'][sec]))
		for seg_pos in intersect:
			_=cell.layerDict['Apical']['outerThird'][sec].pop(cell.layerDict['Apical']['outerThird'][sec].index(seg_pos))

# Function to specify the biophysics of the cell
def getBiophysics(cell):
	cell.c.soma[0].L *= 2

	cell.RaMult = 1.0
	cell.CmMult = 9.8
	cell.a1 = 7.0				#gnatbar_ichan2
	cell.b1 = 2.25				#gkfbar_ichan2
	cell.c1 = 1.0				#gksbar_ichan2
	cell.d1 = 9.0				#gkabar_borgka
	cell.e1 = 1/1.36			#gncabar_nca
	cell.f1 = 0.5				#glcabar_lca
	cell.g1 = 2.0				#gcatbar_cat
	cell.h1 = 1.0				#gskbar_gskch
	cell.i1 = 1/5.0			#gkbar_cagk
	cell.j1 = 7.2538			#gl_ichan2
	cell.k1 = 1.0				#catau_ccanl
	cell.l1 = 1.0				#caiinf_ccanl

	# Now, insert the proper biophysics for each section.
	for sec in cell.c.all:
		sec.insert('ccanl')
		sec.catau_ccanl=10*cell.k1
		sec.caiinf_ccanl=5.0e-6*cell.l1
		sec.Ra=410*cell.RaMult

	for sec in cell.c.somatic:
		sec.insert('ichan2')
		sec.gnatbar_ichan2 = 0.12*cell.a1
		sec.gkfbar_ichan2=0.016*cell.b1
		sec.gksbar_ichan2=0.006*cell.c1
		sec.insert('borgka')
		sec.gkabar_borgka=0.001*cell.d1
		sec.insert('nca')
		sec.gncabar_nca=0.001*cell.e1
		sec.insert('lca')
		sec.glcabar_lca=0.005*cell.f1
		sec.insert('cat')
		sec.gcatbar_cat=0.000037*cell.g1
		sec.insert('gskch')
		sec.gskbar_gskch=0.001*cell.h1
		sec.insert('cagk')
		sec.gkbar_cagk=0.0006*cell.i1
		sec.gl_ichan2=0.00004*cell.j1
		sec.cm=1.0*cell.CmMult

	for sec in cell.c.dend:
		sec.insert('ichan2')
		sec.insert('nca')
		sec.insert('lca')
		sec.insert('cat')
		sec.insert('gskch')
		sec.insert('cagk')

	for sec in cell.granuleCellLayer:
		if len(cell.granuleCellLayer[sec]) > 0:
			for norm_dist in cell.granuleCellLayer[sec]:
				sec(norm_dist).gnatbar_ichan2 = 0.018*cell.a1
				sec(norm_dist).gkfbar_ichan2=0.004
				sec(norm_dist).gksbar_ichan2=0.006
				sec(norm_dist).gncabar_nca=0.003*cell.e1
				sec(norm_dist).glcabar_lca=0.0075
				sec(norm_dist).gcatbar_cat=0.000075
				sec(norm_dist).gskbar_gskch=0.0004
				sec(norm_dist).gkbar_cagk=0.0006*cell.i1
				sec(norm_dist).gl_ichan2=0.00004*cell.j1
				sec(norm_dist).cm=1.0*cell.CmMult

	for sec in cell.innerThird:
		if len(cell.innerThird[sec]) > 0:
			for norm_dist in cell.innerThird[sec]:			
				sec(norm_dist).gnatbar_ichan2 = 0.013*cell.a1
				sec(norm_dist).gkfbar_ichan2=0.004
				sec(norm_dist).gksbar_ichan2=0.006
				sec(norm_dist).gncabar_nca=0.001*cell.e1
				sec(norm_dist).glcabar_lca=0.0075
				sec(norm_dist).gcatbar_cat=0.00025
				sec(norm_dist).gskbar_gskch=0.0002
				sec(norm_dist).gkbar_cagk=0.001*cell.i1
				sec(norm_dist).gl_ichan2=0.000063*cell.j1
				sec(norm_dist).cm=1.6*cell.CmMult

	for sec in cell.middleThird:
		if len(cell.middleThird[sec]) > 0:
			for norm_dist in cell.middleThird[sec]:				
				sec(norm_dist).gnatbar_ichan2 = 0.008*cell.a1
				sec(norm_dist).gkfbar_ichan2=0.001
				sec(norm_dist).gksbar_ichan2=0.006
				sec(norm_dist).gncabar_nca=0.001*cell.e1
				sec(norm_dist).glcabar_lca=0.0005
				sec(norm_dist).gcatbar_cat=0.0005
				sec(norm_dist).gskbar_gskch=0.0
				sec(norm_dist).gkbar_cagk=0.0024*cell.i1
				sec(norm_dist).gl_ichan2=0.000063*cell.j1
				sec(norm_dist).cm=1.6*cell.CmMult

	for sec in cell.outerThird:
		if len(cell.outerThird[sec]) > 0:
			for norm_dist in cell.outerThird[sec]:			
				sec(norm_dist).gnatbar_ichan2 = 0.0*cell.a1
				sec(norm_dist).gkfbar_ichan2=0.001
				sec(norm_dist).gksbar_ichan2=0.008
				sec(norm_dist).gncabar_nca=0.001*cell.e1
				sec(norm_dist).glcabar_lca=0.0
				sec(norm_dist).gcatbar_cat=0.001
				sec(norm_dist).gskbar_gskch=0.0
				sec(norm_dist).gkbar_cagk=0.0024*cell.i1
				sec(norm_dist).gl_ichan2=0.000063*cell.j1
				sec(norm_dist).cm=1.6*cell.CmMult

	for sec in cell.c.all:
		sec.enat = 45
		sec.ekf = -90
		sec.eks = -90
		sec.ek = -90
		sec.elca = 130
		sec.etca = 130
		sec.esk = -90
		sec.el_ichan2 = -73
		sec.cao = 2

# Function to specify the biophysics of the reduced cell model
def getReducedBiophysics(cell):
	cell.soma.nseg = 1
	cell.soma.L = 11.6
	cell.soma.diam = 15
	
	cell.gcl1.nseg = 2 # 5
	cell.gcl1.L = 100
	cell.gcl1.diam = 1
	
	cell.gcl2.nseg = 2 # 5
	cell.gcl2.L = 100
	cell.gcl2.diam = 1
	
	cell.RaMult = 1.0
	cell.CmMult = 9.8
	cell.a1 = 7.0				#gnatbar_ichan2
	cell.b1 = 2.25				#gkfbar_ichan2
	cell.c1 = 1.0				#gksbar_ichan2
	cell.d1 = 9.0				#gkabar_borgka
	cell.e1 = 1/1.36			#gncabar_nca
	cell.f1 = 0.5				#glcabar_lca
	cell.g1 = 2.0				#gcatbar_cat
	cell.h1 = 1.0				#gskbar_gskch
	cell.i1 = 1/5.0			#gkbar_cagk
	cell.j1 = 7.2538			#gl_ichan2
	cell.k1 = 1.0				#catau_ccanl
	cell.l1 = 1.0				#caiinf_ccanl

	cell.soma.insert('ccanl')
	cell.soma.catau_ccanl=10*cell.k1
	cell.soma.caiinf_ccanl=5.0e-6*cell.l1
	cell.soma.Ra=410*cell.RaMult
	
	cell.soma.insert('ichan2')
	cell.soma.gnatbar_ichan2 = 0.12*cell.a1
	cell.soma.gkfbar_ichan2=0.016*cell.b1
	cell.soma.gksbar_ichan2=0.006*cell.c1
	cell.soma.insert('borgka')
	cell.soma.gkabar_borgka=0.001*cell.d1
	cell.soma.insert('nca')
	cell.soma.gncabar_nca=0.001*cell.e1
	cell.soma.insert('lca')
	cell.soma.glcabar_lca=0.005*cell.f1
	cell.soma.insert('cat')
	cell.soma.gcatbar_cat=0.000037*cell.g1
	cell.soma.insert('gskch')
	cell.soma.gskbar_gskch=0.001*cell.h1
	cell.soma.insert('cagk')
	cell.soma.gkbar_cagk=0.0006*cell.i1
	cell.soma.gl_ichan2=0.00004*cell.j1
	cell.soma.cm=1.0*cell.CmMult
	
	cell.gcl1.insert('ccanl')
	cell.gcl1.catau_ccanl=10*cell.k1
	cell.gcl1.caiinf_ccanl=5.0e-6*cell.l1
	cell.gcl1.Ra=410*cell.RaMult
	cell.gcl1.insert('ichan2')
	cell.gcl1.insert('nca')
	cell.gcl1.insert('lca')
	cell.gcl1.insert('cat')
	cell.gcl1.insert('gskch')
	cell.gcl1.insert('cagk')
	cell.gcl1.gnatbar_ichan2 = 0.013*cell.a1
	cell.gcl1.gkfbar_ichan2=0.004
	cell.gcl1.gksbar_ichan2=0.006
	cell.gcl1.gncabar_nca=0.003*cell.e1
	cell.gcl1.glcabar_lca=0.0075
	cell.gcl1.gcatbar_cat=0.000075
	cell.gcl1.gskbar_gskch=0.0004
	cell.gcl1.gkbar_cagk=0.001*cell.i1
	cell.gcl1.gl_ichan2=0.00004*cell.j1
	cell.gcl1.cm=1.0*cell.CmMult
	
	cell.gcl2.insert('ccanl')
	cell.gcl2.catau_ccanl=10*cell.k1
	cell.gcl2.caiinf_ccanl=5.0e-6*cell.l1
	cell.gcl2.Ra=410*cell.RaMult
	cell.gcl2.insert('ichan2')
	cell.gcl2.insert('nca')
	cell.gcl2.insert('lca')
	cell.gcl2.insert('cat')
	cell.gcl2.insert('gskch')
	cell.gcl2.insert('cagk')
	cell.gcl2.gnatbar_ichan2 = 0.013*cell.a1
	cell.gcl2.gkfbar_ichan2=0.004
	cell.gcl2.gksbar_ichan2=0.006
	cell.gcl2.gncabar_nca=0.003*cell.e1
	cell.gcl2.glcabar_lca=0.0075
	cell.gcl2.gcatbar_cat=0.000075
	cell.gcl2.gskbar_gskch=0.0004
	cell.gcl2.gkbar_cagk=0.001*cell.i1
	cell.gcl2.gl_ichan2=0.00004*cell.j1
	cell.gcl2.cm=1.0*cell.CmMult
	
	cell.soma.enat = 45
	cell.soma.ekf = -90
	cell.soma.eks = -90
	cell.soma.ek = -90
	cell.soma.elca = 130
	cell.soma.etca = 130
	cell.soma.esk = -90
	cell.soma.el_ichan2 = -73
	cell.soma.cao = 2
	cell.gcl1.enat = 45
	cell.gcl1.ekf = -90
	cell.gcl1.eks = -90
	cell.gcl1.ek = -90
	cell.gcl1.elca = 130
	cell.gcl1.etca = 130
	cell.gcl1.esk = -90
	cell.gcl1.el_ichan2 = -73
	cell.gcl1.cao = 2
	cell.gcl2.enat = 45
	cell.gcl2.ekf = -90
	cell.gcl2.eks = -90
	cell.gcl2.ek = -90
	cell.gcl2.elca = 130
	cell.gcl2.etca = 130
	cell.gcl2.esk = -90
	cell.gcl2.el_ichan2 = -73
	cell.gcl2.cao = 2

# Function to create a synapse at the chosen segment in a section
def createSyn(synvars,sec_choice,seg_choice):
	if synvars['type'] == "E2-NMDA2":
		syn = h.Exp2Syn(sec_choice(seg_choice))
		nmda = h.Exp2NMDA_Wang(sec_choice(seg_choice))
		nmda_flag = 1
	if synvars['type'] == "E2":	
		syn = h.Exp2Syn(sec_choice(seg_choice))
	if synvars['type'] == "E2_Prob":
		syn = h.E2_Prob(sec_choice(seg_choice))
		syn.P = synvars['P']
	if synvars['type'] == "E2_STP_Prob":
		syn = h.E2_STP_Prob(sec_choice(seg_choice))
	if synvars['type'] == "STDPE2":
		syn = h.STDPE2(sec_choice(seg_choice))
	if synvars['type'] == "STDPE2_Clo":
		syn = h.STDPE2_Clo(sec_choice(seg_choice))	
	if synvars['type'] == "STDPE2_STP"	:
		syn = h.STDPE2_STP(sec_choice(seg_choice))
	if synvars['type'] == "STDPE2_Prob":
		syn = h.STDPE2_Prob(sec_choice(seg_choice))
		syn.P = synvars['P']
	#initializes different variables depending on synapse		
	if (synvars['type'] == "STDPE2_STP")|(synvars['type'] == "E2_STP_Prob"):	
		syn.F1 = synvars['F1']		
	if  (synvars['type'] == "STDPE2_Clo" )|( synvars['type'] == "STDPE2_STP")|( synvars['type'] == "STDPE2")| (synvars['type'] == "STDPE2_Prob"):	
		syn.wmax = synvars['wmax']
		syn.wmin = synvars['wmin']
		syn.thresh = synvars['thresh']
	if  (synvars['type'] == "E2_Prob" )|( synvars['type'] == "E2_STP_Prob")|(synvars['type'] == "STDPE2_STP") | (synvars['type'] == "STDPE2_Prob"):
		h.use_mcell_ran4(1)   		
		syn.seed = self.ranGen.randint(1,4.295e9)
	syn.tau1 = 0.5
	syn.tau2 = 0.6
	syn.e = 0
	
	return syn

# Function to add synapses to the reduced cell model
def addReducedSynapses(cell):
	for syntype in cell.synGroups:
		# soma
		syn = createSyn(cell.synvars,cell.soma,0.5)
		cell.synGroups[syntype]['soma'].append(syn)
		
		# granuleCellLayer
		syn = createSyn(cell.synvars,cell.gcl1,0)
		cell.synGroups[syntype]['granuleCellLayer'].append(syn)
		syn = createSyn(cell.synvars,cell.gcl2,0)
		cell.synGroups[syntype]['granuleCellLayer'].append(syn)
		
		# innerThird
		syn = createSyn(cell.synvars,cell.gcl1,0)
		cell.synGroups[syntype]['innerThird'].append(syn)
		syn = createSyn(cell.synvars,cell.gcl2,0)
		cell.synGroups[syntype]['innerThird'].append(syn)
		
		# middleThird
		syn = createSyn(cell.synvars,cell.gcl1,1)
		cell.synGroups[syntype]['middleThird'].append(syn)
		syn = createSyn(cell.synvars,cell.gcl2,1)
		cell.synGroups[syntype]['middleThird'].append(syn)
		
		# outerThird
		syn = createSyn(cell.synvars,cell.gcl1,1)
		cell.synGroups[syntype]['outerThird'].append(syn)
		syn = createSyn(cell.synvars,cell.gcl2,1)
		cell.synGroups[syntype]['outerThird'].append(syn)

# End of file
