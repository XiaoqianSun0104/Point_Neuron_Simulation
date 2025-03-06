'''
# connectivity.py
# Author: Xiaoqian Sun, 06/04/2024

# **** connectivity among neurons
#   recurrent connection within one subgroup (bi)
#   cross subgroups connection (bi)
#   external to one subgroup (one)
#   ...
# **** this should support differnet topology netowrk
'''


# Import Packages
import os 
import math
import random
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


import warnings
warnings.filterwarnings('ignore')



# connectivity matrix class
#-------------------------------------------------------------------------------------------------
class Connectivity(object):
    '''
    variables that could be accessed:
    N, Ne, Ni, paras{}, CM, CM_df, adjList, outgingCs, incomingCs, CMW, CMW_df
        - CM, CM_df   : connectivity matrix/dataframe
        - adjList     : adjcent list from CM, call .print_adjList() to print it out
        - outgingCs   : as a presynaptic neuron (A), connections from A to other neurons
        - incomingCs  : as a postsynaptic neuron (A), connections point to A
        - CMW, CMW_df : connectivity matrix with weights multiplied
    '''

    def __init__(self,
                 N, Ne, Ni, CM=None,
                 p_exc2exc=0.2, p_exc2inh=0.2, p_inh2exc=0.2, p_inh2inh=0.2, 
                 we2e_max=1.5, we2i_max=1.5, wi2e_max=3, wi2i_max=3,
                 w_exc2exc=0.1, w_exc2inh=0.1, w_inh2exc=0.1, w_inh2inh=0.1,
                 **kwargs
                 ):
        

        # main assign
        self.N=N; self.Ne=Ne; self.Ni=Ni; self.CM=CM
        self.CMWParas = {'p_exc2exc':p_exc2exc, 'p_exc2inh':p_exc2inh, 
                         'p_inh2exc':p_inh2exc, 'p_inh2inh':p_inh2inh,
                         'w_exc2exc':w_exc2exc, 'w_exc2inh':w_exc2inh, 
                         'w_inh2exc':w_inh2exc, 'w_inh2inh':w_inh2inh,
                         'we2e_max':we2e_max,   'we2i_max':we2i_max, 
                         'wi2e_max':wi2e_max,   'wi2i_max':wi2i_max
                         }
        
        # update
        CMWParasKeys=self.CMWParas.keys()
        for k,v in kwargs.items():
            if k in CMWParasKeys:
                self.paras[k] = v
            elif k=='CM':
                self.CM=v
            else:
                logging.warning('No key in Object.CMWParas named {0}'.format(k))
        
        
        # calculations and update values in CMWParas{}
        self.update_attr()


        

    def update_attr(self):
        '''
        once values in CMWParas got updated, we need to do some calculations 
        because those calculations rely on the correct+final CMWParas values
        wrap those in a function so that whenever self.set() change some values, we can 
        call self.update_attr() to update corresponding values 
        '''
        if type(self.CM) == np.ndarray: # if connectivity matrix is given
            self.CM_df = M_to_df(self.CM, self.Ne, self.Ni)
        else: # if not
            self.CM, self.CM_df = self.generate_connectivityMatrix()
        
        # get connectivity attributes
        self.adjList = adjMatrix_2_adjList(self.CM)
        self.outgoingCs = adjMatrix_2_adjList(self.CM)
        self.incomingCs = adjMatrix_2_adjList(self.CM.T)

        # add weights to connectivity (0/1)
        self.CMW, self.CMW_df = self.addWeights()

        # return (self)

    
    def generate_connectivityMatrix(self, ifVerbose=False):
    
        '''
        generate connectivity matrix
        each row is a sigle neuron and randomly select certian number of neurons to connect
        E.g.,
            - 1E, randomly select num=cee exc neurons to connect
            - 1E is the presynaptic neuron and it had cee outgoing connections to other exc neurons
        '''
        
        cee = int(self.CMWParas['p_exc2exc']*self.Ne)
        cie = int(self.CMWParas['p_exc2inh']*self.Ne)
        cei = int(self.CMWParas['p_inh2exc']*self.Ni)
        cii = int(self.CMWParas['p_inh2inh']*self.Ni)
        
        CM = np.zeros((self.N, self.N))
        for i in range(self.Ne):   
            CM[i, generate_randomInts(0, self.Ne-1, cee, i)] = 1
            CM[i, generate_randomInts(self.Ne, self.N-1, cie, i)] = 1

        for i in range(self.Ne, self.N):  
            CM[i, generate_randomInts(0, self.Ne-1, cei, i)] = 1
            CM[i, generate_randomInts(self.Ne, self.N-1, cii, i)] = 1
            
            
        # dataframe for better visualization
        cols = [str(i+1)+'E' for i in range(self.Ne)] + [str(i+1)+'I' for i in range(self.Ni)]
        CM_df = M_to_df(CM, self.Ne, self.Ni)
        
        if ifVerbose:
            print('There are', self.Ne, 'exc neurons and', self.Ni, 'inh neurons in the network')
            print('And we randomly choose pee⋅Ne(E2E)='+str(cee), 'pie⋅Ne(E2I)='+str(cie),
                'pei⋅Ni(I2E)='+str(cei), 'pii⋅Ni(I2I)'+str(cii), 'neurons to connect')
            print('That is: cee='+str(cee)+' | cie='+str(cie)+' | cei='+str(cei)+' | cii='+str(cii))
            
            
        return(CM, CM_df)
    
    def print_adjList(self):
        print_adjList(self.adjList)
    
    def addWeights(self):
        CMW, CMW_df = addWeight_2_connectivityMatrix(self.CM, self.CM_df, self.Ne, 
                                                    self.CMWParas['w_exc2exc'], self.CMWParas['w_exc2inh'], 
                                                    self.CMWParas['w_inh2exc'], self.CMWParas['w_inh2inh'])
        return(CMW, CMW_df)
    

    def get_keysValues(self):

        '''
        return all values stored in the object
            either direct attributes (obj.Nt)
            or dicts in the object (Obj.neuronPara)
        '''
        return (self.__dict__)
    
    def get_keys(self):
        '''
        return a list of attributes name, variables that can be accessed from the object
        '''
        return (self.__dict__.keys())
    
    def set(self, params_dict):
        """ 
        Set key-value pairs. E.g.,
            params_dict = {'V_excRevsal':16,  # in neuron.neuronPara{}
                           'Nt':6}    # neuron.Nt, direct attribute
            neuronObj.set(params_dict) to change corresponding values
        """

        for k, v in params_dict.items():
    
            if not hasattr(self, k):  # inside a {}
                attrNotExist_Flag = True
                
                # loop through subdic, locate key and change value
                for gr in list(self.__dict__.keys()):
                        d = getattr(self, gr)
                        if type(d) == dict:
                            if k in d:
                                d[k] = v
                                attrNotExist_Flag=False
                # if attrNotExist_Flag:
                #     logging.warning('No key in Connectivity Object named {0}'.format(k))
            else:
                setattr(self, k, v)

        self.update_attr()

    def get(self, key):
        """ 
        Get a value for a given key, which can either a direct attribute, or a key inside a dict (neuronPara)
        Raises an exception if no such group/key combination exists.
        """

        if not hasattr(self, key):  # inside a {}
            attrNotExist_Flag = True
            
            # loop through subdic, and get value
            for gr in list(self.__dict__.keys()):
                d = getattr(self, gr)
                if type(d) == dict:
                    if key in d:
                        attrNotExist_Flag=False
                        return (d[key])
            if attrNotExist_Flag:
                logging.warning('No key in Connectivity Object named {0}'.format(key))
        else:
            return(getattr(self, key))




# some general functions outside _connec class
#-------------------------------------------------------------------------------------------------
def generate_randomInts(start, stop, num, autapse):
    '''
    generate a list of integers (num) in a range (start, stop) without this number autapse
    e,g., crate a list of 3 ints in range (0, 9) without number 3: [1, 5, 7]
    this list could have duplicates
    
    Note that autapse in neuroscience means self-connection
    '''
    
    randInts = []
    while len(randInts) < num:
        
        randInt = random.randint(start, stop)
        if randInt != autapse:
            randInts.append(randInt)
        
    
    return (sorted(randInts))

def generate_randomConnection(preSyn_N, postSyn_N, connecP=0.5, weight=None, ifAutapse=True):
    '''
    simply:
    for each presynaptic neuron, randomly select connecP of postsynaptic neurons to connect with
    '''
    
    local_connectivity = np.zeros((preSyn_N, postSyn_N))
    numConnection = int(connecP * postSyn_N)
    
    for i in range(preSyn_N):
        if ifAutapse:
            local_connectivity[i, generate_randomInts(0, postSyn_N-1, numConnection, i)] = 1
        else:
            local_connectivity[i, generate_randomInts(0, postSyn_N-1, numConnection, None)] = 1
            
    return local_connectivity
    
def generate_normDistConnection(preSyn_N, postSyn_N, mean, std, ifShuffle=True):
    '''
    Simply:
    each presynaptic neuron connect with all postsynaptic neuron
    the weights form a normal distribution
    can choose the weight or not
    '''
    
    local_connectivity = np.zeros((preSyn_N, postSyn_N))
    
    for i in range(preSyn_N):
        weights = np.random.normal(mean, std, postSyn_N)
        if ifShuffle:
            np.random.shuffle(weights)
        
        local_connectivity[i] = weights
        
        
    # no negative weights
    local_connectivity[local_connectivity<0] = 0.05
    
    return (local_connectivity)
    
def M_to_df(connections, Ne, Ni, dataType=int):
    '''
    form the matrix to a dataframe for vetter visualization
    '''
    cols = [str(i+1)+'E' for i in range(Ne)] + [str(i+1)+'I' for i in range(Ni)]
    CM_df = pd.DataFrame(connections, columns=cols, index=cols).astype(dataType)
    return (CM_df)

def adjMatrix_2_adjList(connections):
    
        '''
        Cnvert an adjacen Adjacency Matrix to an Adjacency List
        So that we don't need to access the matrix every time
        we just pull out the postsynaptic neurons by key(presynaptic neuron) or vice versa

        '''
        # make sure adjMatrix is an array
        import numpy as np
        if type(connections) != np.ndarray:
            connections = np.array(connections)
        
        adjList = defaultdict(list)
        for i in range(len(connections)):
            
            adjList[i] = list(np.where(connections[i] != 0)[0])
            
            
        return (adjList)

def print_adjList(adjList):
    '''
    how online methods print out adjacency list, but this format doesn't make sense to me
    '''

    for i in adjList:
        print(i, end ="")
        
        for j in adjList[i]:
            print("-> {}".format(j), end =" ")
        
        print()
    
def get_connection_ExcInh(connections, neuronIndex, Ne):
    '''
    get connecting neurons to this neuron (neuronIndex) by type
    if the intput connections is incomingC, then we're picking presynaptic neurons
    if the input connection is outgoingC, then we're picking postsynaptic neurons that this neuron points to
    '''
    
    
    conn = np.array(connections[neuronIndex])
    
    idx_exc = list(conn[conn<Ne])
    idx_inh = list(conn[conn>=Ne])
    
    return (idx_exc, idx_inh)

def connecHalf_randomC(local_connec, connectP=None, connectN=None):
    
    '''
    previously, we arrange the connectivity matrix in :
        - rows of exc-inh, which are presynaptic neurons
        - cols of exc-inh, which are postsynaptic neurons
    and then we have p_exc2exc, p_exc2inh, p_inh2exc, p_inh2inh
    so for each prosyanptic neuron (row), we randomly select # postsynaptic neurons (col) and assign 1 as connection
    
    However, when work on fly olfactory structure, we have many layers of neurons and so a giant connectivity 
    table. The table will still be in row-pre and col-post format as input to simulation object.
    But during the process when you create local connections, you may just create connections from ORNs to PNs.
    
    This function is to solve this question. Basically, given 1 local matrix and 1 p, perform random selection
    and return the local connecitvity.
    
    '''
    pre_num, post_num = local_connec.shape
    
    if connectP:
        pre2post_c_num = int(post_num*connectP)
    
    if connectN:
        pre2post_c_num = int(connectN)

    if type(connectP)==type(None) and type(connectN)==type(None):
        raise ValueError('Must assign connection probability or number of connections')
        
    for i in range(pre_num):   
        local_connec[i, generate_randomInts(0, post_num-1, pre2post_c_num, None)] = 1
        
    
    return (local_connec)
    
    





def addWeight_2_connectivityMatrix(CM, CM_df, Ne, wee, wie, wei, wii):
    
    '''
    if there is a connection, modify that connection with 1*wij to reflect the connection strength
    '''
    
    CMW = CM.copy()
    CMW[0:Ne, 0:Ne] = CMW[0:Ne, 0:Ne]*wee
    CMW[0:Ne, Ne:] = CMW[0:Ne, Ne:]*wie
    CMW[Ne:, 0:Ne] = CMW[Ne:, 0:Ne]*wei
    CMW[Ne:, Ne:] = CMW[Ne:, Ne:]*wii
    
    # form a dataframe
    CMW_df = pd.DataFrame(CMW, columns=CM_df.columns, index=CM_df.index)
    
    return(CMW, CMW_df)

def CMW_upperBound(CMW, Ne, wee_max, wie_max, wei_max, wii_max):
    '''
    make sure the weights don't go beyond the upper bound
    this case can be called as 'saturation'
    '''
    
    CMW[0:Ne, 0:Ne][np.where(CMW[0:Ne, 0:Ne]>wee_max)] = wee_max
    CMW[0:Ne, Ne:][np.where(CMW[0:Ne, Ne:]>wie_max)] = wie_max

    CMW[Ne:, 0:Ne][np.where(CMW[Ne:, 0:Ne]>wei_max)] = wei_max
    CMW[Ne:, Ne:][np.where(CMW[Ne:, Ne:]>wii_max)] = wii_max
    
    return(CMW)

def get_chaningWeights(CMW_rlist, incomingCs, neuronIdx, Ne):
    '''
    here, neuronIdx is the postsynaptic neuron
    we try to get its presynaptic neurons' weights from CMW_rlist
    '''
    
    pre_exc, pre_inh = get_connection_ExcInh(incomingCs, neuronIdx, Ne)
    
    
    WE = []
    for exc in pre_exc:
        we = [cmw[exc,neuronIdx] for cmw in CMW_rlist]
        WE.append(we)
    
    WI = []
    for inh in pre_inh:
        wi = [cmw[inh,neuronIdx] for cmw in CMW_rlist]
        WI.append(wi)

    return(WE, WI)




