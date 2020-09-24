import numpy as np
import time

etabins=np.array([-5.191, -4.889, -4.716, -4.538, -4.363, -4.191, -4.013, -3.839, -3.664, -3.489, -3.314, -3.139, -2.964,
         -2.853, -2.65, -2.5, -2.322, -2.172, -2.043, -1.93, -1.83, -1.74, -1.653, -1.566, -1.479, -1.392, -1.305,
         -1.218, -1.131, -1.044, -0.957, -0.879, -0.783, -0.696, -0.609, -0.522, -0.435, -0.348, -0.261, -0.174,
         -0.087, 0, 0.087, 0.174, 0.261, 0.348, 0.435, 0.522, 0.609, 0.696, 0.783, 0.879, 0.957, 1.044, 1.131,
         1.218, 1.305, 1.392, 1.479,1.566, 1.653, 1.74, 1.83, 1.93, 2.043, 2.172, 2.322, 2.5, 2.65, 2.853, 2.964,
         3.139, 3.314, 3.489, 3.664, 3.839, 4.013, 4.191, 4.363, 4.538, 4.716, 4.889, 5.191])
nEta = 82
etaC=0.5*(etabins[:-1]+etabins[1:])

def getWeights(data, MC):
    """get array weights for the dataset"""
    stime = time.time()
    MAXNPU=100
    h_weights, bins = np.histogram(data.array("mu"), bins=np.linspace(0, MAXNPU, 2*MAXNPU+1))
    h_weights_mc, bins = np.histogram(MC.array("mu"), bins=np.linspace(0, MAXNPU, 2*MAXNPU+1))
    ratio = np.where(h_weights_mc>0, h_weights/h_weights_mc, 0)
    weight = ratio/np.max(ratio)
    print('Time: ', time.time()-stime)
    return weight

def calcGeometricOffset(rCone, E, f_id, mu, mucut):
    E = (E.flatten()).reshape(len(E),nEta)[mu>mucut]
    f_id = (f_id.flatten()).reshape(len(f_id),nEta)[mu>mucut]
    if (len(f_id)!=len(E)):
        print("Error")
    area = 2* np.pi * (etabins[1:] - etabins[:-1])
    return E*f_id*np.pi*rCone*rCone / 255. / np.cosh(etaC) / area

def calcOffsetRC(rCone, E, f_id):
    E = (E.flatten()).reshape(len(E),nEta)
    f_id = (f_id.flatten()).reshape(len(f_id),nEta)
    offset_id = np.zeros((len(E), nEta))
    for ieta in range(len(etaC)):
        # first calculating area of the jetCone depending on etaC
        etaL_eta = etabins[:-1] - etaC[ieta]
        etaR_eta = etabins[1: ] - etaC[ieta]
        d1 = np.where(np.abs(etaL_eta)>np.abs(etaR_eta), np.abs(etaR_eta), np.abs(etaL_eta))
        d2 = np.where(np.abs(etaL_eta)>np.abs(etaR_eta), np.abs(etaL_eta), np.abs(etaR_eta))
        A1 = np.where(d1<=rCone, 0.5*rCone*rCone*(2*np.arccos(d1/rCone) - np.sin(2*np.arccos(d1/rCone))), 0 )
        A2 = np.where(d2<=rCone, 0.5*rCone*rCone*(2*np.arccos(d2/rCone) - np.sin(2*np.arccos(d2/rCone))), 0 )
        area = np.where((etaL_eta*etaR_eta>0), A1-A2, (np.pi*rCone*rCone)-A1-A2)
        area[(etaL_eta*etaR_eta>0) & (np.abs(etaL_eta)>rCone) & (np.abs(etaR_eta)>rCone)]=0
        
        x1 = np.where(np.abs(etaL_eta)>rCone, np.copysign(rCone,etaL_eta), etaL_eta)
        x2 = np.where(np.abs(etaR_eta)>rCone, np.copysign(rCone,etaR_eta), etaR_eta)
        dphi = np.sqrt(rCone**2 - (0.5*(x1+x2))**2)
        dphi[(etaL_eta*etaR_eta>0) & (np.abs(etaL_eta)>rCone) & (np.abs(etaR_eta)>rCone)]=0
        dphi = np.where(dphi > 1e-6, np.sin(dphi)/dphi, 1)
        
        A = area * dphi / (2* np.pi * (etabins[1:] - etabins[:-1]))/ np.cosh(etaC[ieta])
        #print(E.shape, A.shape, ((E*f_id).dot(A)).shape)
        
        offset_id[:,ieta] = ((E*f_id).dot(A))/255
    return offset_id