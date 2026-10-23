#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VSM - Version 2.0

   Volcanic and Seismic source Modelling

   Authors:
        Elisa Trasatti, elisa.trasatti@ingv.it
        Francesca Silverii, francesca.silverii@ingv.it
        Istituto Nazionale di Geofisica e Vulcanologia - Rome (Italy)

   Last update:  April 2026

   License:  E. Trasatti and F. Silverii covered by GNU-GPL License

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

      * Redistributions of source code must retain the above copyright 
        notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright 
 	    notice, this list of conditions and the following disclaimer in 
 	    the documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPY RIGHT OWNER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
   
*******************************************************************************

   BIBLIOGRAPHY

   Amoruso, A., Crescentini, L. (2013). Analytical models of volcanic 
       ellipsoidal expansion sources. Ann. Geophys.
       https://doi.org/10.4401/ag-6441
   Battaglia, M., Cervelli, P.F., Murray, J.R. (2013). Modeling Crustal 
       Deformation near Active Faults and Volcanic Centers—A Catalog of 
       Deformation Models, Book 13, Volcanic Monitoring, USGS.
   Belardinelli, M.E., Bonafede, M., and Nespoli, M. (2019). Stress 
       heterogeneities and failure mechanisms induced by temperature and pore-
       pressure increase in volcanic regions. Earth Plan. Sci. Lett. , 525, 
       115765. doi:https://doi.org/10.1016/j.epsl.2019.115765
   Mantiloni, L., Nespoli, M., Belardinelli, M.E., and Bonafede, M. (2020). 
       Deformation and stress in hydrothermal regions: The case of a disk-shaped 
       inclusion in a half-space. J. Volcanol. Geotherm. Res., 403, 107011.
       doi:10.1016/j.jvolgeores.2020.107011
   Nespoli, M., Belardinelli, M.E., and Bonafede, M. (2021). Stress and 
       deformation induced in layered media by cylindrical thermo-poro-elastic 
       sources: An application to Campi Flegrei (Italy). J. Volcanol. Geotherm. 
       Res., 415, 107269. doi:10.1016/j.jvolgeores.2021
   Okada, Y. (1985). Surface deformation due to shear and tensile faults in a 
       half-space. B. Seismol. Soc. Am. 75, 1135-1154.
       
*******************************************************************************

    This file contains several utilities to perform post-processing.
    Uncomment to use one of them, or use VSM_utilities.ipynb.
    
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

#import sys
#sys.path.append('path-to-VSM-folder')
import VSM

def volume_var_mctigue(dpmu,r,d):
    r3 = r**3
    rd4 = (r/d)**4
    dVolume = np.pi*r3*dpmu*(1+rd4)
    return dVolume

def volume_var_yang(dpmu,s_axis_max,ratio):
    if(ratio > 1): ratio = 1./ratio
    volume = 4./3.*np.pi*s_axis_max**3*ratio*ratio
    f = ratio*ratio/3. - 0.7*ratio + 1.37
    dVolume = volume*dpmu*3./4.*f
    return dVolume

def volume_var_penny(dpmu,r,nu):
    r3 = r**3
    dVolume = 8./3.*(1-nu)*r3*dpmu
    return dVolume

def fault_vertices(TLC,l,w,strike,dip):
    s  = np.deg2rad(strike)
    cs = np.cos(s)
    ss = np.sin(s)
    d  = np.deg2rad(dip)
    cd = np.cos(d)
    sd = np.sin(d)
    vertices = np.zeros((4,3))
    vertices[0] = TLC
    trace = np.zeros((2,3))
    
    # Bottom Left Corner
    coox = TLC[0] + w*cd*cs
    cooy = TLC[1] - w*cd*ss
    cooz = TLC[2] + w*sd
    vertices[1] = [coox,cooy,cooz]
	
    # Bottom Right Corner
    coox = vertices[1,0] + l*ss
    cooy = vertices[1,1] + l*cs
    cooz = vertices[1,2]
    vertices[2] = [coox,cooy,cooz]

    # Up Right Corner
    coox = TLC[0] + l*ss
    cooy = TLC[1] + l*cs
    cooz = TLC[2]
    vertices[3] = [coox,cooy,cooz]

    # Trace Left
    ww = TLC[2]/sd
    coox = TLC[0] - ww*cd*cs
    cooy = TLC[1] + ww*cd*ss
    cooz = 0.0
    trace[0] = [coox,cooy,cooz]
    
    # Trace Right
    ww = TLC[2]/sd
    coox = vertices[3,0] - ww*cd*cs
    cooy = vertices[3,1] + ww*cd*ss
    cooz = 0.0
    trace[1] = [coox,cooy,cooz]

    return vertices,trace

def TPE_potency(H,alpha, dp, dT):
    e0 = dp/(3*H)+alpha*dT/3
    return e0
    
def TPE_H(K,Ks):
    H = 1/(1/K - 1/Ks)
    return H

def TPE_dp(e0, alpha, H, dT):
    dp1 = e0*3*H
    dp2 = (e0 - alpha*dT/3)*3*H
    return dp1, dp2

def TPE_volume_var_cylinder(e0,r,h,ni):
    volume = np.pi * r ** 2 * h
    e1 = (1 + ni)/(1 - ni)*e0
    dVolume = e1*volume
    return volume, dVolume
    
def TPE_volume_var_sphere(e0,r,ni):
    volume = 4/3 * np.pi * r ** 3
    e1 = (1 + ni)/(1 - ni)*e0
    dVolume = e1*volume
    return volume, dVolume
        

if(__name__ == "__main__"):

    
#----------------------------------------------------------------------
# FAULT or DIKE VERTICES from the TOP LEFT CORNER (TLC)
    '''
    TLC = [7.48873e+05,9.82e+06,2.21526e+03]
    length = 9000.
    width  = 4000.
    strike = 180.
    dip    = 83.
    
    vert,trace = fault_vertices(TLC,length,width,strike,dip)
    print('Fault vertices ordered as TLC - BLC - BRC - TRC',vert)
    print('Trace from left to right', trace)
    '''
    
    
#----------------------------------------------------------------------
# VOLUME VARIATION SPHERE (MCTIGUE)
    '''
    a = 600
    dP_mu = 7.47e-3
    depth = 2612
    
    dVol = volume_var_mctigue(dP_mu,a,depth)
    print('Volume variation of the finite volume sphere',dVol)
    '''
    
    
#----------------------------------------------------------------------
# VOLUME VARIATION SPHEROID (YANG)
    '''
    a = 8600
    dP_mu = 1e-3
    ratio = 0.33
    dVol,f = volume_var_yang(dP_mu,a,ratio)
    print('Volume variation of the finite volume spheroid',dVol/1e6, '1e6 m3')
    '''

#----------------------------------------------------------------------
# VOLUME VARIATION SILL (FIALKO) in point-source approximation
# holds for depth/radius > 2 or 2.5
    '''
    r = 1000
    dP_mu = 3.34987e-3
    dVol = volume_var_penny(dP_mu,r,0.25)
    print('Volume variation of the penny-shaped crack',dVol/1e6,' 1e6 m3')
    '''

#----------------------------------------------------------------------
# TPE - Estimate potency assuming others
    '''
    H = 1e9         # poro-elastic parameter (Pa)
    alpha = 3e-5    # thermal expansion coefficient (1/K)
    dp = 10e6       # pore pressure change of the inclusion (Pa)
    dT = 50         # Temperature change of the inclusion (K)
    
    e0 = TPE_potency(H,alpha, dp, dT)
    
    print('\nAssuming H', H/1e9, 'GPa, alpha', alpha, '1/K, dp', dp/1e6, 'MPa, dT', dT,'K')
    print('Potency: ', e0)
    '''

#----------------------------------------------------------------------
# TPE - Estimate H assuming others
    '''
    nu = 0.25       # Poisson coefficient
    mu = 1e9        # shear modulus (Pa)
    
    K = 0.1e9 # (2*mu*(1+nu))/(3*(1-2*nu)) # Drained isothermal bulk modulus of the medium (Pa)
    print('\nK calculated', 2*mu*(1+nu)/(3*(1-2*nu))/1e9)
    Ks = 30e9       # Unjacketed bulk modulus Ks
    
    H = TPE_H(K, Ks)
    print('\nAssuming nu:', nu, ', mu', mu/1e9, 'GPa, Ks', Ks/1e9,'GPa')
    print('Drained isothermal bulk modulus K', K/1e9, 'GPa')
    print('Poroelastic parameter H', H/1e9, 'GPa')
    '''

#----------------------------------------------------------------------
# TPE - Estimate overpressure with/without delta T
    '''
    e0 = 1.61e-3    # potency
    H = 1.e9        # poro-elastic parameter (Pa)
    alpha = 3e-5    # thermal expansion coefficient (1/K)
    dT = 50         # Temperature change of the inclusion (K)
    
    dp1, dp2 = TPE_dp(e0, alpha, H, dT)
    
    print('\nAssuming e0', e0,', dT', dT, 'K, alpha', alpha,'1/K, H', H/1e9, 'GPa')
    print('dP only   ', dp1/1e6,'MPa')
    print('dP with dT', dp2/1e6,'MPa')
    '''

#----------------------------------------------------------------------
# TPE - Estimate volume variation of the cylinder
    '''
    ni = 0.25       # Poisson coefficient
    r = 1897        # cylinder radius (m)
    h = 144         # cylinder height (m)
    e0 = 1.61e-3    # potency

    volume, dVolume = TPE_volume_var_cylinder(e0,r,h,ni)
    print('\nAssuming e0',e0,', radius',r,'m, height',h,'m')
    print('Volume of the cylinder', volume/1e6,'1e6 m3')
    print('Volume variation of the cylinder', dVolume/1e6,'1e6 m3')
    '''

#----------------------------------------------------------------------
# TPE - Estimate volume variation of the sphere
    
    ni = 0.25       # Poisson coeff.
    r = 319         # sphere radius
    e0 = 1.61e-3    # potency

    volume, dVolume = TPE_volume_var_sphere(e0,r,ni)
    print('\nAssuming e0',e0,', radius',r,'m')
    print('Volume of the sphere', volume/1e6,'1e6 m3')
    print('Volume variation of the sphere', dVolume/1e6,'1e6 m3')
    
   


#----------------------------------------------------------------------
# PLOT 1D - 2D statistics via corner
    '''
    folder = '../USECASE/mogi1NA/'
    f_models ='VSM_models.csv'
    f_truths = 'VSM_best.csv'
    f_plot = 'VSM_plot1D2D_new.pdf'
    nskip = 4000
    
    
    df = pd.read_csv(os.path.join(folder,f_models))
    tf = pd.read_csv(os.path.join(folder,f_truths))
    nunk = np.size(tf)
    
    if(f_truths[4:8] == 'best'):
        dd = df.iloc[:,2:] # NA
        tt = tf.values.reshape(nunk) #NA
    else:
        dd = df #BI
        tp = tf.values.reshape(nunk)
        tt = tp[::2] #BI
    
    VSM.plot_1D_2D(dd, tt, dd.columns, os.path.join(folder,f_plot), nskip)
    '''