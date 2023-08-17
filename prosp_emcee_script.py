import os
os.environ['SPS_HOME'] = '/home/goio/fsps/'

import re
from pathlib import Path
import itertools
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
from matplotlib import rc
import numpy as np

import fsps
import dynesty
import sedpy
import h5py, astropy
import astroquery
from prospect.utils.obsutils import fix_obs
import sedpy
from sedpy.observate import list_available_filters
from sedpy import observate
from astropy.coordinates import SkyCoord
import astropy.wcs as wcs
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, WMAP9 as cosmo
from prospect.models.templates import TemplateLibrary
from prospect.models import SpecModel
import prospect.models
from prospect.models import priors
from prospect.sources import CSPSpecBasis
from prospect.sources import FastStepBasis
from prospect.likelihood import lnlike_spec, lnlike_phot, write_log
from prospect.fitting import fit_model
from prospect.models import PolySpecModel
import prospect.models
from prospect.fitting import lnprobfn, fit_model
from prospect.io import write_results as writer


parser = argparse.ArgumentParser(
                    prog='emceeProspect',
                    description='Process one SN with propsector',
                    epilog='Text at the bottom of help')
parser.add_argument('target')           # positional argument
parser.add_argument('analysis', choices=['global', 'local'])
parser.add_argument('-a', '--aperture')
args = parser.parse_args()
target, analysis, ap = args.target, args.analysis, args.aperture
if analysis == 'local' and ap is None:
    print("Aperture needed for local analysis.")
    exit()
print(target, analysis, ap, type(ap))

surveys = {'PS1':['g','r','i','z','y'], '2MASS':['J','H','Ks'], 'unWISE':['W1','W2','W3','W4'], 'SDSS':['u','g','r','i','z'], 'DES':['Y','g','r','i','z'], 'GALEX':['FUV','NUV']}
PS1_filt=[]
for filt in ['g','r','i','z','y']:
    PS1 = observate.Filter("PS1_"+filt, directory="/home/goio/analisis/filterpars/")
    PS1_filt.append(PS1)
    
twomass_filt = ['twomass_'+filt for filt in ['J','H','Ks']]
wise_filt = ['wise_w'+filt for filt in ['1','2','3','4']]
sdss_filt = ['sdss_'+filt+'0' for filt in ['u','g','r','i','z']]
DES_filt=['decam_'+filt for filt in ['Y','g','r','i','z']]
galex_filt= ['galex_'+filt for filt in ['FUV','NUV']]

twomass_filt+=wise_filt+sdss_filt+DES_filt+galex_filt

def_filt=observate.load_filters(twomass_filt)

PS1_filt+=def_filt


counter = itertools.count(0)
filters_dict = {survey:{filt:PS1_filt[next(counter)] for filt in surveys[survey]} for survey in surveys}

path = Path('images') / target

survey_pattern = f"{analysis}_phot_(.+)\.csv"

files=[f for f in sorted(os.listdir(path)) if f.startswith(analysis)]

filters_list = []
mags_and_errors = ([], [])

z_obs = pd.read_csv(path / files[0], index_col=False)['zspec'].item()

invalid_items = [np.inf, -np.inf]
for fname in files:
    survey = re.match(survey_pattern, fname).group(1)
    df = pd.read_csv(path / fname, index_col=False)
    for band in surveys[survey]:
        if survey == 'SDSS' and band != 'u':
            continue
        if survey == 'PS1' and filters_dict['DES']['Y' if band == 'y' else band] in filters_list:  # We use the fact that DES is always loaded first
            continue
        
        if analysis=='local':
            key = band+ap
        else:
            key = band
        if (mag := df[key].item()) not in invalid_items and not np.isnan(mag):
            mags_and_errors[0].append(mag)
            filters_list.append(filters_dict[survey][band])
        else:
            continue
        if np.isnan(err := df[key + '_err'].item()):
            err = 0
        mags_and_errors[1].append(err)
                
#Now you have float values of the u filter and arrays corresponding to the magnituds and errors per each band.
#IMPORTANT! The order matters, depending on the index of the array you have certain filter!
#Then you can just append this values into one array, (mAB and eMAB) by the same order that you import the filters
print()

print(filters_list)
print(len(filters_list))
print(mags_and_errors)
print(len(mags_and_errors[0]))


obs={}
obs['filters']=filters_list
obs["phot_wave"] = np.array([f.wave_effective for f in obs["filters"]])


#Add the redshift value by hand or load it


obs['redshift']=z_obs

#Then we load the magnitudes using a .txt file or using an array.
#IMPORTANT: The input that we provide to the dictionary has to follow the same order
#as the list of filters.

m_AB=np.array(mags_and_errors[0])
e_mAB=np.array(mags_and_errors[1])


obs["maggies"] = 10**(-0.4*m_AB)

#print('The error magnitudes are', e_mAB)

e_mAB = np.clip(e_mAB, 0.05, np.inf) #It is strongly recommended to add a floor error
obs["maggies_unc"]=e_mAB*10**(-0.4*m_AB)/1.086

#Since we are not trying to fit from input spectra the obs dictionary inputs related to that input should be None

obs["wavelength"] = None
obs["spectrum"] = None
obs['unc']=None
obs['mask']=None
obs=fix_obs(obs)

print(f'Creating model for {target}...')

model_params = TemplateLibrary["continuity_sfh"]
 #We fit the known redshift
    
    
model_params["zred"]['isfree'] = False
model_params["zred"]["init"] = z_obs
    
nbins_sfh=5
    
model_params["nbins_sfh"] = dict(N=1, isfree=False, init=nbins_sfh)
model_params['agebins']['N'] = nbins_sfh
model_params['mass']['N'] = nbins_sfh
model_params['logsfr_ratios']['N'] = nbins_sfh-1
model_params['logsfr_ratios']['init'] = np.full(nbins_sfh-1, 0.0)  # constant SFH
model_params['logsfr_ratios']['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1, 0.0),
                                                         scale=np.full(nbins_sfh-1, 0.3),
                                                         df=np.full(nbins_sfh-1, 2))
               
 # add redshift scaling to agebins, such that t_max = t_univ
def zred_to_agebins(zred=None, nbins_sfh=None, **extras):
        tuniv = np.squeeze(cosmo.age(zred).to("yr").value)
        ncomp = np.squeeze(nbins_sfh)
        tbinmax = (tuniv*0.9)
        agelims = [0.0, 7.4772] + np.linspace(8.0, np.log10(tbinmax), ncomp-2).tolist() + [np.log10(tuniv)]
        agebins = np.array([agelims[:-1], agelims[1:]])
        return agebins.T

def logmass_to_masses(logmass=None, logsfr_ratios=None, zred=None, **extras):
    agebins = zred_to_agebins(zred=zred, **extras)
    logsfr_ratios = np.clip(logsfr_ratios, -10, 10)  # numerical issues...
    nbins = agebins.shape[0]
    sratios = 10**logsfr_ratios
    dt = (10**agebins[:, 1] - 10**agebins[:, 0])
    coeffs = np.array([(1./np.prod(sratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
    m1 = (10**logmass) / coeffs.sum()
    return m1 * coeffs

model_params['agebins']['depends_on'] = zred_to_agebins
model_params['mass']['depends_on'] = logmass_to_masses

  # --- metallicity (flat prior) ---
model_params["logzsol"]["prior"] = priors.TopHat(mini=-1.0, maxi=0.19)

  # --- complexify the dust ---
model_params['dust_type']['init'] = 4
model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=2.0, mean=0.3, sigma=1)
model_params["dust_index"] = dict(N=1, isfree=True, init=0,
                                        prior=priors.TopHat(mini=-1.0, maxi=0.2))

def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
      return dust1_fraction*dust2

model_params['dust1'] = dict(N=1, isfree=False, init=0,
                                   prior=None, depends_on=to_dust1)
model_params['dust1_fraction'] = dict(N=1, isfree=True, init=1.0,prior=priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3))

#model_params['duste_gamma'] = {'N':1, 'isfree':False, 'init':0.01}
#model_params['duste_umin'] = {'N':1, 'isfree':False, 'init':1.0}
#model_params['duste_qpah'] = {'N':1, 'isfree':False, 'init':2.0}

#We store all the initial fixed values and priors and build the model
model=SpecModel(model_params)


sps = FastStepBasis(zcontinuous=1)
noise_model = (None, None)


# --- start minimization ----
run_params={}
run_params["dynesty"] = False
run_params["emcee"] = False
run_params["optimize"] = True
run_params["min_method"] = 'lm'
# We'll start minimization from "nmin" separate places, 
# the first based on the current values of each parameter and the 
# rest drawn from the prior.  Starting from these extra draws 
# can guard against local minima, or problems caused by 
# starting at the edge of a prior (e.g. dust2=0.0)
run_params["nmin"] = 12

output = fit_model(obs, model, sps, lnprobfn=lnprobfn,noise=noise_model, **run_params)

print("Done optimization in {}s".format(output["optimization"][1]))


# Set this to False if you don't want to do another optimization
# before emcee sampling (but note that the "optimization" entry 
# in the output dictionary will be (None, 0.) in this case)
# If set to true then another round of optmization will be performed 
# before sampling begins and the "optmization" entry of the output
# will be populated.
run_params["optimize"] = False
run_params["emcee"] = True
run_params["dynesty"] = False
# Number of emcee walkers
run_params["nwalkers"] = 128
# Number of iterations of the MCMC sampling
run_params["niter"] = 1024
# Number of iterations in each round of burn-in
# After each round, the walkers are reinitialized based on the 
# locations of the highest probablity half of the walkers.
run_params["nburn"] = [16, 32, 64]

output = fit_model(obs, model, sps, lnprobfn=lnprobfn, noise=noise_model, **run_params)

print('done emcee in {0}s'.format(output["sampling"][1]))


fit_type='emcee'
apstr = '_' + ap if ap is not None else ''
hfile = "./PROSPECT_results/"+target+"_"+fit_type+"_"+analysis+apstr+"_mcmc.h5"
writer.write_hdf5(hfile, run_params, model, obs,
                 output["sampling"][0], output["optimization"][0],
                 sps=sps,
                 tsample=output["sampling"][1],
                 toptimize=output["optimization"][1])