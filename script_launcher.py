import os
import subprocess as sp
import multiprocessing as mp
from pathlib import Path

cmd = ['/home/goio/miniconda3/envs/prospector/bin/python', '/home/goio/analisis/prosp_emcee_script.py']
filefolder = Path('images')
resultsfolder = Path('PROSPECT_results')
redo = False
global_skip_list = ['2006bh']
local_skip_list = []

def execute_batch(targets, analysis, apertures=None):
    print(f"Processing {batch_list}")
    pool = mp.Pool(len(targets))
    if analysis == 'local':
        cmds = [cmd + [target, analysis, '-a', aperture] for target, aperture in zip(targets, apertures)]
    else:
        cmds = [cmd + [target, analysis] for target in targets]
    print(cmds)
    pool.map(sp.run, cmds)
    pool.close()

snes = sorted(os.listdir(filefolder))
print(snes)

batch_list = []
for sn in snes:
    if not redo:
        result_name = f'{sn}_emcee_global_mcmc.h5'
        if (resultsfolder/result_name).exists() or sn in global_skip_list:
            continue
    batch_list.append(sn)
    if len(batch_list) == 4:
        execute_batch(batch_list, 'global')
        batch_list = []
if len(batch_list) != 0:
    execute_batch(batch_list, 'global')

apertures = ['1', '2', '3', '4']
batch_list = []
aperture_list = []
for sn in snes:
    for aperture in apertures:
        if not redo:
            result_name = f'{sn}_emcee_local_{aperture}_mcmc.h5'
            if (resultsfolder/result_name).exists() or (sn, aperture) in local_skip_list:
                continue
        batch_list.append(sn)
        aperture_list.append(aperture)
        if len(batch_list) == 4:
            execute_batch(batch_list, 'local', aperture_list)
            batch_list = []
            aperture_list = []
if len(batch_list) != 0:
    execute_batch(batch_list, 'local', aperture_list)