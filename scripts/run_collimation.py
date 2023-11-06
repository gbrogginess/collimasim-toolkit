import os
import re
import sys
import time
import json
import glob
import copy
import gzip
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from warnings import warn
from collections import namedtuple
from itertools import cycle, islice, dropwhile
from multiprocessing import Pool
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr, contextmanager

import yaml
import pandas as pd
import numexpr as ne
import pdb
from IPython import embed
from schema import Schema, And, Or, Use, Optional, SchemaError

import xobjects as xo
import xtrack as xt
import xpart as xp
import xfields as xf
from pylhc_submitter.job_submitter import main as htcondor_submit

XTRACK_TWISS_KWARGS = {}

# Sometimes attempting to import collimasim when all the libraries are not on the path
# will break the program, even when using try-except
# Avoid this by testing the import in a subprocess
import subprocess
if subprocess.run([sys.executable, '-c', 'import collimasim']).returncode == 0:
    import collimasim as cs
else:
    warn("collimasim cannot be imported, some features will not work")

ParticleInfo = namedtuple('ParticleInfo', ['name', 'pdgid', 'mass', 'charge'])
MADX_ELECTRON_MASS_EV = 510998.95
PARTICLE_INFO_DICT = {
    # 'electron': ParticleInfo('electron', 11, xp.constants.ELECTRON_MASS_EV, -1),
    # 'positron': ParticleInfo('positron', -11, xp.constants.ELECTRON_MASS_EV, 1),
    'electron': ParticleInfo('electron', 11, MADX_ELECTRON_MASS_EV, -1),
    'positron': ParticleInfo('positron', -11, MADX_ELECTRON_MASS_EV, 1),
    'proton': ParticleInfo('proton', 2212, xp.PROTON_MASS_EV, 1),
}
def _check_supported_particle(particle_name):
    if particle_name in PARTICLE_INFO_DICT:
        return True
    else:
        return False

# Note that YAML has inconsitencies when parsing numbers in scientific notation
# To avoid numbers parsed as strings in some configurations, always cast to float / int
to_float = lambda x: float(x)
to_int = lambda x: int(float(x))

unit_GeV = 1e9 # eV

INPUT_SCHEMA = Schema({'machine': str,
                       'xtrack_line': os.path.exists,
                       'collimator_file': os.path.exists,
                       'bdsim_config': os.path.exists,
                       Optional('tfs_file'): os.path.exists,
                       Optional('material_rename_map', default={}): Schema({str: str}),
                       })

BEAM_SCHEMA = Schema({'particle': _check_supported_particle,
                      'momentum': Use(to_float),
                      'emittance': Or(Use(to_float), {'x': Use(to_float), 'y': Use(to_float)}),
                      })

GPDIST_DIST_SCHEMA = Schema({'file': os.path.exists})

XSUITE_DIST_SCHEMA = Schema({'file': os.path.exists,
                             Optional('keep_ref_particle', default=False): Use(bool),
                             Optional('copy_file', default=False): Use(bool),
                             })
HALO_POINT_SCHM = Schema({'type': And(str, lambda s: s in ('halo_point', )),
                          'impact_parameter': Use(to_float),
                          'side': And(str, lambda s: s in ('positive', 'negative', 'both')),
                          'sigma_z': Use(to_float)
                         })
HALO_MPOINT_SCHM = Schema({'type': And(str, lambda s: s in ('halo_point_momentum', )),
                           'impact_parameter': Use(to_float),
                           'num_betatron_sigma': Use(to_float),
                           'sigma_z': Use(to_float),
                           'side': And(str, lambda s: s in ('positive', 'negative', 'both')),
                          })
HALO_DIR_SCHM = Schema({'type': And(str, lambda s: s in ('halo_direct',)),
                        'sigma_spread': Use(to_float),
                        'side': And(str, lambda s: s in ('positive', 'negative', 'both')),
                        'sigma_z': Use(to_float)
                        })
MATCHED_SCHM = Schema({'type': And(str, lambda s: s in ('matched_beam',)),
                        'sigma_z': Use(to_float)
                        })


DIST_SCHEMA = Schema({'source': And(str, lambda s: s in ('gpdist', 'internal', 'xsuite')),
             Optional('start_element', default=None): Or(str.lower, None),
        'parameters': Or(GPDIST_DIST_SCHEMA,
                         XSUITE_DIST_SCHEMA,
                         MATCHED_SCHM,
                         HALO_DIR_SCHM,
                         HALO_POINT_SCHM,
                         HALO_MPOINT_SCHM),
        })

DYNC_ELE_SCHEMA = Schema({Optional('element_name'): str,
                          Optional('element_regex'): str,
                          'parameter': str,
                          'change_function': str
                         })

DYNC_SCHEMA = Schema({'element': Or(DYNC_ELE_SCHEMA, Schema([DYNC_ELE_SCHEMA])),
                     })

INSERT_ELE_SCHEMA = Schema({'type': str,
                            'name': str,
                            'at_s': Or(Use(float), list),
                            'parameters': dict,
                           })

# Needed for dict
# # Enable beamstrahlung flag
# # Particle distribution loading

ONE_BEAMBEAM_SCHEMA = Schema({'at_element': str,
                              'bunch_intensity': Use(to_float),
                              'sigma_z': Use(to_float),
                              'crossing_angle': Use(to_float),
                              'other_beam_q0': Use(to_int),
                              'n_slices': Use(to_int)
                            })
BEAMBEAM_SCHEMA = Schema({'beambeam': Or(ONE_BEAMBEAM_SCHEMA, 
                                         Schema([ONE_BEAMBEAM_SCHEMA])),})

RUN_SCHEMA = Schema({'energy_cut': Use(to_float),
                     'seed': Use(to_int),
                     'turns': Use(to_int),
                     'nparticles': Use(to_int),
                     'max_particles': Use(to_int),
                     Optional('radiation', default='off'): And(str, lambda s: s in ('off', 'mean', 'quantum')),
                     Optional('beamstrahlung', default='off'): And(str, lambda s: s in ('off', 'mean', 'quantum')),
                     Optional('turn_rf_off', default=False): Use(bool),
                     Optional('compensate_sr_energy_loss', default=False): Use(bool),
                     Optional('sr_compensation_delta', default=None): Or(Use(to_float), None),
                     Optional('aperture_interp', default=None): Or(Use(to_float), None),
                     Optional('outputfile', default='part.hdf'): str,
                     Optional('batch_mode', default=True): Use(bool),
                     })

JOB_SUBMIT_SCHEMA = Schema({'mask': Or(os.path.exists, lambda s: s=='default'),
                            'working_directory': str,
                            'num_jobs': Use(to_int),
                            Optional('replace_dict'): str,
                            Optional('append_jobs'): bool,
                            Optional('dryrun'): bool,
                            Optional('executable', default='bash'): str,
                            Optional('htc_arguments'): dict,
                            Optional('job_output_dir'): str,
                            Optional('jobflavour'): str,
                            Optional('jobid_mask'): str,
                            Optional('num_processes'): Use(to_int),
                            Optional('resume_jobs'): bool,
                            Optional('run_local'): bool,
                            Optional('script_arguments'): bool,
                            Optional('script_extension'): str,
                            Optional('ssh'): str,
                          })


LOSSMAP_SCHEMA = Schema({'norm': And(str, lambda s: s in ('none','total', 'max', 'max_coll', 'total_coll')),
                         'weights': And(str, lambda s: s in ('none', 'energy')),
                         'aperture_binwidth': Use(to_float),
                         Optional('make_lossmap', default=True): Use(bool),
                         })

CONF_SCHEMA = Schema({'input': INPUT_SCHEMA,
                      'beam': BEAM_SCHEMA,
                      'dist': DIST_SCHEMA,
                      'run': RUN_SCHEMA,
                      Optional('insert_element'): Or(INSERT_ELE_SCHEMA, Schema([INSERT_ELE_SCHEMA])),
                      Optional('dynamic_change'): DYNC_SCHEMA,
                      Optional('lossmap'): LOSSMAP_SCHEMA,
                      Optional('jobsubmission'): JOB_SUBMIT_SCHEMA,
                      Optional(object): object})  # Allow input flexibility with extra keys


PART_DATA_VARS = ['s', 'x', 'y', 'px', 'py', 'zeta', 'delta',
                  'charge_ratio', 'weight', 'particle_id',
                  'at_element', 'at_turn', 'state',
                  'parent_particle_id']

# FCC_EE_WARM_REGIONS = np.array([0, 9E9])
FCC_EE_WARM_REGIONS = np.array([
    [0.0, 2.2002250210956813], [2.900225021095681, 2.9802250210956807], 
    [4.230225021095681, 4.310225021095681], [5.560225021095681, 5.860225021095681],
    [7.110225021095681, 7.1902250210956815], [8.440225021095682, 14.01430698619792], 
    [17.51430698619792, 21.054141203075883], [24.554141203075883, 64.54594481671067], 
    [68.04594481671067, 70.34999097862814], [73.84999097862814, 114.10750772564008], 
    [117.60750772564008, 22296.16304172309], [22299.66304172309, 22359.90116677595], 
    [22363.40116677595, 22466.117401781252], [22469.617401781252, 22537.590351601168], 
    [22541.090351601168, 22677.974916715677], [22681.474916715677, 22785.08912369255], 
    [22786.33912369255, 22786.41912369255], [22787.66912369255, 22787.969123692554], 
    [22789.219123692554, 22789.29912369255], [22790.54912369255, 22790.62912369256],  
    [22791.32912369256, 22795.729573734752], [22796.429573734753, 22796.50957373476], 
    [22797.75957373476, 22797.83957373476], [22799.08957373476, 22799.38957373476], 
    [22800.63957373476, 22800.71957373476], [22801.96957373476, 22807.543655699865], 
    [22811.043655699865, 22814.583489916746], [22818.083489916746, 22858.075293530368], 
    [22861.575293530368, 22863.87933969229], [22867.37933969229, 22907.63685643929], 
    [22911.13685643929, 45089.692390437995], [45093.192390437995, 45153.43051549084], 
    [45156.93051549084, 45259.646750496155], [45263.146750496155, 45331.11970031608], 
    [45334.61970031608, 45471.50426543059], [45475.00426543059, 45578.61847240747], 
    [45579.86847240747, 45579.94847240747], [45581.19847240747, 45581.498472407475], 
    [45582.748472407475, 45582.82847240748], [45584.07847240748, 45584.158472407486], 
    [45584.85847240748, 45589.25892244966], [45589.95892244966, 45590.03892244966], 
    [45591.28892244966, 45591.36892244966], [45592.61892244966, 45592.91892244967], 
    [45594.16892244967, 45594.248922449675], [45595.498922449675, 45601.07300441477], 
    [45604.57300441477, 45608.11283863164], [45611.61283863164, 45651.604642245286], 
    [45655.104642245286, 45657.40868840722], [45660.90868840722, 45701.16620515424], 
    [45704.66620515424, 67883.22174475367], [67886.72174475367, 67946.95986980652], 
    [67950.45986980652, 68053.17610481184], [68056.67610481184, 68124.64905463177], 
    [68128.14905463177, 68265.03361974626], [68268.53361974626, 68372.14782672313], 
    [68373.39782672313, 68373.47782672313], [68374.72782672313, 68375.02782672313],
    [68376.27782672313, 68376.35782672314], [68377.60782672314, 68377.68782672312], 
    [68378.38782672312, 68382.78827676529], [68383.48827676529, 68383.56827676529], 
    [68384.81827676529, 68384.89827676529], [68386.14827676529, 68386.4482767653],
    [68387.6982767653, 68387.7782767653], [68389.0282767653, 68394.60235873041], 
    [68398.10235873041, 68401.64219294729], [68405.14219294729, 68445.13399656092], 
    [68448.63399656092, 68450.93804272286], [68454.43804272286, 68494.69555946988], 
    [68498.19555946988, 90676.75109905902], [90680.25109905902, 90740.4892241119], 
    [90743.9892241119, 90846.7054591172], [90850.2054591172, 90918.17840893712], 
    [90921.67840893712, 91058.5629740516], [91062.0629740516, 91165.67718102848], 
    [91166.92718102848, 91167.00718102849], [91168.25718102849, 91168.55718102849], 
    [91169.80718102849, 91169.88718102848], [91171.13718102848, 91171.21718102848]])

HLLHC_WARM_REGIONS = np.array([
    [0.00000000e+00, 2.25000000e+01],
    [8.31530000e+01, 1.36689000e+02],
    [1.82965500e+02, 2.01900000e+02],
    [2.10584700e+02, 2.24300000e+02],
    [3.09545428e+03, 3.15562858e+03],
    [3.16774008e+03, 3.18843308e+03],
    [3.21144458e+03, 3.26386758e+03],
    [3.30990008e+03, 3.35497408e+03],
    [3.40100558e+03, 3.45342858e+03],
    [3.47644008e+03, 3.49406558e+03],
    [3.50588528e+03, 3.56831858e+03],
    [6.40540880e+03, 6.45791380e+03],
    [6.46877850e+03, 6.85951380e+03],
    [6.87037850e+03, 6.92353380e+03],
    [9.73590702e+03, 9.82473052e+03],
    [9.83083202e+03, 9.86173052e+03],
    [9.87873202e+03, 9.93998552e+03],
    [9.95054802e+03, 1.00434620e+04],
    [1.00540245e+04, 1.01152780e+04],
    [1.01322795e+04, 1.01639705e+04],
    [1.01700720e+04, 1.02576030e+04],
    [1.31036000e+04, 1.31200300e+04],
    [1.31238892e+04, 1.31471237e+04],
    [1.31918002e+04, 1.32476472e+04],
    [1.33067940e+04, 1.33520892e+04],
    [1.34110312e+04, 1.34670082e+04],
    [1.35114547e+04, 1.35357845e+04],
    [1.35388592e+04, 1.35552845e+04],
    [1.63946378e+04, 1.64508713e+04],
    [1.64569728e+04, 1.64872713e+04],
    [1.64933728e+04, 1.68308713e+04],
    [1.68369728e+04, 1.68672713e+04],
    [1.68733728e+04, 1.69282948e+04],
    [1.97348504e+04, 1.97606997e+04],
    [1.97715644e+04, 2.02179087e+04],
    [2.02287734e+04, 2.02529744e+04],
    [2.30899797e+04, 2.31385770e+04],
    [2.31503967e+04, 2.31713755e+04],
    [2.31943870e+04, 2.32468100e+04],
    [2.32928425e+04, 2.33379155e+04],
    [2.33839480e+04, 2.34363710e+04],
    [2.34593825e+04, 2.34800825e+04],
    [2.34921940e+04, 2.35531160e+04],
    [2.64334879e+04, 2.64483032e+04],
    [2.64569832e+04, 2.64759232e+04],
    [2.65221932e+04, 2.65757332e+04],
    [2.66363832e+04, 2.66588832e+04],
])


C_LIGHT = 299792458  # m / s

# A thin container to hold some information about a particle
Particle = namedtuple('Particle', ['mass', 'momentum', 'kinetic_energy',
                                   'total_energy', 'charge', 'pdgid'])

# Some utility functions
def _kinetic_energy(mass, energy): return energy - mass
def _momentum(mass, energy): return np.sqrt(energy ** 2 - mass ** 2)
def _energy(mass, momentum): return np.sqrt(momentum ** 2 + mass ** 2)


class BeamHeater():
    def __init__(self, name, max_kick_x, max_kick_y):
        self.max_kick_x = max_kick_x
        self.max_kick_y = max_kick_y
    def track(self, particles):
        particles.px += -self.max_kick_x + 2 * np.random.uniform(size=len(particles.px)) * self.max_kick_x
        particles.py += -self.max_kick_y + 2 * np.random.uniform(size=len(particles.py)) * self.max_kick_y

CUSTOM_ELEMENTS = {'BeamHeater': BeamHeater} 

try:
    from xcain import laser_interaction as xc
    print('XCain found, LaserInteraction will be available as a user element')
    CUSTOM_ELEMENTS['LaserInteraction'] = xc.LaserInteraction
except ImportError:
    pass


def find_apertures(line):
    i_apertures = []
    apertures = []
    for ii, ee in enumerate(line.elements):
        if ee.__class__.__name__.startswith('Limit'):
            i_apertures.append(ii)
            apertures.append(ee)
    return np.array(i_apertures), np.array(apertures)


def find_collimators(line):
    i_apertures = []
    apertures = []
    for ii, ee in enumerate(line.elements):
        if ee.__class__.__name__.startswith('BeamInteraction'):
            i_apertures.append(ii)
            apertures.append(ee)
    return np.array(i_apertures), np.array(apertures)



def insert_collimator_bounding_apertures(line):
    # Place aperture defintions around all collimators in order to ensure
    # the correct functioning of the aperture loss interpolation
    # the aperture definitions are taken from the nearest neighbour aperture in the line
    s_pos = line.get_s_elements(mode='upstream')
    apert_idx, apertures = find_apertures(line)
    apert_s = np.take(s_pos, apert_idx)

    coll_idx, collimators = find_collimators(line)
    coll_names = np.take(line.element_names, coll_idx)
    coll_s_start = np.take(s_pos, coll_idx)
    coll_s_end = np.take(s_pos, coll_idx + 1)

    # Find the nearest neighbour aperture in the line
    coll_apert_idx_start = np.searchsorted(apert_s, coll_s_start, side='left')
    coll_apert_idx_end = coll_apert_idx_start + 1

    aper_start = apertures[coll_apert_idx_start]
    aper_end = apertures[coll_apert_idx_end]

    # A bit of tolerance to ensure the correct aperture is selected
    # TODO: the tolerance is maybe not required
    tolerance = 1e-12
    for ii in range(len(collimators)):
        line.insert_element(at_s=coll_s_start[ii] - tolerance,
                            element=aper_start[ii].copy(),
                            name=coll_names[ii] + '_aper_start')

        line.insert_element(at_s=coll_s_end[ii] + tolerance,
                            element=aper_end[ii].copy(),
                            name=coll_names[ii] + '_aper_end')


def _insert_user_element(line, elem_def):
    elements = {**vars(xt.beam_elements.elements), **CUSTOM_ELEMENTS}
    # try a conversion to float, as because of the arbitraty yaml
    # inputs, no type enforcement can be made at input validation time
    # anything that can be cast to a number is likely a number
    parameters = {}
    for param, value in elem_def['parameters'].items():
        try:
            parameters[param] = float(value)
        except:
            parameters[param] = value
    print(parameters)
    elem_name = elem_def['name']
    #elem_obj = getattr(xt, elem_def['type'])(**elem_def['parameters'])
    elem_obj = elements[elem_def['type']](**parameters)
    s_position = elem_def['at_s']

    if not isinstance(s_position, list):
        print(f'Inserting {elem_name} ({elem_obj}) at s={s_position} m')
        line.insert_element(at_s=float(s_position), element=elem_obj, 
                            name=elem_name)
    else:
        for i, s_pos in enumerate(s_position):
            # TODO: Is a new instance really needed every time here?
            unique_name = f'{elem_name}_{i}'
            #unique_elem_obj = getattr(xt, elem_def['type'])(**elem_def['parameters'])
            unique_elem_obj = elements[elem_def['type']](**parameters)
            print(f'Inserting {unique_name} ({unique_elem_obj}) at s={s_pos} m')

            line.insert_element(at_s=float(s_pos), 
                                element=unique_elem_obj, 
                                name=unique_name)
            
def _make_bb_lens(nb, phi, sigma_z, alpha, n_slices, other_beam_q0,
                  sigma_x, sigma_px, sigma_y, sigma_py, beamstrahlung_on=False):
       
    slicer = xf.TempSlicer(n_slices=n_slices, sigma_z=sigma_z, mode="shatilov")

    el_beambeam = xf.BeamBeamBiGaussian3D(
            #_context=context,
            config_for_update = None,
            other_beam_q0=other_beam_q0,
            phi=phi, # half-crossing angle in radians
            alpha=alpha, # crossing plane
            # decide between round or elliptical kick formula
            min_sigma_diff = 1e-28,
            # slice intensity [num. real particles] n_slices inferred from length of this
            slices_other_beam_num_particles = slicer.bin_weights * nb,
            # unboosted strong beam moments
            slices_other_beam_zeta_center = slicer.bin_centers,
            slices_other_beam_Sigma_11    = n_slices*[sigma_x**2], # Beam sizes for the other beam, assuming the same is approximation
            slices_other_beam_Sigma_22    = n_slices*[sigma_px**2],
            slices_other_beam_Sigma_33    = n_slices*[sigma_y**2],
            slices_other_beam_Sigma_44    = n_slices*[sigma_py**2],
            # only if BS on
            slices_other_beam_zeta_bin_width_star_beamstrahlung = None if not beamstrahlung_on else slicer.bin_widths_beamstrahlung / np.cos(phi),  # boosted dz
            # has to be set
            slices_other_beam_Sigma_12    = n_slices*[0],
            slices_other_beam_Sigma_34    = n_slices*[0],
        )
    el_beambeam.iscollective = True # Disable in twiss

    return el_beambeam
    

def _insert_beambeam_elements(line, config_dict, twiss_table, emit):
    beamstrahlung_mode = config_dict['run'].get('beamstrahlung', 'off')
    beamstrahlung_on = beamstrahlung_mode != 'off'

    beambeam_block = config_dict.get('beambeam', None)
    if beambeam_block is not None:

        beambeam_list = beambeam_block
        if not isinstance(beambeam_list, list):
            beambeam_list = [beambeam_list, ]

        print('Beam-beam definitions found, installing beam-beam elements at: {}'
              .format(', '.join([dd['at_element'] for dd in beambeam_list])))
            
        for bb_def in beambeam_list:
            element_name = bb_def['at_element']
            # the beam-beam lenses are thin and have no effects on optics so no need to re-compute twiss
            element_twiss_index = list(twiss_table.name).index(element_name)
            # get the line index every time as it changes when elements are installed
            element_line_index = line.element_names.index(element_name)
            #element_spos = twiss_table.s[element_twiss_index]
            
            sigmas = twiss_table.get_betatron_sigmas(*emit if hasattr(emit, '__iter__') else (emit, emit))

            bb_elem = _make_bb_lens(nb=float(bb_def['bunch_intensity']), 
                                    phi=float(bb_def['crossing_angle']), 
                                    sigma_z=float(bb_def['sigma_z']),
                                    n_slices=int(bb_def['n_slices']),
                                    other_beam_q0=int(bb_def['other_beam_q0']),
                                    alpha=0, # Put it to zero, it is okay for this use case
                                    sigma_x=sigmas['Sigma11'][element_twiss_index], 
                                    sigma_px=sigmas['Sigma22'][element_twiss_index], 
                                    sigma_y=sigmas['Sigma33'][element_twiss_index], 
                                    sigma_py=sigmas['Sigma44'][element_twiss_index], 
                                    beamstrahlung_on=beamstrahlung_on)
            
            line.insert_element(index=element_line_index, 
                                element=bb_elem,
                                name=f'beambeam_{element_name}')
        

def cycle_line(line, name):
    names = line.element_names.copy()

    if name not in names:
        raise ValueError(f'No element name {name} found in the line.')

    # Store the original s of the element
    s0 = line.get_s_elements(mode='upstream')[names.index(name)]

    names_cyc = list(
        islice(dropwhile(lambda n: n != name, cycle(names)),  len(names)))
    line.element_names = names_cyc

    return line, s0


def _configure_tracker_radiation(line, radiation_model, beamstrahlung_model=None, for_optics=False):
    mode_print = 'optics' if for_optics else 'tracking'

    print_message = f"Tracker synchrotron radiation mode for '{mode_print}' is '{radiation_model}'"

    _beamstrahlung_model = None if beamstrahlung_model == 'off' else beamstrahlung_model

    if radiation_model == 'mean':
        if for_optics:
            # Ignore beamstrahlung for optics
            line.configure_radiation(model=radiation_model)
        else:
            line.configure_radiation(model=radiation_model, model_beamstrahlung=_beamstrahlung_model)

         # The matrix stability tolerance needs to be relaxed for radiation and tapering
        line.matrix_stability_tol = 0.5

    elif radiation_model == 'quantum':
        if for_optics:
            print_message = ("Cannot perform optics calculations with radiation='quantum',"
            " reverting to radiation='mean' for optics.")
            line.configure_radiation(model='mean')
        else:
            line.configure_radiation(model='quantum', model_beamstrahlung=_beamstrahlung_model)
        line.matrix_stability_tol = 0.5

    elif radiation_model == 'off':
        pass
    else:
        raise ValueError('Unsupported radiation model: {}'.format(radiation_model))
    print(print_message)


def _save_particles_hdf(particles=None, lossmap_data=None, filename='part'):
    if not filename.endswith('.hdf'):
        filename += '.hdf'

    fpath = Path(filename)
    # Remove a potential old file as the file is open in append mode
    if fpath.exists():
        fpath.unlink()

    if particles is not None:
        df = particles.to_pandas(compact=True)
        df.to_hdf(fpath, key='particles', format='table', mode='a',
                  complevel=9, complib='blosc')

    if lossmap_data is not None:
        for key, lm_df in lossmap_data.items():
            lm_df.to_hdf(fpath, key=key, mode='a', format='table',
                         complevel=9, complib='blosc')


def _load_particles_hdf(filename):
    return xp.Particles.from_pandas(pd.read_hdf(filename, key='particles'))


def _read_particles_hdf(filename):
    return pd.read_hdf(filename, key='particles')


def _load_lossmap_hdf(filename):
    keys = ('lossmap_scalar', 'lossmap_aper', 'lossmap_coll')

    lm_dict = {}
    for key in keys:
        # Pandas HDF file table format doesn't save empty dataframes
        try:
            lm_dict[key] = pd.read_hdf(filename, key=key)
        except KeyError:
            lm_dict[key] = None
    return lm_dict


def check_warm_loss(s, warm_regions):
    return np.any((warm_regions.T[0] < s) & (warm_regions.T[1] > s))


def load_config(config_file):
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)
    return config_dict


def _compensate_energy_loss(line, delta0=0.):
    _configure_tracker_radiation(line, 'mean')
    line.compensate_radiation_energy_loss(delta0=delta0)


def load_and_process_line(config_dict):
    beam = config_dict['beam']
    inp = config_dict['input']
    run = config_dict['run']

    emittance = beam['emittance']
    if isinstance(emittance, dict):
        emit = (emittance['x'], emittance['y'])
    else:
        emit = emittance

    particle_name = config_dict['beam']['particle']
    particle_info = None
    if particle_name in PARTICLE_INFO_DICT:
        particle_info = PARTICLE_INFO_DICT[particle_name]
    else:
        raise Exception('Particle {} not supported.'
        'Supported types are: {}'.format(particle_name, ', '.join(PARTICLE_INFO_DICT.keys())))

    p0 = beam['momentum']
    mass = particle_info.mass
    q0 = particle_info.charge
    ke0 = _kinetic_energy(mass, p0)
    e0 = _energy(mass, p0)
    ref_part = xp.Particles(p0c=p0, mass0=mass, q0=q0)

    comp_eloss = run.get('compensate_sr_energy_loss', False)

    # load the line and compute optics
    with open(inp['xtrack_line'], 'r') as fid:
        line = xt.Line.from_dict(json.load(fid))
    line.particle_ref = ref_part
    
    rf_cavities = line.get_elements_of_type(xt.elements.Cavity)[0]

    if run.get('turn_rf_off', False):
        print('Turning RF cavities off (set voltage to 0)')
        for cv in rf_cavities:
            cv.voltage = 0

    if not any((cv.voltage > 0 for cv in rf_cavities)) or not any((cv.frequency > 0 for cv in rf_cavities)):
        assert not comp_eloss, 'Cannot compensate SR energy loss with cavities off'
        print('RF cavities have no voltage or frequency, Twiss will be 4D')
        XTRACK_TWISS_KWARGS['method'] = '4d'

    if 'tfs_file' in inp and inp['tfs_file']:
        print('Using provided MAD-X twiss file for collimator optics')
        twiss = inp['tfs_file']
    else:
        print('Using Xtrack-generated twiss table for collimator optics')
        # Use a clean tracker to compute the optics
        # TODO: reduce the copying here
        optics_line = line.copy()
        optics_line.build_tracker()
        radiation_mode = run['radiation']
        if comp_eloss:
            # If energy loss compensation is required, taper the lattice
            print('Compensating synchrotron energy loss (tapering mangets)')
            comp_eloss_delta0 = run.get('sr_compensation_delta', 0.0)
            _compensate_energy_loss(optics_line, comp_eloss_delta0)
            line = optics_line.copy()
        _configure_tracker_radiation(optics_line, radiation_mode, for_optics=True)
        twiss = optics_line.twiss(**XTRACK_TWISS_KWARGS)
        line.tracker = None


    g4man = cs.Geant4CollimationManager(collimator_file=inp['collimator_file'],
                                        bdsim_config_file=inp['bdsim_config'],
                                        tfs_file=twiss,
                                        reference_pdg_id=particle_info.pdgid,
                                        reference_kinetic_energy=ke0,
                                        emittance_norm=emit,
                                        relative_energy_cut=run['energy_cut'],
                                        seed=run['seed'],
                                        material_rename_map=inp['material_rename_map'],
                                        batchMode=run['batch_mode'],
                                        )
    g4man.place_all_collimators(line)
    insert_collimator_bounding_apertures(line)

    s0 = 0
    start_element = config_dict['dist'].get('start_element', None)
    if start_element is not None:
        s0 = line.get_s_position(at_elements=start_element, mode='upstream')

    # Insert additional elements if any are specified:
    insert_elems = config_dict.get('insert_element', None)
    if insert_elems is not None:
        print('Inserting user-defined elements in the lattice')
        insert_elem_list = insert_elems
        if not isinstance(insert_elem_list, list):
            insert_elem_list = [insert_elem_list, ]
        
        for elem_def in insert_elem_list:
            _insert_user_element(line, elem_def)

    # Insert beam-beam lenses if any are specified:
    _insert_beambeam_elements(line, config_dict, twiss, emit)
    return line, ref_part, start_element, s0


def load_xsuite_csv_particles(dist_file, ref_particle, line, element, num_part, capacity, keep_ref_particle=False, copy_file=False):

    orig_file_path = Path(dist_file)
    if copy_file:
        dest = Path.cwd() / f'copied_{orig_file_path.name}'
        shutil.copy2(orig_file_path, dest)
        dist_file_path = dest
    else:
        dist_file_path = orig_file_path

    part_df = pd.read_csv(dist_file_path, index_col=0)

    # if keep_ref_particle:
    #     part_in = xp.Particles.from_pandas(part_df.iloc[:5])
    #     p0c =  part_in.p0c[0]
    #     mass0 = part_in.mass0
    #     q0 = part_in.q0
    # else:
    #     p0c =  ref_particle.p0c[0]
    #     mass0 = ref_particle.mass0
    #     q0 = ref_particle.q0

    at_element = line.element_names.index(element)
    start_s = line.get_s_position(at_elements=at_element, mode="upstream")

    # Need to make sure there is enough space allocated
    # particles = xp.Particles(_capacity=capacity,
    #                          p0c = p0c,
    #                          mass0 = mass0,
    #                          q0 = q0,
    #                          s = np.full(num_part, fill_value=start_s),
    #                          at_element = np.full(num_part, fill_value=start_s),
    #                          **{var: part_df[var][:num_part] for var in part_df.columns})
    
    particles = line.build_particles(
                _capacity=capacity,
                particle_ref=ref_particle,
                mode='set',
                x = part_df['x'].values[:num_part],
                px = part_df['px'].values[:num_part],
                y = part_df['y'].values[:num_part],
                py = part_df['py'].values[:num_part],
                zeta = part_df['zeta'].values[:num_part],
                delta = part_df['delta'].values[:num_part],
                **XTRACK_TWISS_KWARGS,
            )
    
    particles.start_tracking_at_element = at_element
    particles.at_element = at_element
    particles.s = start_s
    
    return particles

def load_gpdist_distr(dist_file, ref_particle, capacity):
    # Load a file with initial coordinates from gpdist and convert it to MADX inrays format

    # Be careful of the header, it may change
    #names = ['pid', 'genid', 'weight', 'x', 'y', 'z', 'xp', 'yp', 'zp', 'A', 'Z', 'm', 'p', 't']
    names = ['pid', 'genid', 'weight', 'x', 'y',
             'xp', 'yp', 'm', 'p', 't', 'q', 'g4pid']

    coords = pd.read_csv(dist_file, delim_whitespace=True,
                         index_col=False, names=names)

    loadpart = None
    if loadpart is not None:
        print(f'Running only {loadpart} particles.')

    coords = coords.iloc[0:loadpart]

    # transform the coorindates
    # The energy is in GeV 
    p0c = ref_particle.p0c
    mass0 = ref_particle.mass0
    q0 = ref_particle.q0

    coords['delta'] = (coords['p'] * unit_GeV - p0c) / p0c
    # TODO: check that zeta is ct, minus sign needed?
    coords['zeta'] = coords['t'] * C_LIGHT

    # TODO: Are the px, py coordinates the same as the ones in xtrack?
    coords.rename({'xp': 'px', 'yp': 'py'}, axis=1, inplace=True)

    xtrack_columns = ['x', 'px', 'y', 'py', 'delta', 'zeta']
    coords = coords[xtrack_columns]

    particles = xp.Particles(
        _capacity=capacity,
        p0c=p0c,  # In eV here
        mass0=mass0,
        q0=q0,
        x=coords['x'],
        px=coords['px'],
        y=coords['y'],
        py=coords['py'],
        zeta=coords['zeta'],
    )

    particles.delta[:len(coords['delta'])] = coords['delta']

    return particles


def _generate_direct_halo(line, ref_particle, coll_name, 
                          emitt_x, emitt_y, radiation_mode,
                          side, imp_par, 
                          spread, spread_symmetric, spread_isnormed, 
                          sigma_z, nsigma_for_offmom, 
                          num_particles, capacity):
    
    _configure_tracker_radiation(line, radiation_mode, for_optics=True)

    g4man = line.element_dict[coll_name].interaction_process.g4manager

    if not coll_name in g4man.collimators:
        raise Exception(f'Cannot generate a direct halo beam at element: {coll_name},'
                        'not a registered collimator')

    coll_dict = g4man.collimators[coll_name]
    angle = coll_dict['angle']
    nsigma_coll = coll_dict['nsigma']
    halfgap = coll_dict['halfgap']

    twiss = line.twiss(**XTRACK_TWISS_KWARGS)
    sigmas = twiss.get_betatron_sigmas(nemitt_x=emitt_x, nemitt_y=emitt_y)

    gemitts = dict(x = emitt_x/ref_particle.beta0[0]/ref_particle.gamma0[0],
                   y = emitt_x/ref_particle.beta0[0]/ref_particle.gamma0[0])

    plane = None
    converging = None
    if np.isclose(angle, 0):
        plane = 'x'

    elif np.isclose(angle, np.pi/2):
        plane = 'y'
        converging = coll_dict['alfy'] > 0
    else:
        plane = 's'
        raise Exception('Beams generation at skew collimators not implemented yet')

    # Take the positive (left) jaw, it is the same for the right
    betatron_angle = (-nsigma_for_offmom * twiss[f'alf{plane}', coll_name] 
                      * np.sqrt(gemitts[plane] / twiss[f'bet{plane}', coll_name]))
 
    delta_cut = ((halfgap - nsigma_for_offmom * sigmas[f'sigma_{plane}', coll_name])
                / twiss[f'd{plane}', coll_name])
    
    off_mom_angle = (np.sign(twiss[f'd{plane}', coll_name]) 
                     * delta_cut * twiss[f'dp{plane}', coll_name])
    
    converging = (betatron_angle + off_mom_angle) < 0


    plane_print = {'x': 'HORIZONTAL', 'y': 'VERTICAL'}[plane]
    conv_print = 'CONVERGING' if converging else 'DIVERGING'
    edge_print = 'UPSTREAM' if converging else 'DOWNSTREAM'
    print(f'Collimator {coll_name} identified as {plane_print}')
    print(f'Optics identified as {conv_print}, setting up the distribution at the {edge_print} edge')

    # Compute the extent of the halo
    coll_index = line.element_names.index(coll_name)
    match_element_index = coll_index if converging else coll_index + 1

    coll_s = line.get_s_position(at_elements=coll_name)
    coll_length = coll_dict['length']
    if converging:
        match_s = coll_s
    else:
        match_s = coll_s + coll_length

    gemitt = gemitts[plane]
    beta = twiss[f'bet{plane}', match_element_index]
    disp = twiss[f'd{plane}', match_element_index]
    disp_prime = twiss[f'dp{plane}', match_element_index]

    sigma = np.sqrt(beta * gemitt)
    phys_cut = halfgap + imp_par
    phys_cut_sigma = phys_cut / sigma

    phys_cut_betatron = nsigma_for_offmom * sigma

    if nsigma_for_offmom is not None:
        assert abs(disp) > 0, 'Must have non-zero dispersion for off-mometnum beam'
        momentum_cut = abs((phys_cut_sigma - nsigma_for_offmom)*sigma / disp)
    else:
        momentum_cut = abs(phys_cut / disp) if abs(disp) > 0 else np.nan
    
    if spread_isnormed:
        spread_phys=spread * sigma
    else:
        spread_phys=spread

    if spread_symmetric is True and phys_cut_betatron > spread_phys:
        halo_cut_inner = phys_cut_betatron - spread_phys/2 # in meters
        halo_cut_outer = phys_cut_betatron + spread_phys/2
    else:
        halo_cut_inner = phys_cut_betatron
        halo_cut_outer = phys_cut_betatron + spread_phys

    halo_nsigma_inner = halo_cut_inner / sigma
    halo_nsigma_outer = halo_cut_outer / sigma
    dr_sigmas = halo_nsigma_outer - halo_nsigma_inner

    # Extent computation done, generate the coordinates
    coord_dict = {
        'x': None, 'px': None, 
        'y': None, 'py': None, 
        'zeta': None, 'delta': None,
        'x_norm': None, 'px_norm': None,
        'y_norm': None, 'py_norm': None,
        # No delta_norm, as actually pzeta_norm is required and it is not needed anyway
    }

    if plane=='x':
        abs_plane, norm_plane = 'x', 'y'
    else:
        abs_plane, norm_plane = 'y', 'x'

    embed()
    # Collimator plane: generate pencil distribution in absolute coordinates
    abs_coords = []
    nsides = len(side) # 1 or 2
    _round_funcs = (np.ceil, np.floor) # ensure the total number of particles is conserved
    for i, _ss in enumerate(list(side)):
        factor = -1 if _ss == '-' else 1
        npart = int(_round_funcs[i](num_particles / nsides))

        # Generate the required betatron absolute coordinates
        coords = list(xp.generate_2D_pencil_with_absolute_cut(
                npart, 
                line = line,
                plane=abs_plane,
                absolute_cut=factor*halo_cut_inner,# + coll_dict[abs_plane], 
                dr_sigmas=dr_sigmas,
                side=_ss,
                nemitt_x=emitt_x, nemitt_y=emitt_y,
                at_element=coll_name, match_at_s=match_s,
                **XTRACK_TWISS_KWARGS,))

        # Add the dispersive offsets
        # the delta sign is computed to push the particles
        # towards the selected side
        delta_sign = factor * np.sign(disp)
        coords[0] += delta_sign * momentum_cut * disp
        coords[1] += delta_sign * momentum_cut * disp_prime
        coords.append(delta_sign * momentum_cut)
        
        abs_coords.append(coords)
    
    coord_dict[f'{abs_plane}'] = np.concatenate([cc[0] for cc in abs_coords])
    coord_dict[f'p{abs_plane}'] = np.concatenate([cc[1] for cc in abs_coords])
    coord_dict['delta'] = np.concatenate([cc[2] for cc in abs_coords])

    # Other plane: generate a gaussian
    norm_coords = xp.generate_2D_gaussian(num_particles)
    coord_dict[f'{norm_plane}_norm'] = norm_coords[0]
    coord_dict[f'p{norm_plane}_norm'] = norm_coords[1]

    # Longitudinal distribution if specified
    assert sigma_z >= 0
    if sigma_z > 0:
        print(f'Paramter sigma_z > 0, preparing a longitudinal distribution matched to the RF bucket')
        zeta_match, delta_match = line.generate_longitudinal_coordinates(
        num_particles=num_particles, distribution='gaussian',
        sigma_z=sigma_z)
    else:
        zeta_match = delta_match = 0

    # Longitudinal closed orbit
    delta_co = twiss.delta[match_element_index] # The closed orbit delta
    zeta_co = twiss.zeta[match_element_index]

    coord_dict['zeta'] = coord_dict['zeta'] + zeta_co + zeta_match
    coord_dict['delta'] = coord_dict['delta'] + delta_co + delta_match

    part = line.build_particles(
            _capacity=capacity,
            x=coord_dict['x'], px=coord_dict['px'],
            x_norm=coord_dict['x_norm'], px_norm=coord_dict['px_norm'],
            y=coord_dict['y'], py=coord_dict['py'],
            y_norm=coord_dict['y_norm'], py_norm=coord_dict['py_norm'],
            zeta=coord_dict['zeta'], delta=coord_dict['delta'],
            nemitt_x=emitt_x, nemitt_y=emitt_y,
            at_element=coll_name,
            match_at_s=match_s,
            **XTRACK_TWISS_KWARGS,
            )

    #"""
    embed()
    import matplotlib.pyplot as plt
    part_for_plot = part.copy()
    part_for_plot_prop = part.copy()

    drifted = False
    if not converging:
        drifted = True
        drift_coll_length = xt.elements.Drift(length=coll_length)
        drift_coll_length.track(part_for_plot)

    part_for_plot.hide_lost_particles()
    part_for_plot_prop.hide_lost_particles()

    if plane == 'x':
        offs = coll_dict['x']
        data_x = part_for_plot.x
        data_y = part_for_plot.px
        data_prop_x = part_for_plot_prop.x
        data_prop_y = part_for_plot_prop.px
        label_x = 'x [m]'
        label_y = 'px [rad]'
    else:
        offs = coll_dict['y']
        data_x = part_for_plot.y
        data_y = part_for_plot.py
        data_prop_x = part_for_plot_prop.y
        data_prop_y = part_for_plot_prop.py
        label_x = 'y [m]'
        label_y = 'py [rad]'

    plt.scatter(data_x, data_y, marker='.', label='halo particles')
    if drifted:
     plt.scatter(data_prop_x, data_prop_y, marker='.', label='starting particles')
    plt.axvline(halfgap + offs, c='r', label='coll. edge')
    plt.axvline(-halfgap + offs, c='r')
    plt.axvline(halfgap + imp_par + offs, c='g', label=f'coll. edge + imp. par. ({imp_par*1e6}um)')
    plt.axvline(-halfgap - imp_par + offs, c='g')
    ax = plt.gca()
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.legend()
    plt.show()
    fig, ax2 = plt.subplots()
    ax2.scatter(part_for_plot.zeta, part_for_plot.delta, marker='.', label='halo particles')
    ax2.set_xlabel('zeta')
    ax2.set_ylabel('delta')
    plt.show()
    embed()
    raise SystemExit()
    #"""
    return part


def _prepare_direct_halo(config_dict, line, ref_particle, element, emitt_x, emitt_y, num_particles, capacity):
    # The tracker is needed to compute optics here, so set the radiation mode appropriately
    radiation_mode = config_dict['run']['radiation']
    coll_name = element

    dist_params = config_dict['dist']['parameters']
    dist_type = dist_params['type']

    nsigma_for_offmom = None
    spread_symmetric = True
    spread_isnormed = False
    POINT_BEAM_SPREAD = 1e-9
    if dist_type == 'halo_point':
        imp_par = dist_params['impact_parameter']
        spread = POINT_BEAM_SPREAD
    elif dist_type == 'halo_point_momentum':
        imp_par = dist_params['impact_parameter']
        spread = POINT_BEAM_SPREAD
        nsigma_for_offmom = dist_params['num_betatron_sigma']
    elif dist_type == 'halo_direct':
        imp_par = 0
        spread = dist_params['sigma_spread']
        spread_symmetric = False
        spread_isnormed = True
    else:
        raise Exception('Cannot process distribution type')
    
    sigma_z = dist_params['sigma_z']
    
    side_def = dist_params['side']
    if side_def == 'negative':
        side = '-'
    elif side_def == 'positive':
        side = '+'
    elif side_def == 'both':
        side = '+-'

    part = _generate_direct_halo(line, ref_particle, coll_name, 
                                 emitt_x, emitt_y, radiation_mode,
                                 side, imp_par, 
                                 spread, spread_symmetric, spread_isnormed, 
                                 sigma_z, nsigma_for_offmom, 
                                 num_particles, capacity)
    return part

def _prepare_matched_beam(config_dict, line, ref_particle, element, emitt_x, emitt_y, num_particles, capacity):
    print(f'Preparing a matched Gaussian beam at {element}')
    sigma_z = config_dict['dist']['parameters']['sigma_z']
    radiation_mode =  config_dict['run'].get('radiation', 'off')

    _configure_tracker_radiation(line, radiation_mode, for_optics=True)

    x_norm, px_norm = xp.generate_2D_gaussian(num_particles)
    y_norm, py_norm = xp.generate_2D_gaussian(num_particles)
    
    # The longitudinal closed orbit needs to be manually supplied for now
    twiss = line.twiss(**XTRACK_TWISS_KWARGS)
    element_index = line.element_names.index(element)
    zeta_co = twiss.zeta[element_index] 
    delta_co = twiss.delta[element_index] 

    assert sigma_z >= 0
    zeta = delta = 0
    if sigma_z > 0:
        print(f'Paramter sigma_z > 0, preparing a longitudinal distribution matched to the RF bucket')
        zeta, delta = xp.generate_longitudinal_coordinates(
                        line=line,
                        num_particles=num_particles, distribution='gaussian',
                        sigma_z=sigma_z, particle_ref=ref_particle)

    part = line.build_particles(
        _capacity=capacity,
        particle_ref=ref_particle,
        x_norm=x_norm, px_norm=px_norm,
        y_norm=y_norm, py_norm=py_norm,
        zeta=zeta + zeta_co,
        delta=delta + delta_co,
        nemitt_x=emitt_x,
        nemitt_y=emitt_y,
        at_element=element,
        **XTRACK_TWISS_KWARGS,
        )

    return part

def generate_xpart_particles(config_dict, line, ref_particle, capacity):
    dist_params = config_dict['dist']['parameters']
    num_particles = config_dict['run']['nparticles']
    element = config_dict['dist']['start_element']

    emittance = config_dict['beam']['emittance']
    if isinstance(emittance, dict): # Normalised emittances
        ex, ey = emittance['x'], emittance['y']
    else:
        ex = ey = emittance

    particles = None
    dist_type = dist_params.get('type', '')
    if dist_type in ('halo_point', 'halo_point_momentum', 'halo_direct'):
        particles = _prepare_direct_halo(config_dict, line, ref_particle, 
                                         element, ex, ey, num_particles, capacity)
    elif dist_type == 'matched_beam':
        particles = _prepare_matched_beam(config_dict, line, ref_particle, 
                                          element, ex, ey, num_particles, capacity)
    else:
        raise Exception('Cannot process beam distribution')

    # TODO: Add offsets here
    
    # Disable this option as the tracking from element is handled
    # separately for consistency with other distribution sources
    particles.start_tracking_at_element = -1

    return particles

def load_xsuite_particles(config_dict, line, ref_particle, capacity):
    dist_params = config_dict['dist']['parameters']
    num_particles = config_dict['run']['nparticles']
    element = config_dict['dist']['start_element']

    dist_file = dist_params['file']
    keep_ref_part = dist_params['keep_ref_particle']
    copy_file = dist_params['copy_file']

    part = load_xsuite_csv_particles(dist_file, ref_particle, line, element, num_particles, capacity, 
                                     keep_ref_particle=keep_ref_part, copy_file=copy_file)
    
    return part
    

def prepare_particles(config_dict, line, ref_particle):
    dist = config_dict['dist']
    capacity = config_dict['run']['max_particles']

    _supported_dist = ['gpdist', 'internal']

    if dist['source'] == 'gpdist':
        particles = load_gpdist_distr(dist['file'], ref_particle, capacity)
    elif dist['source'] == 'xsuite':
        particles = load_xsuite_particles(config_dict, line, ref_particle, capacity)
    elif dist['source'] == 'internal':
        particles = generate_xpart_particles(config_dict, line, ref_particle, capacity)
    else:
        raise ValueError('Unsupported distribution source: {}. Supported ones are: {}'
                         .format(dist['soruce'], ','.join(_supported_dist)))
    return particles


def build_collimation_tracker(line):
    # Chose a context
    context = xo.ContextCpu()  # Only support CPU for Geant4 coupling TODO: maybe not needed anymore?
    # Transfer lattice on context and compile tracking code
    global_aper_limit = 1e3  # Make this large to ensure particles lost on aperture markers

    line.build_tracker(_context=context)
    line.config.global_xy_limit=global_aper_limit


def _compute_parameter(parameter, expression, turn, max_turn, extra_variables={}):
    # custom function handling - random numbers, special functions
    # populate the local variables with the computed values
    # TODO This is a bit wonky - may need a full parser later on
    if 'rand_uniform' in expression:
        rand_uniform = np.random.random()
    if 'rand_onoff' in expression:
        rand_onoff = np.random.randint(2)
    var_dict = {**locals(), **extra_variables}
    return type(parameter)(ne.evaluate(expression, local_dict=var_dict))


def _collect_element_names(line, match_string, regex_mode=False):
    match_re = re.compile(match_string) if regex_mode else re.compile(re.escape(match_string))
    names = [name for name in line.element_names if match_re.fullmatch(name)]
    if not names:
        raise Exception(f'Found no elements matching {match_string}')
    return names


def _prepare_dynamic_element_change(line, twiss_table, gemit_x, gemit_y, change_dict_list, max_turn):
    if change_dict_list is None:
        return None
    
    tbt_change_list = []
    for change_dict in change_dict_list:
        if not ('element_name' in change_dict) != ('element_regex' in change_dict):
            raise ValueError('Element name for dynamic change not speficied.')

        element_match_string = change_dict.get('element_regex', change_dict.get('element_name'))
        regex_mode = bool(change_dict.get('element_regex', False))

        parameter = change_dict['parameter']
        change_function = change_dict['change_function']

        element_names = _collect_element_names(line, element_match_string, regex_mode)
        # Handle the optional parameter[index] specifiers
        # like knl[0]
        param_split = parameter.replace(']','').split('[')
        param_name = param_split[0]
        param_index = int(param_split[1]) if len(param_split)==2 else None

        #parameter_values = []
        ebe_change_dict = {}
        if Path(change_function).exists():
            turn_no_in, value_in = np.genfromtxt('tbt_params.txt', 
                                                 converters = {0: int, 1: float}, 
                                                 unpack=True, comments='#')
            parameter_values = np.interp(range(max_turn), turn_no_in, value_in).tolist()
            # If the the change function is loaded from file,
            # all of the selected elements in the block have the same values
            for ele_name in element_names:
                ebe_change_dict[ele_name] = parameter_values
        else:
            ebe_keys = set(twiss_table._col_names) - {'W_matrix', 'name'}
            scalar_keys = (set(twiss_table._data.keys()) 
                           - set(twiss_table._col_names) 
                           - {'R_matrix', 'values_at', 'particle_on_co'})

            # If the change is computed on the fly, iterative changes
            # are permitted, e.g a = a + 5, so must account for different starting values
            for ele_name in element_names:
                parameter_values = []

                elem = line.element_dict[ele_name]
                elem_index = line.element_names.index(ele_name)
                elem_twiss_vals = {kk: float(twiss_table[kk][elem_index]) for kk in ebe_keys}
                scalar_twiss_vals = {kk: twiss_table[kk] for kk in scalar_keys}
                twiss_vals = {**elem_twiss_vals, **scalar_twiss_vals}

                twiss_vals['sigx'] = np.sqrt(gemit_x * twiss_vals['betx'])
                twiss_vals['sigy'] = np.sqrt(gemit_y * twiss_vals['bety'])
                twiss_vals['sigxp'] = np.sqrt(gemit_x * twiss_vals['gamx'])
                twiss_vals['sigyp'] = np.sqrt(gemit_y * twiss_vals['gamy'])

                param_value = getattr(elem, param_name)
                if param_index is not None:
                    param_value = param_value[param_index]
                twiss_vals['parameter0'] = param_value # save the intial value for use too

                for turn in range(max_turn):
                    param_value = _compute_parameter(param_value, change_function, 
                                                     turn, max_turn, 
                                                     extra_variables=twiss_vals)
                    parameter_values.append(param_value)

                ebe_change_dict[ele_name] = parameter_values

        tbt_change_list.append([param_name, param_index, ebe_change_dict])
        print('Dynamic element change list: ', tbt_change_list)

    return tbt_change_list


def _set_element_parameter(element, parameter, index, value):
    if index is not None:
        getattr(element, parameter)[index] = value
    else:
        setattr(element, parameter, value)


def _apply_dynamic_element_change(line, tbt_change_list, turn):
    for param_name, param_index, ebe_change_dict in tbt_change_list:
        for ele_name in ebe_change_dict:
            element = line.element_dict[ele_name]
            param_val = ebe_change_dict[ele_name][turn]
            _set_element_parameter(element, param_name, param_index, param_val)


def run(config_dict, line, particles, ref_part, start_element, s0):
    radiation_mode = config_dict['run']['radiation']
    beamstrahlung_mode = config_dict['run']['beamstrahlung']

    nturns = config_dict['run']['turns']
    
    _configure_tracker_radiation(line, radiation_mode, for_optics=True)
    twiss_table = line.twiss(**XTRACK_TWISS_KWARGS)

    emittance = config_dict['beam']['emittance']
    if isinstance(emittance, dict):
        emit = (emittance['x'], emittance['y'])
    else:
        emit = (emittance, emittance)

    gemit_x = emit[0]/ref_part.beta0[0]/ref_part.gamma0[0]
    gemit_y = emit[1]/ref_part.beta0[0]/ref_part.gamma0[0]

    # Look for changes to element parameters to apply every turn
    tbt_change_list = None
    if 'dynamic_change' in config_dict:
        dyn_change_dict =  config_dict['dynamic_change']
        if 'element' in dyn_change_dict:
            dyn_change_elem = dyn_change_dict.get('element', None)
            if dyn_change_elem is not None and not isinstance(dyn_change_elem, list):
                dyn_change_elem = [dyn_change_elem,]
            tbt_change_list = _prepare_dynamic_element_change(line, twiss_table, gemit_x, gemit_y, dyn_change_elem, nturns)

    
    _configure_tracker_radiation(line, radiation_mode, beamstrahlung_mode, for_optics=False)
    if radiation_mode == 'quantum':
        # Explicitly initialise the random number generator for the quantum mode
        seed = config_dict['run']['seed']
        seed_offset = 1e7 # Make sure the seeds are unique and don't overlap between simulations
        if seed > 1e5:
            raise ValueError('The random seed is too large. Please use a smaller seed (<1e5).')
        seeds = np.full(particles._capacity, seed) * seed_offset + np.arange(particles._capacity)
        particles._init_random_number_generator(seeds=seeds)

    ###################################################
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots()
    ###############################

    t0 = time.time()
    # Track (saving turn-by-turn data)
    for turn in range(nturns):
        print(f'Start turn {turn}, Survivng particles: {particles._num_active_particles}')
        if tbt_change_list is not None:
            _apply_dynamic_element_change(line, tbt_change_list, turn)

        #######################################################################
        # print(tbt_change_list)
        # part_copy = particles.copy()
        # part_copy.hide_lost_particles()
        # ax.scatter(part_copy.x, part_copy.px, marker='.', label=f'turn {turn}')
        #########################################################################
        
        if turn == 0 and particles.start_tracking_at_element < 0:
            line.track(particles, ele_start=start_element, num_turns=1)
        else:
            line.track(particles, num_turns=1)

        if particles._num_active_particles == 0:
            print(f'All particles lost by turn {turn}, teminating.')
            break

    print(f'Tracking {nturns} turns done in: {time.time()-t0} s')
    ####################################################
    # ax.set_xlabel('y [m]')
    # ax.set_ylabel('py [rad]')
    # ax.legend()
    # plt.savefig('dist_turns.png', bbox_inches='tight')
    # # plt.show()
    # # raise SystemExit()
    ########################################################################
    particles_before = particles.copy()

    aper_interp = config_dict['run']['aperture_interp']
    if aper_interp is not None:
        loss_refiner = xt.LossLocationRefinement(line, n_theta=360,
                                                 r_max=1, dr=50e-6,
                                                 ds=aper_interp, 
                                                 allowed_backtrack_types=(xt.elements.Multipole,
                                                                          xt.elements.DipoleEdge,
                                                                          xt.elements.Marker, 
                                                                          xt.elements.Cavity))
        loss_refiner.refine_loss_location(particles)

    if ('lossmap' in config_dict
            and config_dict['lossmap'].get('make_lossmap', False)):
        binwidth = config_dict['lossmap']['aperture_binwidth']
        weights = config_dict['lossmap'].get('weights', 'none')
        lossmap_data = prepare_lossmap(
            particles, line, s0, binwidth=binwidth, weights=weights)
    else:
        lossmap_data = None

    output_file = config_dict['run'].get('outputfile', 'part.hdf')
    _save_particles_hdf(
        particles, lossmap_data=lossmap_data, filename=output_file)

    # Save another file with uninterpolated losses
    # DEBUG
    if ('lossmap' in config_dict
        and config_dict['lossmap'].get('make_lossmap', False)):
        nointerp_filename = output_file.replace('.hdf', '') + '_nointerp.hdf' 
        binwidth = config_dict['lossmap']['aperture_binwidth']
        weights = config_dict['lossmap'].get('weights', 'none')
        nointerp_lossmap_data = prepare_lossmap(
            particles_before, line, s0, binwidth=binwidth,  weights=weights)
        _save_particles_hdf(particles_before, lossmap_data=nointerp_lossmap_data,
                            filename=nointerp_filename)


def load_output(directory, output_file, match_pattern='*part.hdf*',
                imax=None, load_lossmap=True, load_particles=False):

    t0 = time.time()

    job_dirs = glob.glob(os.path.join(directory, 'Job.*')
                         )  # find directories to loop over

    job_dirs_sorted = []
    for i in range(len(job_dirs)):
        # Very inefficient, but it sorts the directories by their numerical index
        job_dir_idx = job_dirs.index(
            os.path.join(directory, 'Job.{}'.format(i)))
        job_dirs_sorted.append(job_dirs[job_dir_idx])

    part_hdf_files = []
    part_dataframes = []
    lossmap_dicts = []

    print(f'Parsing directories...')
    dirs_visited = 0
    files_loaded = 0
    for i, d in enumerate(job_dirs_sorted):
        if imax is not None and i > imax:
            break

        #print(f'Processing {d}')
        dirs_visited += 1
        output_dir = os.path.join(d, 'Outputdata')
        output_files = glob.glob(os.path.join(output_dir, match_pattern))
        if output_files:
            of = output_files[0]
            part_hdf_files.append(of)
            files_loaded += 1
        else:
            print(f'No output found in {d}')

    part_merged = None
    if load_particles:
        print(f'Loading particles...')
        p = Pool()
        part_dataframes = p.map(_read_particles_hdf, part_hdf_files)
        part_objects = [xp.Particles.from_pandas(
            pdf) for pdf in part_dataframes]
        print('Particles load finished, merging...')
        part_merged = xp.Particles.merge(part_objects)

    # Load the loss maps
    lmd_merged = None
    if load_lossmap:
        print(f'Loading loss map data...')
        p = Pool()
        lossmap_dicts = p.map(_load_lossmap_hdf, part_hdf_files)

        print('Loss map load finished, merging..')

        num_tol = 1e-9
        lmd_merged = lossmap_dicts[0]
        for lmd in lossmap_dicts[1:]:
            # Scalar parameters
            # Ensure consistency
            identical_params = ('s_min', 's_max', 'binwidth', 'nbins')
            identical_strings = ('weights',)
            for vv in identical_params:
                assert np.isclose(lmd_merged['lossmap_scalar'][vv],
                                  lmd['lossmap_scalar'][vv],
                                  num_tol)
            for vv in identical_strings:
                assert np.all(lmd_merged['lossmap_scalar'][vv] == lmd['lossmap_scalar'][vv])

            lmd_merged['lossmap_scalar']['n_primaries'] += lmd['lossmap_scalar']['n_primaries']

            # Collimator losses
            # These cannot be empty dataframes even if there is no losses
            assert np.allclose(lmd_merged['lossmap_coll']['coll_start'],
                               lmd['lossmap_coll']['coll_start'],
                               atol=num_tol)

            assert np.allclose(lmd_merged['lossmap_coll']['coll_end'],
                               lmd['lossmap_coll']['coll_end'],
                               atol=num_tol)
            
            assert np.array_equal(lmd_merged['lossmap_coll']['coll_element_index'],
                                  lmd['lossmap_coll']['coll_element_index'])
            
            assert np.array_equal(lmd_merged['lossmap_coll']['coll_name'],
                                  lmd['lossmap_coll']['coll_name'])

            lmd_merged['lossmap_coll']['coll_loss'] += lmd['lossmap_coll']['coll_loss']

            # Aperture losses
            alm = lmd_merged['lossmap_aper']
            al = lmd['lossmap_aper']

            # If the aperture loss dataframe is empty, it is not stored on HDF
            if al is not None:
                if alm is None:
                    lmd_merged['lossmap_aper'] = al
                else:
                    lm = alm.aper_loss.add(al.aper_loss, fill_value=0)
                    lmd_merged['lossmap_aper'] = pd.DataFrame(
                        {'aper_loss': lm})

    _save_particles_hdf(particles=part_merged,
                        lossmap_data=lmd_merged, filename=output_file)

    print('Directories visited: {}, files loaded: {}'.format(
        dirs_visited, files_loaded))
    print(f'Processing done in {time.time() -t0} s')


def prepare_lossmap(particles, line, s0, binwidth, weights):
    lossmap_weights = ['none', 'energy']
    if weights not in lossmap_weights:
        raise ValueError('weights must be in [{}]'.format(', '.join(lossmap_weights)))

    s_ele = np.array(line.get_s_elements(mode='downstream'))
    max_s = max(s_ele)
    s_range = (0, max_s)  # Get the end point as assume the start point is zero

    coll_idx = []
    for idx, elem in enumerate(line.elements):
        if isinstance(elem, xt.BeamInteraction) and isinstance(elem.interaction_process, cs.Geant4Collimator):
            coll_idx.append(idx)
    coll_idx = np.array(coll_idx)
    coll_names = np.take(line.element_names, coll_idx)

    # Ignore unallocated array slots
    mask_allocated = particles.state > -9999
    particles = particles.filter(mask_allocated)

    # Count the number of primary particles for information
    mask_prim = particles.parent_particle_id == particles.particle_id
    n_prim = len(particles.filter(mask_prim).x)

    # Select particles same as the beam particle
    mask_part_type = abs(particles.chi - 1) < 1e-7
    mask_lost = particles.state <= 0
    particles = particles.filter(mask_part_type & mask_lost)

    # Get a mask for the collimator losses
    mask_losses_coll = np.in1d(particles.at_element, coll_idx)
    #mask_warm = np.array([check_warm_loss(s, HLLHC_WARM_REGIONS) for s in particles.s])

    if weights == 'energy':
        part_mass_ratio = particles.charge_ratio / particles.chi
        part_mom = (particles.delta + 1) * particles.p0c * part_mass_ratio
        part_mass = (particles.charge_ratio / particles.chi) * particles.mass0
        part_tot_energy = np.sqrt(part_mom**2 + part_mass**2)
        histo_weights = part_tot_energy
    elif weights == 'none':
        histo_weights = np.full_like(particles.x, 1)
    else:
        raise ValueError('weights must be in [{}]'.format(', '.join(lossmap_weights)))

    # Collimator losses binned per element
    h_coll, edges_coll = np.histogram(particles.at_element[mask_losses_coll],
                                      bins=range(max(coll_idx)+2),
                                      weights=histo_weights[mask_losses_coll])

    # Process the collimator per element histogram for plotting
    coll_lengths = np.array([line.elements[ci].length for ci in coll_idx])
    # reduce the empty bars in the histogram
    coll_values = np.take(h_coll, coll_idx)

    coll_end = np.take(s_ele, coll_idx)
    coll_start = coll_end - coll_lengths

    # Aperture losses binned in S
    nbins_ap = int(np.ceil((s_range[1] - s_range[0])/binwidth))
    bins_ap = np.linspace(s_range[0], s_range[1], nbins_ap)

    aper_loss, _ = np.histogram(particles.s[~mask_losses_coll],
                                bins=bins_ap,
                                weights=histo_weights[~mask_losses_coll])

    # Prepare structures for optimal storage
    aper_loss_series = pd.Series(aper_loss)

    # Scalar variables go in their own DF to avoid replication
    # The bin edges can be re-generated with linspace, no need to store
    scalar_dict = {
        'binwidth': binwidth,
        'weights': weights,
        'nbins': nbins_ap,
        's_min': s_range[0],
        's_max': s_range[1],
        'n_primaries': n_prim,
    }

    # Drop the zeros while preserving the index
    aperloss_dict = {
        'aper_loss': aper_loss_series[aper_loss_series > 0],
    }

    coll_dict = {
        'coll_name': coll_names,
        'coll_element_index': coll_idx,
        'coll_start': coll_start,
        'coll_end': coll_end,
        'coll_loss': coll_values
    }

    scalar_df = pd.DataFrame(scalar_dict, index=[0])
    coll_df = pd.DataFrame(coll_dict)
    aper_df = pd.DataFrame(aperloss_dict)

    lm_dict = {'lossmap_scalar': scalar_df,
               'lossmap_aper': aper_df,
               'lossmap_coll': coll_df}

    return lm_dict


def plot_lossmap(lossmap_file, extra_ranges=None, norm='total'):
    lossmap_norms = ['none', 'max', 'coll_max','total', 'coll_total']
    if norm not in lossmap_norms:
        raise ValueError('norm must be in [{}]'.format(', '.join(lossmap_norms)))

    # A dict of three dataframes
    lossmap_data = _load_lossmap_hdf(lossmap_file)

    lms = lossmap_data['lossmap_scalar']
    lma = lossmap_data['lossmap_aper']
    lmc = lossmap_data['lossmap_coll']

    s_min = lms['s_min'][0]
    s_max = lms['s_max'][0]
    nbins = lms['nbins'][0]
    binwidth = lms['binwidth'][0]
    s_range = (s_min, s_max)

    # Collimator losses
    coll_start = lmc['coll_start']
    coll_end = lmc['coll_end']
    coll_values = lmc['coll_loss']

    coll_lengths = coll_end - coll_start
    
    if norm == 'total':
        norm_val = sum(coll_values) + sum(lma['aper_loss'])
    elif norm == 'max':
        norm_val = max(max(coll_values), max(lma['aper_loss']))
    elif norm == 'coll_total':
        norm_val = sum(coll_values)
    elif norm == 'coll_max':
        norm_val = max(coll_values)
    elif norm == 'none':
        norm_val = 1

    coll_values /= (norm_val * coll_lengths)

    # There can be an alternative way of plotting using a bar plot
    # Make the correct edges to get the correct width of step plot
    # The dstack and flatten merges the arrays one set of values at a time
    zeros = np.full_like(lmc.index, 0)  # Zeros to pad the bars
    coll_edges = np.dstack(
        [coll_start, coll_start, coll_end, coll_end]).flatten()
    coll_loss = np.dstack([zeros, coll_values, coll_values, zeros]).flatten()

    # Aperture losses
    aper_edges = np.linspace(s_min, s_max, nbins)

    aper_loss = lma['aper_loss'].reindex(range(0, nbins-1), fill_value=0)

    aper_loss /= (norm_val * binwidth)

    # Check if the start of the bin is in a warm region
    #warm_regions = HLLHC_WARM_REGIONS
    warm_regions = FCC_EE_WARM_REGIONS
    mask_warm = np.array([check_warm_loss(s, warm_regions)
                         for s in aper_edges[:-1]])

    warm_loss = aper_loss * mask_warm
    cold_loss = aper_loss * ~mask_warm

    # Make the plots
    fig, ax = plt.subplots(figsize=(12, 4))

    # The zorder determines the plotting order = collimator -> warm-> cold (on top)
    # The edge lines on the plots provide the dynamic scaling feature of the plots
    # e.g a 10 cm aperture loss spike is still resolvable for a full ring view
    lw = 1
    ax.stairs(warm_loss, aper_edges, color='r',
              lw=lw, ec='r', fill=True, zorder=20)
    ax.stairs(cold_loss, aper_edges, color='b',
              lw=lw, ec='b', fill=True, zorder=30)

    ax.fill_between(coll_edges, coll_loss, step='pre', color='k', zorder=9)
    ax.step(coll_edges, coll_loss, color='k', lw=lw, zorder=10)

    plot_margin = 500
    ax.set_xlim(s_range[0] - plot_margin, s_range[1] + plot_margin)
    ax.set_ylim(1e-6, 2)

    ax.yaxis.grid(visible=True, which='major', zorder=0)
    ax.yaxis.grid(visible=True, which='minor', zorder=0)

    ax.set_yscale('log', nonpositive='clip')
    ax.set_xlabel(r'S [$m$]')
    ax.set_ylabel(r'Cleaning inefficiency [$m^{-1}$]')

    # plt.show()
    # return

    plt.savefig('lossmap_full.pdf', bbox_inches='tight')

    if extra_ranges is not None:
        for srange in extra_ranges:
            ax.set_xlim(srange)
            plt.savefig("lossmap_{}_{}.pdf".format(srange[0], srange[1]))


def rebin(part_file, outfile, line, s0, binwidth, weights):
    particles = _load_particles_hdf(part_file)
    lossmap_data = prepare_lossmap(
        particles, line, s0, binwidth=binwidth, weights=weights)
    _save_particles_hdf(particles=particles,
                        lossmap_data=lossmap_data, filename=outfile)


def execute(config_dict):
    config_dict = CONF_SCHEMA.validate(config_dict)
    
    line, ref_part, start_elem, s0 = load_and_process_line(config_dict)
    build_collimation_tracker(line)

    particles = prepare_particles(config_dict, line, ref_part)

    output_file = config_dict['run']['outputfile']

    # modifies the Particles object in place
    run(config_dict, line, particles, ref_part, start_elem, s0)

    plot_lossmap(output_file, extra_ranges=[(33000, 35500), (44500, 46500)], norm='total')


@contextmanager
def set_directory(path: Path):
    """
    Taken from: https://dev.to/teckert/changing-directory-with-a-python-context-manager-2bj8
    """
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


def resolve_and_cache_paths(iterable_obj, resolved_iterable_obj, cache_destination):
    if isinstance(iterable_obj, (dict, list)):
        for k, v in (iterable_obj.items() if isinstance(iterable_obj, dict) else enumerate(iterable_obj)):
            possible_path = Path(str(v))
            if not isinstance(v, (dict, list)) and possible_path.exists() and possible_path.is_file():
                shutil.copy(possible_path, cache_destination)
                resolved_iterable_obj[k] = possible_path.name
            resolve_and_cache_paths(v, resolved_iterable_obj[k], cache_destination)


def dump_dict_to_yaml(dict_obj, file_path):
        with open(file_path, 'w') as yaml_file:
            yaml.dump(dict_obj, yaml_file, 
                      default_flow_style=False, sort_keys=False)


def submit_jobs(config_dict, config_file):
    # Relative path from the config file should be relative to
    # the file itself, not to where the script is executed from
    if config_file:
        conf_path = Path(config_file).resolve()
        conf_dir = conf_path.parent
        conf_fname = conf_path.name
    else:
        conf_dir = Path().resolve()
        conf_fname = 'config_collimation.yaml'
        conf_path = Path(conf_dir, conf_fname)
        
    with set_directory(conf_dir):
        config_dict = CONF_SCHEMA.validate(config_dict)

        sub_dict = config_dict['jobsubmission']
        workdir = Path(sub_dict['working_directory']).resolve()
        num_jobs = sub_dict['num_jobs']
        replace_dict_in = sub_dict.get('replace_dict', {})
        executable = sub_dict.get('executable', 'bash')
        mask_abspath = Path(sub_dict['mask']).resolve()
        
        max_local_jobs = 10
        if sub_dict.get('run_local', False) and num_jobs > max_local_jobs:
            raise Exception(f'Cannot run more than {max_local_jobs} jobs locally,'
                            f' {num_jobs} requested.')
            
        # Make a directory to copy the files for the submission
        input_cache = Path(workdir, 'input_cache')
        os.makedirs(workdir)
        os.makedirs(input_cache)

        # Copy the files to the cache and replace the path in the config
        # Copy the configuration file
        if conf_path.exists():
            shutil.copy(conf_path, input_cache)
        else:
            # If the setup came from a dict a dictionary still dump it to archive
            dump_dict_to_yaml(config_dict, Path(input_cache, conf_path.name))
            
        exclude_keys = {'jobsubmission',} # The submission block is not needed for running
        # Preserve the key order
        reduced_config_dict = {k: config_dict[k] for k in 
                               config_dict.keys() if k not in exclude_keys}
        resolved_config_dict = copy.deepcopy(reduced_config_dict)
        resolve_and_cache_paths(reduced_config_dict, resolved_config_dict, input_cache)

        resolved_conf_file = f'for_jobs_{conf_fname}' # config file used to run each job
        dump_dict_to_yaml(resolved_config_dict, Path(input_cache, resolved_conf_file))

        # compress the input cache to reduce network traffic
        shutil.make_archive(input_cache, 'gztar', input_cache)
        # for fpath in input_cache.iterdir():
        #     fpath.unlink()
        # input_cache.rmdir()

        # Set up the jobs
        seeds = np.arange(num_jobs) + 1 # Start the seeds at 1
        replace_dict_base = {'seed': seeds.tolist(),
                             'config_file': resolved_conf_file,
                             'input_cache_archive': str(input_cache) + '.tar.gz'}

        # Pass through additional replace dict option and other job_submitter flags
        if replace_dict_in:
            replace_dict = {**replace_dict_base, **replace_dict_in}
        else:
            replace_dict = replace_dict_base
        
        processed_opts = {'working_directory', 'num_jobs', 'executable', 'mask'}
        submitter_opts = list(set(sub_dict.keys()) - processed_opts)
        submitter_options_dict = { op: sub_dict[op] for op in submitter_opts }
        
        # Send/run the jobs via the job_submitter interface
        htcondor_submit(
            mask=mask_abspath,
            working_directory=workdir,
            executable=executable,
            replace_dict=replace_dict,
            **submitter_options_dict)

        print('Done!')

    
def main():
    if len(sys.argv) != 3:
        raise ValueError(
            'The script only takes two inputs: the mode and the target')

    if sys.argv[1] == '--run':
        if 'collimasim' not in sys.modules:
            raise SystemExit('collimasim not imported, cannot run.')
        t0 = time.time()
        config_file = sys.argv[2]
        config_dict = load_config(config_file)
        execute(config_dict)
        print(f'Done! Time taken: {time.time()-t0} s')
    elif sys.argv[1] == '--submit':
        config_file = sys.argv[2]
        config_dict = load_config(config_file)
        submit_jobs(config_dict, config_file)
    elif sys.argv[1] == '--merge':
        match_pattern = '*part.hdf*'
        output_file = 'part_merged.hdf'
        load_output(sys.argv[2], output_file, match_pattern=match_pattern, load_particles=True)
    elif sys.argv[1] == '--plot':
        plot_lossmap(sys.argv[2], extra_ranges=[(33000, 35500), (44500, 46500)], norm='total')
    else:
        raise ValueError('The mode must be one of --run, --submit, --merge, --plot')

if __name__ == '__main__':
    # with open('output.log', 'w') as of:
    #    with redirect_stdout(of):
    main()
