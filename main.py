#!/usr/bin/env python
import os
import glob
import shutil
import vmecPlot2
import numpy as np
from mpi4py import MPI
from pathlib import Path
from simsopt import make_optimizable
from simsopt.mhd import Vmec
from simsopt.util import MpiPartition
from simsopt.solve import least_squares_mpi_solve
from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from objectives import MaxElongationPen
import time
def pprint(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:  # only pprint on rank 0
        print(*args, **kwargs)
mpi = MpiPartition()
############################################################################
#### Input Parameters
############################################################################
MAXITER = 30
max_modes = [2, 3]
mean_shear_target = -0.6
shear_weight = 1e2
elongation_weight = 1e0
QA_or_QH_or_warm = 'warm' # define initial input VMEC file
rel_step = 1e-2
abs_step = 1e-5
minimum_iota = 0.35
min_iota_weight = 1e4
aspect_ratio_target = 3 if QA_or_QH_or_warm=='QA' else 4
aspect_ratio_weight = 1e-2
opt_quasisymmetry = False
boozxform_nsurfaces=10
ftol=1e-3
######################################
######################################
if QA_or_QH_or_warm == 'QA': filename = os.path.join(os.path.dirname(__file__), 'input.nfp2_QA')
elif QA_or_QH_or_warm == 'QH': filename = os.path.join(os.path.dirname(__file__), 'input.nfp4_QH')
else: filename = os.path.join(os.path.dirname(__file__), 'input.warm_start')
vmec = Vmec(filename, mpi=mpi, verbose=False)
surf = vmec.boundary
######################################
OUT_DIR=os.path.join(Path(__file__).parent.resolve(),f'out_meanShear{mean_shear_target}_nfp{vmec.indata.nfp}_{QA_or_QH_or_warm}')
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
######################################
def elongationCostFunction(v):
    v.run()
    nphi = 7
    elongation = MaxElongationPen(v,ntheta=19,nphi=nphi, return_elongation=True)
    return elongation
def shearCostFunction(v):
    v.run()
    mean_shear = v.mean_shear()
    return mean_shear
def iotaCostFunction(v):
    v.run()
    iotas = np.abs(np.array(v.wout.iotas[1:]))
    return np.max(np.concatenate(([0],minimum_iota-iotas)))
optElongationCostFunction = make_optimizable(elongationCostFunction, vmec)
optShearCostFunction = make_optimizable(shearCostFunction, vmec)
optaIotaCostFunction = make_optimizable(iotaCostFunction, vmec)
######################################
pprint("Initial aspect ratio:", vmec.aspect())
pprint("Initial max elongation:", np.max(optElongationCostFunction.J()))
pprint("Initial mean_shear", optShearCostFunction.J())
pprint("Initial iota_axis:", vmec.iota_axis())
pprint("Initial iota_edge:", vmec.iota_edge())
######################################
if QA_or_QH_or_warm == 'QH': qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)
else: qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)    
opt_tuple = [(vmec.aspect, aspect_ratio_target, aspect_ratio_weight)]
opt_tuple = []
opt_tuple.append((optElongationCostFunction.J, 0, elongation_weight))
opt_tuple.append((optShearCostFunction.J, mean_shear_target, shear_weight))
opt_tuple.append((optaIotaCostFunction.J, 0, min_iota_weight))
if opt_quasisymmetry: opt_tuple.append((qs.residuals, 0, 1))
######################################
for max_mode in max_modes:
    pprint('-------------------------')
    pprint(f'Optimizing with max_mode = {max_mode}')
    pprint('-------------------------')
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    ######################################
    prob = LeastSquaresProblem.from_tuples(opt_tuple)
    pprint("Total objective before optimization:", prob.objective())
    least_squares_mpi_solve(prob, mpi, grad=True, rel_step=rel_step, abs_step=abs_step, max_nfev=MAXITER, ftol=ftol)
    ######################################
    pprint("Final aspect ratio:", vmec.aspect())
    pprint("Final max elongation:", np.max(optElongationCostFunction.J()))
    pprint("Final mean_shear", optShearCostFunction.J())
    pprint("Final iota_axis:", vmec.iota_axis())
    pprint("Final iota_edge:", vmec.iota_edge())
    pprint("Total objective after optimization:", prob.objective())
    ######################################
try:
    for objective_file in glob.glob("objective_*"):
        os.remove(objective_file)
    for residuals_file in glob.glob("residuals_*"):
        os.remove(residuals_file)
    for jac_file in glob.glob("jac_log_*"):
        os.remove(jac_file)
    for threed_file in glob.glob("threed1.*"):
        os.remove(threed_file)
except Exception as e:
    pprint(e)
######################################
time.sleep(2)
vmec.write_input(os.path.join(OUT_DIR, f'input.final'))
vmec_final = Vmec(os.path.join(OUT_DIR, f'input.final'), mpi=mpi, verbose=False)
vmec_final.indata.ns_array[:3]    = [  16,    51,    101]
vmec_final.indata.niter_array[:3] = [ 4000, 10000, 10000]
vmec_final.indata.ftol_array[:3]  = [1e-12, 1e-13, 1e-14]
vmec_final.run()
if mpi.proc0_world:
    shutil.move(os.path.join(OUT_DIR, f"wout_final_000_000000.nc"), os.path.join(OUT_DIR, f"wout_final.nc"))
    os.remove(os.path.join(OUT_DIR, f'input.final_000_000000'))
    try: vmecPlot2.main(file=os.path.join(OUT_DIR, f"wout_final.nc"), name='EP_opt', figures_folder=OUT_DIR)
    except Exception as e: print(e)
############################################################################
############################################################################