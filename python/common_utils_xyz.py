#
# Additional functions used for cases with non-uniform nk in the xyz directions.
# Users can use nkx, nky, and nkz instead an equal nk in each dimension. 
#

import argparse
import h5py
import numpy as np
import os
import pyscf.lib.chkfile as chk
from numba import jit
from pyscf import gto as mgto
from pyscf.pbc import tools, gto, df, scf, dft

import integral_utils as int_utils
import common_utils as comm


def save_data_xyz(args, mycell, mf, kmesh, ind, weight, num_ik, ir_list, conj_list, Nk, nkx, nky, nkz, NQ, F, S, T, hf_dm, 
              Zs, last_ao):
    inp_data = h5py.File(args.output_path, "w")
    inp_data["grid/k_mesh"] = kmesh
    inp_data["grid/k_mesh_scaled"] = mycell.get_scaled_kpts(kmesh)
    inp_data["grid/index"] = ind
    inp_data["grid/weight"] = weight
    inp_data["grid/ink"] = num_ik
    inp_data["grid/nkx"] = nkx
    inp_data["grid/nky"] = nky
    inp_data["grid/nkz"] = nkz
    inp_data["grid/ir_list"] = ir_list
    inp_data["grid/conj_list"] = conj_list
    inp_data["HF/Nk"] = Nk
    inp_data["HF/nkx"] = nkx
    inp_data["HF/nky"] = nky
    inp_data["HF/nkz"] = nkz
    inp_data["HF/Energy"] = mf.e_tot
    inp_data["HF/Energy_nuc"] = mf.cell.energy_nuc()
    inp_data["HF/Fock-k"] = F.view(np.float64).reshape(F.shape[0], F.shape[1], F.shape[2], F.shape[3], 2)
    inp_data["HF/Fock-k"].attrs["__complex__"] = np.int8(1)
    inp_data["HF/S-k"] = S.view(np.float64).reshape(S.shape[0], S.shape[1], S.shape[2], S.shape[3], 2)
    inp_data["HF/S-k"].attrs["__complex__"] = np.int8(1)
    inp_data["HF/H-k"] = T.view(np.float64).reshape(T.shape[0], T.shape[1], T.shape[2], T.shape[3], 2)
    inp_data["HF/H-k"].attrs["__complex__"] = np.int8(1)
    inp_data["HF/madelung"] = tools.pbc.madelung(mycell, kmesh)
    inp_data["HF/mo_energy"] = mf.mo_energy
    inp_data["HF/mo_coeff"] = mf.mo_coeff
    inp_data["mulliken/Zs"] = Zs
    inp_data["mulliken/last_ao"] = last_ao
    inp_data["params/nao"] = S.shape[2]
    inp_data["params/nso"] = S.shape[2]
    inp_data["params/ns"] = S.shape[0]
    inp_data["params/nel_cell"] = mycell.nelectron
    inp_data["params/nkx"] = kmesh.shape[0]
    inp_data["params/nky"] = kmesh.shape[1]
    inp_data["params/nkz"] = kmesh.shape[2]
    inp_data["params/NQ"] = NQ
    inp_data.close()
    chk.save(args.output_path, "Cell", mycell.dumps())
    inp_data = h5py.File("dm.h5", "w")
    inp_data["HF/dm-k"] = hf_dm.view(np.float64).reshape(hf_dm.shape[0], hf_dm.shape[1], hf_dm.shape[2], hf_dm.shape[3], 2)
    inp_data["HF/dm-k"].attrs["__complex__"] = np.int8(1)
    inp_data["dm_gamma"] = hf_dm[:, 0, :, :]
    inp_data.close()


def add_common_params_xyz(parser):
    """
    An alternative argument parser for different nk in xyz directions.
    """
    parser.add_argument("--a", type=comm.parse_geometry, help="lattice geometry", required=True)
    parser.add_argument("--atom", type=comm.parse_geometry, help="poistions of atoms", required=True)
    parser.add_argument("--nkx", type=int, help="number of k-points in each direction", required=True)
    parser.add_argument("--nky", type=int, help="number of k-points in each direction", required=True)
    parser.add_argument("--nkz", type=int, help="number of k-points in each direction", required=True)
    parser.add_argument("--symm", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default='true', help="Use inversion symmetry")
    parser.add_argument("--Nk", type=int, default=0, help="number of plane-waves in each direction for integral evaluation")
    parser.add_argument("--basis", type=str, nargs="*", help="basis sets definition. First specify atom then basis for this atom", required=True)
    parser.add_argument("--auxbasis", type=str, nargs="*", default=[None], help="auxiliary basis")
    parser.add_argument("--ecp", type=str, nargs="*", default=[None], help="effective core potentials")
    parser.add_argument("--pseudo", type=str, nargs="*", default=[None], help="pseudopotential")
    parser.add_argument("--shift", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="mesh shift")
    parser.add_argument("--center", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="mesh center")
    parser.add_argument("--xc", type=str, nargs="*", default=[None], help="XC functional")
    parser.add_argument("--dm0", type=str, nargs=1, default=None, help="initial guess for density matrix")
    parser.add_argument("--df_int", type=int, default=1, help="prepare density fitting integrals or not")
    parser.add_argument("--int_path", type=str, default="df_int", help="path to store ewald corrected integrals")
    parser.add_argument("--hf_int_path", type=str, default="df_hf_int", help="path to store hf integrals")
    parser.add_argument("--output_path", type=str, default="input.h5", help="output file with initial data")
    parser.add_argument("--orth", type=int, default=0, help="Transform to orthogonal basis or not. 0 - no orthogonal transformation, 1 - data is in orthogonal basis.")
    parser.add_argument("--beta", type=float, default=None, help="Emperical parameter for even-Gaussian auxiliary basis")
    parser.add_argument("--active_space", type=int, nargs='+', default=None, help="active space orbitals")
    parser.add_argument("--spin", type=int, default=0, help="Local spin")
    parser.add_argument("--restricted", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default='false', help="Spin restricted calculations.")
    parser.add_argument("--print_high_symmetry_points", default=False, action='store_true', help="Print available high symmetry points for current system and exit.")
    parser.add_argument("--high_symmetry_path", type=str, default=None, help="High symmetry path")
    parser.add_argument("--high_symmetry_path_points", type=int, default=0, help="Number of points for high symmetry path")
    parser.add_argument("--memory", type=int, default=700, help="Memory bound for integral chunk in MB")
    parser.add_argument("--diffuse_cutoff", type=float, default=0.0, help="Remove the diffused Gaussians whose exponents are less than the cutoff")
    parser.add_argument("--damping", type=float, default=0.0, help="Damping factor for mean-field iterations")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations in the SCF loop")

def init_dca_params(a, atoms):
    parser = argparse.ArgumentParser(description="GF2 initialization script")
    add_common_params_xyz(parser, a, atoms)
    parser.add_argument("--lattice_size", type=int, default=3, help="size of the super lattice in each direction")
    parser.add_argument("--interaction_lattice_size", type=int, default=3, help="size of the super lattice mesh for Coulomb interaction in each direction")
    parser.add_argument("--interaction_lattice_point_i", type=int, default=0, help="first interction momentum index")
    parser.add_argument("--interaction_lattice_point_j", type=int, default=0, help="second interction momentum index")
    parser.add_argument("--keep", type=int, default=0, help="keep cderi files")
    parser.add_argument("--regenerate", type=int, default=0, help="regenerate integrals")
    args = parser.parse_args()
    args.basis = parse_basis(args.basis)
    args.auxbasis = parse_basis(args.auxbasis)
    args.ecp = parse_basis(args.ecp)
    args.pseudo = parse_basis(args.pseudo)
    args.xc = parse_basis(args.xc)
    if args.xc is not None:
        args.mean_field = dft.KRKS if args.restricted else dft.KUKS
    else:
        args.mean_field = scf.KRHF if args.restricted else scf.KUHF
    args.ns = 1 if args.restricted else 2
    return args


def init_pbc_params():
    parser = argparse.ArgumentParser(description="GF2 initialization script")
    add_common_params_xyz(parser)
    args = parser.parse_args()
    args.basis = comm.parse_basis(args.basis)
    args.auxbasis = comm.parse_basis(args.auxbasis)
    args.ecp = comm.parse_basis(args.ecp)
    args.pseudo = comm.parse_basis(args.pseudo)
    args.xc = comm.parse_basis(args.xc)
    if args.xc is not None:
        args.mean_field = dft.KRKS if args.restricted else dft.KUKS
    else:
        args.mean_field = scf.KRHF if args.restricted else scf.KUHF
    args.ns = 1 if args.restricted else 2
    return args


def init_k_mesh_xyz(args, mycell):
    '''
    init k-points mesh for GDF

    :param args: script arguments
    :param mycell: unit cell for simulation
    :return: kmesh,
    '''
    kmesh = mycell.make_kpts([args.nkx, args.nky, args.nkz], scaled_center=args.center)
    for i, kk in enumerate(kmesh):
        ki = kmesh[i]
        ki = mycell.get_scaled_kpts(ki) + args.shift
        ki = [comm.wrap_k(l) for l in ki]
        kmesh[i] = mycell.get_abs_kpts(ki)
    for i, ki in enumerate(kmesh):
        ki = mycell.get_scaled_kpts(ki)
        ki = [comm.wrap_k(l) for l in ki]
        ki = mycell.get_abs_kpts(ki)
        kmesh[i] = ki

    print(kmesh)
    print(mycell.get_scaled_kpts(kmesh))

    if not args.symm :
        nkpts = kmesh.shape[0]
        weight = np.ones(nkpts)
        ir_list = np.array(range(nkpts))
        ind = np.array(range(nkpts))
        conj_list = np.zeros(nkpts)
        k_ibz = np.copy(kmesh)
        num_ik = nkpts
        return kmesh, k_ibz, ir_list, conj_list, weight, ind, num_ik

    print("Compute irreducible k-points")


    k_ibz = mycell.make_kpts([args.nkx, args.nky, args.nkz], scaled_center=args.center)
    ind = np.arange(np.shape(k_ibz)[0])
    weight = np.zeros(np.shape(k_ibz)[0])
    for i, ki in enumerate(k_ibz):
        ki = mycell.get_scaled_kpts(ki)
        ki = [comm.wrap_1stBZ(l) for l in ki]
        k_ibz[i] = ki

    # Time-reversal symmetry
    Inv = (-1) * np.identity(3)
    for i, ki in enumerate(k_ibz):
        ki = np.dot(Inv,ki)
        ki = [comm.wrap_1stBZ(l) for l in ki]
        for l, kl in enumerate(k_ibz[:i]):
            if np.allclose(ki,kl):
                k_ibz[i] = kl
                ind[i] = l
                break

    uniq = np.unique(ind, return_counts=True)
    for i, k in enumerate(uniq[0]):
        weight[k] = uniq[1][i]
    ir_list = uniq[0]

    # Mark down time-reversal-reduced k-points
    conj_list = np.zeros(args.nkx * args.nky * args.nkz)
    for i, k in enumerate(ind):
        if i != k:
            conj_list[i] = 1
    num_ik = np.shape(uniq[0])[0]

    return kmesh, k_ibz, ir_list, conj_list, weight, ind, num_ik


def init_lattice_mesh_xyz(args, mycell, kmesh, L=None):
    if L is None:
        L = args.lattice_size
    nao = mycell.nao_nr()
    b = mycell.reciprocal_vectors()
    lattice = np.copy(b)
    lattice[0] = b[0] / args.nkx
    lattice[1] = b[1] / args.nky
    lattice[2] = b[2] / args.nkz
    print(b)
    print(lattice)

    lattice_kmesh = comm.lattice_points(lattice, L)

    full_mesh = np.zeros([args.nkx*L*args.nky*L*args.nkz*L, 3])
    print(lattice_kmesh.shape)
    for iK, K in enumerate(kmesh):
        for ik, k in enumerate(lattice_kmesh):
            full_mesh[iK*(L**3) + ik] = K + k

    print(full_mesh.shape)

    H0_lattice = np.zeros([kmesh.shape[0], lattice_kmesh.shape[0], nao, nao], dtype=np.complex128)
    S_lattice = np.zeros([kmesh.shape[0], lattice_kmesh.shape[0], nao, nao], dtype=np.complex128)

    new_mf    = dft.KUKS(mycell,full_mesh).density_fit()
    H0kl = new_mf.get_hcore()
    print(H0kl.shape)
    H0_lattice[:, :, :, :] = H0kl.reshape([kmesh.shape[0], lattice_kmesh.shape[0], nao, nao])
    Skl = new_mf.get_ovlp()
    print(Skl.shape)
    S_lattice[:, :, :, :] = Skl.reshape([kmesh.shape[0], lattice_kmesh.shape[0], nao, nao])
    return lattice_kmesh, full_mesh, H0_lattice, S_lattice

