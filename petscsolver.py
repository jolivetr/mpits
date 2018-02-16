'''
Class solving the problem with the solvers in PETSc

Written by R. Jolivet 2017

License:
    MPITS: Multi-Pixel InSAR Time Series
    Copyright (C) 2018  <Romain Jolivet>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''

import numpy as np
import sys
import scipy.linalg as lm
import scipy.io as sio
import h5py
import datetime as dt
import tsinsar as ts
import os
import copy

import matplotlib.pyplot as plt

class petscsolver(object):

    def __init__(self, massive, solver='lsqr', pc='none', tol=None, norm='l2'):
        '''
        Initializes the solver.
        Args:
            * massive       : Instance of massivets
            * solver        : Solver type for PETSc. Default is lsqr. 
                              Only lsqr can deal with rectangulat matrices, so far...
            * pc            : Pre-conditioner for PETSc. Defualt is none. 
                              See PETSc manual.
            * tol           : [Relative Tolerance, Absolute tolerance, 
                               Diverging threshold, Maximum iteration]
            * norm          : Sets the kind of norm used by petsc to check convergence
                            'l2', 'l1', 'l12' (norm 1 and 2 computed)

        Tolerances are defined such as, for R = ||Ax - b||2, the solver will stop if 
                            R < max(rtol*||b||2,atol). 

        The diverging threshold specifies how much the residuals can increase before 
        it hits KSPDefaultConverged.

        Maximum iteration is the maximum iteration number.

        Options can be specified using the command line arguments. If they are specified 
        in the Kwargs, command line arguments are ineffective.
        '''

        # Get PETSc and the Communicator
        self.PETSc = massive.PETSc
        self.MPI = massive.MPI
        self.Com = massive.MPI.COMM_WORLD

        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Initialize the PETSc solver')

        # Pass some things
        self.d = massive.d
        self.m = massive.m
        self.massive = massive

        # Create the solver
        self.Solver = self.PETSc.KSP().create(comm=self.Com)

        # Command line options 
        self.Solver.setFromOptions()

        # Set additional stuff
        self.Solver.setType(solver)
        self.Solver.pc.setType(pc)

        # Norm type
        if norm in ('l2', '2', 'Frobenius', 'distance'):
            norm = self.PETSc.NormType.FROBENIUS
        elif norm in ('l1', '1', 'sum'):
            norm = self.PETSc.NormType.NORM_1
        elif norm in ('max', 'MAX', 'Max', 'maximum', 'Maximum','l12', '12', 'combined'):
            self.PETSc.Sys.Print(' !!!!!! Unsupported NormType for PETSc lsqr solver !!!!!!')
            self.PETSc.Sys.Print(' !!!!!!         Switching to l2 type norm          !!!!!!')
        else:
            self.PETSc.Sys.Print(' !!!!!! Unknown NormType asked for solver !!!!!!')
            self.PETSc.Sys.Print(' !!!!!!     Switching to l2 type norm     !!!!!!')
        self.Solver.setNormType(norm)

        # Set the tolerances
        if tol is not None:
            self.Solver.setTolerances(tol[0], tol[1], tol[2], tol[3])

        # Set the operator
        self.Solver.setOperators(massive.G)

        # Get the tolerances asked and print some stuffs
        rtol, atol, divit, it = self.Solver.getTolerances()
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Absolute Tolerance: %e'%atol)
        self.PETSc.Sys.Print(' Relative Tolerance: %e'%rtol)
        self.PETSc.Sys.Print(' Divergence Thresh.: %d'%divit)
        self.PETSc.Sys.Print(' Maximum iterations: %i'%it)
        self.PETSc.Sys.Print(' ')

        # All done
        return

    def destroy(self):
        ''' 
        Destroys the solver.
        '''

        # Destroy
        self.Solver.destroy()

        # All done
        return

    def Solve(self, view=False, zeroOutInitial=True):
        '''
        Solve the problem.
        Args:
            * view              : If True, prints out a bunch of stuff from Solver.view()
            * zeroOutInitial    : False, uses self.m as an initial guess
        '''

        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Solving...')

        # Set initial Guess
        if zeroOutInitial:
            self.Solver.setInitialGuessNonzero(0)
        else:
            self.Solver.setInitialGuessNonzero(1)

        # Solve 
        self.Solver(self.d, self.m)

        # Get some infos
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Converged in %d iterations '%(self.Solver.getIterationNumber()))
        self.PETSc.Sys.Print(' Tolerance Asked: %e %e %d %d'%(self.Solver.getTolerances()))
        self.PETSc.Sys.Print(' Residual Norm: {}'.format(self.Solver.getResidualNorm()))
        self.PETSc.Sys.Print(' Converged Reason: %s'%(self.convergedReason(self.Solver.getConvergedReason())))
        self.PETSc.Sys.Print(' ')
        if view:
            self.Solver.view()
        self.PETSc.Sys.Print(' ')

        # Solved
        self.massive.solved = True

        # All done
        return

    def convergedReason(self, index):
        '''
        Translates the converged reason from PETSc to something that makes sense in English.
        Args:
            * index     : PETSc integer.
        '''

        # Converged
        if index == 1:
            reason = "KSP_CONVERGED_RTOL_NORMAL"
        elif index == 9:
            reason = "KSP_CONVERGED_ATOL_NORMAL"
        elif index == 2:
            reason = "KSP_CONVERGED_RTOL"
        elif index == 3:
            reason = "KSP_CONVERGED_ATOL"
        elif index == 4:
            reason = "KSP_CONVERGED_ITS"
        elif index == 5:
            reason = "KSP_CONVERGED_CG_NEG_CURVE"
        elif index == 6:
            reason = "KSP_CONVERGED_CG_CONSTRAINED"
        elif index == 7:
            reason = "KSP_CONVERGED_STEP_LENGTH"
        elif index == 8:
            reason = "KSP_CONVERGED_HAPPY_BREAKDOWN"
        # Diverged
        elif index == -2:
            reason = "KSP_DIVERGED_NULL"
        elif index == -3:
            reason = "KSP_DIVERGED_ITS"
        elif index == -4:
            reason = "KSP_DIVERGED_DTOL"
        elif index == -5:
            reason = "KSP_DIVERGED_BREAKDOWN"
        elif index == -6:
            reason = "KSP_DIVERGED_BREAKDOWN_BICG"
        elif index == -7:
            reason = "KSP_DIVERGED_NONSYMMETRIC"
        elif index == -8:
            reason = "KSP_DIVERGED_INDEFINITE_PC"
        elif index == -9:
            reason = "KSP_DIVERGED_NAN" 
        elif index == -10:
            reason = "KSP_DIVERGED_INDEFINITE_MAT"
        # Last reason
        elif index == 0:
            reason = "KSP_CONVERGED_ITERATING"
        # Unknown reason
        else:
            reason = "UNKNOWN_REASON"

        # All done
        return reason

