'''
Class solving the problem with a Conjugate Direction method.
Algorithm is from Tarantolla, 2005, Inverse Problem Theory, SIAM, Chap.6, p. 218.

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
import itertools
import time
import sys, gc
import scipy.optimize as sciopt

# Myself
from . import utils

class cdsolver(object):

    def __init__(self, massive, dataCovariance, modelCovariance, 
                 orbitVariance=None, iteration=1, mprior=None, preconditioner=None, 
                 rtol = None, atol = None, initialModel=None, 
                 debug=False):
        '''
        Initializes the solver.
        Args:
            * massive               : Instance of massivets
            * dataCovariance        : array/list of tuples of (Lambda; Sigma) for the data covariance
            * modelCovariance       : array/list of tuples of (Lambda; Sigma) for the model covariance
            * orbitVariance         : array/list of variances for the orbit parameters
            * iteration             : number of iterations
            * mprior                : PETSc vector of the a priori model (if None, set to 0.)
            * rtol                  : Stop the code when the change in least square norm is less than rtol between 
                                      2 successive iterations (%)
            * atol                  : Stop the code when the change in least square norm is less than atol (%)
            * initialModel          : Dictionary of the initial model {0: np.array(Ny, Nx), 2: np.array(Ny, Nx),....} 
            * preconditioner        : PETSc Matrix for a preconditioner of the steepest Ascent vector (if None, identity is used).

        The solver is going to iterate over the conjugate direction for least squares expression.
        The covariances are exponential covariances so the matrix multiplication can be done in the Fourier
        domain.
        '''

        self.debug = debug

        # Get PETSc and the Communicator
        self.PETSc = massive.PETSc
        self.MPI = massive.MPI
        self.Com = massive.MPI.COMM_WORLD

        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Initialize the Conjugate Direction for least squares solver')

        # Pass some things
        self.massive = massive
        self.rank = self.massive.rank

        # Talk to me
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print('      Prepare Covariances')

        # Data Covariance
        if type(dataCovariance) in (tuple, float):
            self.Cd = [dataCovariance for i in range(self.massive.Ndata)]
        else:
            assert len(dataCovariance)==self.massive.Ndata,\
                    'Need to provide as many tuples of (Lambda, Sigma) as there is interferograms'
            self.Cd = dataCovariance
        self.dataConvolution = []
        for cd in self.Cd:
            if type(cd) is float:
                self.dataConvolution.append(utils._diagonalConvolution)
            elif type(cd) is tuple:
                assert len(cd)==2, 'Tuple for exponential covariance needs 2 elements...'
                self.dataConvolution.append(utils._expConvolution)
            else:
                assert False, '{}: Unkown covariance type for data...'.format(cd)

        # Model Covariance
        if type(modelCovariance) in (tuple, float):
            self.Cm = [modelCovariance for i in range(self.massive.nParams)]
        else:
            assert len(modelCovariance)==self.massive.nParams, \
                    'Need to provide {} tuples of (Lambda, Sigma)'.format(self.massive.nParams)
            self.Cm = modelCovariance
        self.modelConvolution = []
        for cm in self.Cm:
            if type(cm) is float:
                self.modelConvolution.append(utils._diagonalConvolution)
            elif type(cm) is tuple:
                assert len(cm)==2, 'Tuple for exponential covariance needs 2 elements...'
                self.modelConvolution.append(utils._expConvolution)
            else:
                assert False, '{}: Unkown covariance type for model...'.format(cm)

        # Orbit Variance
        if self.massive.orbit:
            assert orbitVariance is not None, 'You need to provide a variance for orbital parameters'
        nOrb = self.massive.nOrb*self.massive.Nsar+self.massive.Nifg
        if type(orbitVariance) is float:
            self.oCm = [orbitVariance for i in range(nOrb)]
        elif type(orbitVariance) is type(None):
            self.oCm = None
        else:
            assert len(orbitVariance) == nOrb, 'Need to provide as many \
                                                variance values as there is \
                                                orbital parameters ({})'.format(nOrb)
            self.oCm = orbitVariance

        # iteration
        assert type(iteration) is int and iteration>0, 'Iteration number has to be an integer > 0'
        self.iteration = 0
        self.iterations = iteration
        self.rtol = rtol
        self.atol = atol
        self.Norms = []
        self.uNorms = []

        # Talk to me
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print('      Get some informations')

        # Get things
        self.m = self.massive.m
        self.d = self.massive.d
        self.G = self.massive.G

        # Talk to me
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print('      Initialize tables')
        self.PETSc.Sys.Print(' ')

        # Some initialization things
        self.massive.mIndex2ParamsPixels(onmyown=True)

        # Initialize mprior
        if mprior is None:
            self.mprior = self.PETSc.Vec().createMPI(self.massive.Nc, comm=self.Com)
            self.mprior.zeroEntries()
        else:
            self.mprior = mprior
        self.mprior.assemble()

        # Initialize Pre-Conditioner
        self.F0 = None
        if preconditioner is not None:
            self.F0 = preconditioner

        # X and Y arrays
        x = range(self.massive.Nx)
        y = range(self.massive.Ny)
        self.x, self.y = np.meshgrid(x,y)

        # Initialize first step
        self.refineMu = False
        self.muResidual = False
        self.muLS = True
        self.initialize(initialModel=initialModel)
        self.printState(iteration=0)

        # All done
        return
    
    def initialize(self, initialModel=None):
        '''
        Initialize the things needed.
        '''

        # Speak to me
        self.PETSc.Sys.Print('      Initialize vectors for conjugate directions')
        self.PETSc.Sys.Print(' ')

        # Initialize the lists
        self.steepestAscents = []
        self.Lambdans = []
        self.Phins = []
        self.Alphans = []
        self.Mus = []
        self.ms = []

        # Compute the first steepest ascent
        self.resConv = False
        self.computeSteepestAscent()

        # Compute the first Lambdan
        self.computeLambdan()

        # The first Phin is Lambdan
        self.phin = self.lambdan.copy()
        self.phin.assemble()
        self.Phins.append(self.phin)

        # The first Alphans is 0.
        self.alphan = 0.0
        self.Alphans.append(self.alphan)

        # Initialize mu
        self.mu = None
        self.Mus.append(self.mu)

        # Initial Model
        if initialModel is not None:
            self.massive.setmodel(initialModel, vector=self.m)

        # Save the first model
        mcopy = self.m.copy()
        mcopy.assemble()
        self.ms.append(mcopy)

        # Starting Norm
        self.Norms.append(self.computeNorm())
        self.uNorms.append(self.computeResidualNorm())

        # All done
        return

    def Solve(self, view=False, zeroOutInitial=True, writeintermediatesolution=False, 
                    plotthings=False):
        '''
        Solve the problem.
        Args:
            * view                          : If True, prints out a bunch of stuff from Solver.view()
            * zeroOutInitial                : False, uses self.m as an initial guess
            * writeintermediatesolution     : if True, writes every step in a new h5 file
        '''

        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Solving...')
        self.PETSc.Sys.Print(' ')

        # Initialize check
        self.runAgain = True

        # Set initial Guess
        if zeroOutInitial:
            self.PETSc.Sys.Print(' Initial Model has been set to zero')
            self.PETSc.Sys.Print(' ')
            self.m.zeroEntries()
        else:
            self.PETSc.Sys.Print(' Initial Model taken from what was in self.m')
            self.PETSc.Sys.Print(' ')

        # Check if write
        if writeintermediatesolution:
            self.writeState2File(iteration=self.iteration)

        # Iterations
        self.runAgain = True
        while self.runAgain:
            
            # Make an additional iteration
            self.oneIteration()
            atol, rtol = self.getARTol(self.iteration)
            
            # Check if one has to stop
            if self.iteration>=self.iterations: 
                self.runAgain = False
            if rtol is not None: 
                if rtol<self.rtol:
                    self.runAgain = False
            if atol is not None:
                if atol<self.atol:
                    self.runAgain = False

            # Print state
            self.printState(iteration=self.iteration)
            if writeintermediatesolution:
                self.writeState2File(iteration=self.iteration)

            # Plot 
            if plotthings:

                # plot something to screen
                orbits = []
                for m in self.ms:
                    orbs, ind, wor = self.massive.getOrbits(vector=m, 
                                                            target=0)
                    orbits.append(orbs)
    
                import matplotlib.pyplot as plt
                if self.Com.Get_rank()==0:
                    plt.figure()
    
                    plt.subplot(511)
                    for steepest in self.steepestAscents:
                        sa = steepest.getValues(range(100))
                        plt.semilogy(np.abs(sa), '.-')
    
                    plt.subplot(512)
                    for m in self.ms:
                        mo = m.getValues(range(100))
                        plt.plot(mo, '.-')
                    plt.plot(self.mprior.getValues(range(100)), '+-k')
    
                    plt.subplot(513)
                    for m in self.ms:
                        mo = m.getValues(range(100))
                        plt.plot(np.abs(mo - self.mprior.getValues(range(100))), '+-')
    
                    plt.subplot(514)
                    plt.plot(self.residuals.getValues(range(130)))

                    plt.subplot(515)
                    for orb in orbits:
                        plt.plot(orb, '.-')
    
                    plt.show()

        # Solved
        self.massive.solved = True

        # All done
        return

    def oneIteration(self):
        '''
        Do one iteration of the solver.
        '''

        # Calculate the residuals
        self.computeSteepestAscent()

        # Calculate Lambda_n
        self.computeLambdan()

        # Calculate Alpha_n
        self.computeAlphan()

        # Calculate Phi_n
        self.computePhin()

        # Update mu
        self.mu = self.computeMuLinearised()

        #assert self.mu>0., 'Why is mu<0?' 
        if self.refineMu:
            if self.mu<0.: self.mu *= -1.
            self.searchMu(initialGuess=self.mu)
        self.Mus.append(self.mu)

        # Update m
        self.updatem()

        # Update Norms and iteration
        self.Norms.append(self.computeNorm())
        self.uNorms.append(self.computeResidualNorm())
        self.iteration += 1

        # Activate Garbage Collector to clean up memory
        gc.collect()

        # All done
        return

    def computeNorm(self, model='m'):
        '''
        Compute the norm (d-Gm).Cd-1.(d-Gm) + (m-mprior)Cm-1(m-mprior).
        '''

        # Get m
        if type(model) is str:
            m = getattr(self, model)
        else:
            m = model

        # Compute residuals
        self.computeResiduals(model=m)

        # If residuals have been convolved
        self.cr = self.residuals.copy()
        self.cr.assemble()
        self.convolveDataSpace(vector=self.residuals, inverse=True)
        gc.collect()

        # Create a temporary vector
        self.mdiff = m.copy()
        self.mdiff.axpy(-1.0, self.mprior)
        self.mdiff.assemble()
        self.mdcopy = self.mdiff.copy()
        self.mdcopy.assemble()

        # Convolution
        self.convolveModelSpace(self.mdiff, inverse=True)
        gc.collect()

        # Norm
        mNorm = self.mdcopy.dot(self.mdiff)
        dNorm = self.cr.dot(self.residuals)
        norm = mNorm+dNorm

        # Destroy temp vector
        self.mdiff.destroy(); del self.mdiff
        self.cr.destroy(); del self.cr
        self.mdcopy.destroy(); del self.mdcopy

        # All done
        return norm, mNorm, dNorm

    def computeResidualNorm(self, model='m'):
        '''
        Compute the l2 norm of the residuals (unweighted).
        '''

        # Compute the residuals
        self.computeResiduals(factor=1., model=model)

        # Get norm
        norm = self.residuals.norm()

        # all done
        return norm

    def computeResiduals(self, factor=1., model='m'):
        '''
        Computes the residuals (d-Gm)
        '''

        # Check if vector exists
        if hasattr(self, 'residuals'):
            self.residuals.destroy()
            del self.residuals

        # Create a residual vector if needed
        self.residuals = self.PETSc.Vec().createMPI(self.massive.Nl, comm=self.Com)
        self.residuals.assemble()

        # Get m 
        if type(model) is str:
            m = getattr(self, model)
        else:
            m = model

        # Compute the prediction
        self.G.mult(m, self.residuals)

        # Compute the residuals
        self.residuals.aypx(-1., self.d)

        # Multiply
        self.residuals *= factor

        # Final
        self.residuals.assemble()

        # Not convolved
        self.resConv = False

        # All done
        return

    def computeSteepestAscent(self):
        '''
        Compute the steepest Ascent vector.
        '''

        # Create a vector if it does not exist
        if hasattr(self, 'steepestAscent'):
            self.steepestAscent.destroy()
            del self.steepestAscent
        self.steepestAscent = self.PETSc.Vec().createMPI(self.massive.Nc, comm=self.Com)
        self.steepestAscent.assemble()

        # Convolution in data space
        if not self.resConv:
            self.computeResiduals(factor=-1.)
            self.convolveDataSpace(vector=self.residuals, inverse=True)
            gc.collect()

        # Multiplication
        self.G.multTranspose(self.residuals, self.steepestAscent)
        self.steepestAscent.assemble()

        # Convolution in model space
        self.convolveModelSpace(self.steepestAscent, inverse=False)
        gc.collect()
        self.steepestAscent.assemble()

        # Add m
        self.steepestAscent.axpy(1., self.m)
        self.steepestAscent.assemble()

        # Remove mprior
        self.steepestAscent.axpy(-1., self.mprior)
        self.steepestAscent.assemble()

        # Keep it, copy and save
        sacopy = self.steepestAscent.copy()
        sacopy.assemble()
        self.steepestAscents.append(sacopy)

        # All done
        return

    def computeLambdan(self):
        '''
        Compute and save Lambdan.
        '''
        
        # Copy to create a new one of the same size
        self.lambdan = self.steepestAscents[-1].copy()
        self.lambdan.assemble()

        # Precondition
        if self.F0 is not None:
            self.lambdan.zeroEntries()
            self.F0.mult(self.steepestAscents[-1], self.lambdan)

        # Copy and save
        lacopy = self.lambdan.copy()
        lacopy.assemble()
        self.Lambdans.append(lacopy)

        # All done
        return
            
    def computeAlphan(self):
        '''
        Compute and save Alphan
        '''

        # Copy Lambdans
        lambdan = self.Lambdans[-1].copy()
        lambdanminus1 = self.Lambdans[-2].copy()
        lambdan.assemble()
        lambdanminus1.assemble()

        # Convolve these guys
        self.convolveModelSpace(lambdan, inverse=True)
        self.convolveModelSpace(lambdanminus1, inverse=True)

        # Compute values
        wn1 = self.steepestAscents[-1].dot(lambdan)
        wn2 = self.steepestAscents[-2].dot(lambdanminus1)
        wn3 = self.steepestAscents[-2].dot(lambdan)
        self.alphan = (wn1 - wn3)/wn2

        # Save it
        self.Alphans.append(self.alphan)

        # Clean up 
        lambdan.destroy(); del lambdan
        lambdanminus1.destroy(); del lambdanminus1
        gc.collect()

        # All done
        return

    def computePhin(self):
        '''
        Compute and save Phin.
        '''

        # Create a new vector of the right size
        self.phin = self.Phins[-1].copy()
        self.phin.assemble()

        # Do the multiplication
        self.phin.aypx(self.Alphans[-1], self.Lambdans[-1])
        self.phin.assemble()

        # Copy and save
        phcopy = self.phin.copy()
        phcopy.assemble()
        self.Phins.append(phcopy)

        # All done
        return

    def searchMu(self, initialGuess=1e-10):
        '''
        Search for the function that will minimize the 
        Least Square norm.
        '''

        # Function
        def createFunction(normtype, initialNorm, ownx):
            def getLSNorm(snes, x, f):

                # Get m
                m = self.m.copy()
                m.assemble()

                # Get mu
                if ownx:
                    mu = np.exp(x[0])
                else:
                    mu = None
                self.mu = self.Com.bcast(mu, root=0)

                # Update m
                m.axpy(-1.0*self.mu, self.Phins[-1])
                m.assemble()
            
                # Compute norm 
                if normtype is 'ls':
                    a = self.computeNorm(model=m)[0]
                elif normtype is 'residual':
                    a = self.computeResidualNorm(model=m)

                # Put it in f
                if ownx:
                    f[0] = a/initialNorm

                #self.PETSc.Sys.Print('Mu: {} -- {} -- {} -- {}'.\
                #format(self.mu, a, initialNorm, a/initialNorm))

                # All done
                f.assemble()

            # Save it
            self.muFunction = getLSNorm

            # All done
            return
        
        # Print stuff
        self.PETSc.Sys.Print('--------------------------------------------')
        self.PETSc.Sys.Print('--------------------------------------------')
        self.PETSc.Sys.Print('Searchin for the best Step Size...')
        self.PETSc.Sys.Print('Initial Guess: {}'.format(initialGuess))

        # Create a snes solver to search for mu
        snes = self.PETSc.SNES().create()
        snes.setType(self.PETSc.SNES.Type.NCG)

        # Create the vectors we need
        f = self.PETSc.Vec().createMPI(1, comm=self.Com)
        b = self.PETSc.Vec().createMPI(1, comm=self.Com)

        # Who owns it
        ownx = False
        if b.getOwnershipRange()[0]<=self.Com.Get_rank()<b.getOwnershipRange()[1]:
            ownx = True

        # Initialize them
        b.set(0.0)

        # Assemble
        b.assemble()

        # Search for mu that will minimize the residual norm
        if self.muResidual and not self.muLS:
            self.PETSc.Sys.Print('Searching first guess based on the Residual norm')
            initialNorm = self.computeResidualNorm()
            createFunction('residual', initialNorm, ownx)

        # Minimize then the LS norm
        elif self.muLS and not self.muResidual:
            self.PETSc.Sys.Print('Searching Step Size by optimizing the LS norm')
            initialNorm = self.computeNorm()[0]
            createFunction('ls', initialNorm, ownx)

        else:
            assert False, 'I cannot understand your choice to optimize step size'

        # Line search 
        opts = self.PETSc.Options()
        opts['snes_linesearch_type'] = 'l2'
        snes.setFromOptions()

        # Tolerances
        snes.setTolerances(1e-50, 1e-50, 100, 10000)

        # Convergence History
        snes.setConvergenceHistory()

        # Set function
        snes.setFunction(self.muFunction, f)
        
        # Initial Guess
        x = self.PETSc.Vec().createMPI(1, comm=self.Com)
        x.set(np.log(initialGuess))
        x.assemble()

        # Solve
        snes.solve(b, x)

        ## View
        #snes.view()

        # Save it
        if ownx:
            if initialNorm<f[0]:
                mu = initialGuess
            else:
                mu = np.exp(x[0])
        else:
            mu = None
        self.mu = self.Com.bcast(mu, root=0)


        # Print stuff
        self.PETSc.Sys.Print('Best Step Size found: {}'.format(self.mu))

        # All done
        return 
    
    def computeMuLinearised(self):
        '''
        Estimate Mu by linearisation of g(m) (Tarantolla, 2005, p. 232)
        '''

        # Convolve self.phin
        self.convolveModelSpace(self.phin, inverse=True)
        gc.collect()

        # Get values 
        up = self.steepestAscents[-1].dot(self.phin)
        down1 = self.Phins[-1].dot(self.phin)

        # temporary vector
        self.bn = self.d.copy()
        self.bn.assemble()
        self.G.mult(self.Phins[-1], self.bn)

        # Second temporary vector
        self.bn2 = self.bn.copy()
        self.bn2.assemble()

        # Convolve temporary vector
        self.convolveDataSpace(vector=self.bn, inverse=True)
        gc.collect()

        # Get values
        down2 = self.bn2.dot(self.bn)

        # Delete self.bn
        self.bn.destroy()
        self.bn2.destroy()
        del self.bn 
        del self.bn2

        # All done
        return up/(down1+down2)

    def updatem(self):
        '''
        Compute the in-place update of m
        '''

        # Update current m
        self.m.axpy(-1.0*self.mu, self.Phins[-1])
        self.m.assemble()

        # Copy and save
        mcopy = self.m.copy()
        mcopy.assemble()
        self.ms.append(mcopy)

        # all done
        return

    def convolveDataSpace(self, vector, inverse=False):
        '''
        Do the convolution of the residuals.
        '''

        # Get the residual interferograms
        tstart = time.ctime()
        dataSpace = self.massive.getDataSpace(vector=vector)
        tend = time.ctime()
        if self.debug:
            self.PETSc.Sys.Print('-------------------------------------')
            self.PETSc.Sys.Print('-------------------------------------')
            self.PETSc.Sys.Print('Data Convolution.....') 
            self.PETSc.Sys.Print('-------------------------------------')
            self.PETSc.Sys.Print('-------------------------------------')
            self.Com.Barrier()
            print('Worker {} gets the data between {} and {}: {}'.format(self.Com.Get_rank(), tstart, tend, [data[4] for data in dataSpace]))

        # Wait 
        self.Com.Barrier()

        # Do the convolutions
        for data in dataSpace:
            tstart = time.ctime()
            # Get Lambda and Sigma
            covariance = self.Cd[data[4]]
            # get the convolution function
            dataConv = self.dataConvolution[data[4]]
            # Do the convolution
            conv = dataConv(data[3],   # Image
                            data[0],   # X coord
                            data[1],   # Y coord
                            self.massive.dx, 
                            self.massive.dy, 
                            covariance,
                            inverse=inverse)
            # Save the results
            data[3] = conv
            # Set the results
            tend = time.ctime()
            if self.debug:
                print('Data Convolution Worker {}: Ifg {} start: {}, end: {}'.format(self.Com.Get_rank(), data[4], tstart, tend))

        # Wait 
        self.Com.Barrier()

        # Send back the data
        tstart = time.ctime()
        self.massive.setbackDataSpace(dataSpace, vector)
        tend = time.ctime()
        if self.debug:
            print('Worker {} puts back the data between {} and {}'.format(self.Com.Get_rank(), tstart, tend))

        # Clean up
        del dataSpace
        if 'conv' in locals():
            del conv

        # check
        if vector==self.residuals:
            self.resConv = True

        # All done 
        return
        
    def convolveModelSpace(self, vector, inverse=False):
        '''
        Do the convolution in the Model space.
        '''

        cm = self.massive.m.copy()
        cm.assemble()

        # Models
        tstart = time.ctime()
        Models = self.massive.getModelSpace(vector=vector, target=None)
        tend = time.ctime()
        if self.debug:
            self.PETSc.Sys.Print('-------------------------------------')
            self.PETSc.Sys.Print('-------------------------------------')
            self.PETSc.Sys.Print('Model Convolution.....') 
            self.PETSc.Sys.Print('-------------------------------------')
            self.PETSc.Sys.Print('-------------------------------------')
            self.Com.Barrier()
            print('Worker {} gets the model between {} and {}: {}'.format(self.Com.Get_rank(), tstart, tend, [model[4] for model in Models]))

        # Wait 
        self.Com.Barrier()

        # Convolve models
        for model in Models:
            tstart = time.ctime()
            # Get Lambda and Sigma
            covariance = self.Cm[model[4]]
            # Get the function
            modelConv = self.modelConvolution[model[4]]
            # Do the convolution
            conv = modelConv(model[3],
                             model[0],
                             model[1],
                             self.massive.dx, 
                             self.massive.dy, 
                             covariance,
                             inverse=inverse)
            # Save the results
            model[3] = conv
            # Set the results
            tend = time.ctime()
            if self.debug:
                print('Model Convolution Worker {}: Model {} start: {}, end: {}'.format(self.Com.Get_rank(), model[4], tstart, tend))

        # Wait 
        self.Com.Barrier()

        # Put back the model vector
        tstart = time.ctime()
        self.massive.setbackModelSpace(Models, vector=vector)
        tend = time.ctime()
        if self.debug:
            print('Worker {} puts back the model between {} and {}'.format(self.Com.Get_rank(), tstart, tend))

        # Clean up
        del Models
        if 'conv' in locals():
            del conv

        # Multiply orbits by variance
        if self.oCm is not None:
            Orbits, Indexes, Workers = self.massive.getOrbits(vector=vector, target=0)
            convOrbits = []
            for orbit,variance in zip(Orbits,self.oCm):
                if inverse:
                    convOrbits.append(orbit / variance)
                else:
                    convOrbits.append(orbit * variance)
            self.massive.setbackOrbits([convOrbits, Indexes, Workers], vector=vector)

        # All done
        return

    def writeState2File(self, iteration):
        '''
        Writes the solution to a in the h5file.
        '''
    
        # Write the model to file
        name = 'parms {:d}'.format(iteration)
        self.massive.writeModel2File(name=name)
        
        # Write the phase evolution
        name = 'recons {:d}'.format(iteration)
        self.massive.writePhaseEvolution2File(name=name, Jmat=False)

        return

    def getARTol(self, iteration):
        '''
        Returns the relative change in the Least-Sqaure norm between iteration and 
        iteration-1 (in percent).
        '''
        
        # Get it
        rtol = 1.-(self.Norms[iteration][0]/self.Norms[iteration-1][0])
        atol = (self.Norms[iteration][0]/self.Norms[0][0])

        # All done
        return atol, rtol

    def cleanup(self):
        '''
        Cleans up memory
        '''

        del self.Alphans

        for lambdan in self.Lambdans:
            lambdan.destroy()
            del lambdan
        del self.Lambdans

        for phin in self.Phins:
            phin.destroy()
            del phin
        del self.Phins

        for sa in self.steepestAscents:
            sa.destroy()
            del sa
        del self.steepestAscents

        # Call the garbage collcetor
        gc.collect()

        # All done
        return

    def printState(self, iteration=0, nel=3, minmax=True):
        '''
        Show me what the current state is...
        '''

        self.PETSc.Sys.Print('--------------------------------------------')
        self.PETSc.Sys.Print('--------------------------------------------')
        self.PETSc.Sys.Print('Iteration {} / {}: '.format(iteration, self.iterations))
        self.PETSc.Sys.Print('Iter. {}: Step Size: {}'.format(iteration, self.Mus[-1]))

        # Print norm
        try:
            self.PETSc.Sys.Print('Iter. {}: Least Squares Norm: {} ({}%% reduction)'\
                                            .format(iteration, 
                                                    self.Norms[-1][0], 
                                                    100.*(1.-(self.Norms[-1][0]/self.Norms[-2][0]))))
        except:
            self.PETSc.Sys.Print('Iter. {}: Least Squares Norm: {}'.format(iteration, self.Norms[-1][0]))

        self.PETSc.Sys.Print('Iter. {}: Model Contrib. {}'.format(iteration, self.Norms[-1][1]))
        self.PETSc.Sys.Print('Iter. {}: Data Contrib. {}'.format(iteration, self.Norms[-1][2]))

        try:
            self.PETSc.Sys.Print('Iter. {}: Unweighted Residual Norm: {} ({}%% reduction)'.\
                                 format(iteration, 
                                        self.uNorms[-1], 
                                        100.*(1. - self.uNorms[-1]/self.uNorms[-2])))
        except:
            self.PETSc.Sys.Print('Iter. {}: Unweighted Residual Norm: {}'.\
                                 format(iteration, self.uNorms[-1]))

        self.PETSc.Sys.Print('Iter. {}: Steepest Ascent: {} to {}'.format(iteration, 
                            self.steepestAscent.min()[1], 
                            self.steepestAscent.max()[1]))
        self.PETSc.Sys.Print('Iter. {}: Lambda_n: {} to {}'.format(iteration, 
                            self.lambdan.min()[1], 
                            self.lambdan.max()[1]))
        self.PETSc.Sys.Print('Iter. {}: Alpha_n: {}'.format(iteration, self.Alphans[-1]))
        self.PETSc.Sys.Print('Iter. {}: Phi_n: {} to {}'.format(iteration, 
                            self.Phins[-1].min()[1],
                            self.Phins[-1].max()[1]))

        # Minmax
        if minmax:

            # Data
            imin, min = self.d.min()
            imax, max = self.d.max()
            self.PETSc.Sys.Print('Iter. {}: Data vector from {} to {}'.format(iteration, min, max))

            # Model 
            imin, min = self.m.min()
            imax, max = self.m.max()
            self.PETSc.Sys.Print('Iter. {}: Model vector from {} to {}'.format(iteration, min, max))


        # All done
        return


# EOF
