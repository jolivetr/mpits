'''
Class solving the problem with a simple steepest descent.

Written by R. Jolivet 2017

This has not been tested since 2017... 

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
import scipy.fftpack as fftpack
import scipy.interpolate as sciint

class sdsolver(object):

    def __init__(self, massive, dataCovariance, modelCovariance, orbitVariance, stepSize, iteration=1, mprior=None):
        '''
        Initializes the solver.
        Args:
            * massive           : Instance of massivets
            * dataCovariance    : array/list of tuples of (Lambda; Sigma) for the data covariance
            * modelCovariance   : array/list of tuples of (Lambda; Sigma) for the model covariance
            * orbitVariance     : array/list of variances for the orbit parameters
            * stepSize          : stepSize in the steepest descent 
            * iteration         : number of iterations
            * mprior            : PETSc vector of the a priori model (if None, set to 0.)

        The solver is going to iterate over the steepest descent expression given by Tarantolla 2005 (p79).
        The covariances are exponential covariances so the matrix multiplication can be done in the Fourier
        domain.
        '''

        # Get PETSc and the Communicator
        self.PETSc = massive.PETSc
        self.MPI = massive.MPI
        self.Com = massive.MPI.COMM_WORLD

        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Initialize the Steepest Descent solver')

        # Pass some things
        self.massive = massive

        # Step Size
        assert type(stepSize) is float, 'stepSize has to be float...'
        self.mu = stepSize
    
        # Talk to me
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print('      Prepare Covariances')

        # Data Covariance
        if type(dataCovariance) is tuple:
            self.Cd = [dataCovariance for i in range(self.massive.Ndata)]
        else:
            assert len(dataCovariance)==self.massive.Ndata, 'Need to provide as many \
                                                            tuples of (Lambda, Sigma) as there \
                                                            is interferograms'
            self.Cd = dataCovariance

        # Model Covaraince
        if type(modelCovariance) is tuple:
            self.Cm = [modelCovariance for i in range(self.massive.nParams)]
        else:
            assert len(modelCovariance)==self.massive.nParams, 'Need to provide as many \
                                                                tuples of (Lambda, Sigma) as \
                                                                there is functional parameters'
            self.Cm = modelCovariance

        # Orbit Variance
        if type(orbitVariance) is float:
            self.oCm = [orbitVariance for i in range(self.massive.nOrb*self.massive.Nsar)]
        else:
            assert len(orbitVariance)==self.massive.nOrb*self.massive.Nsar, 'Need to provide as many \
                                                                             variance values as there is \
                                                                             orbital parameters ({})'.format(\
                                                                             self.massive.nOrb*self.massive.Nsar)
            self.oCm = orbitVariance

        # iteration
        assert type(iteration) is int and iteration>0, 'Iteration number has to be an integer > 0'
        self.iteration = 0
        self.iterations = iteration
        self.Norms = []

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
        self.massive.Glines2IfgPixels(onmyown=True)
        self.massive.mIndex2ParamsPixels(onmyown=True)
        self.pixels = self.massive.ifgsInG
        self.params = self.massive.parsInG

        # Initialize mprior
        if mprior is None:
            self.mprior = self.PETSc.Vec().createMPI(self.massive.Nc, comm=self.Com)
            self.mprior.zeroEntries()
        else:
            self.mprior = mprior
        self.mprior.assemble()

        # X and Y arrays
        x = range(self.massive.Nx)
        y = range(self.massive.Ny)
        self.x, self.y = np.meshgrid(x,y)

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

        # Starting Norm
        self.Norms.append(self.computeNorm())
        self.PETSc.Sys.Print('          Unweighted Residual Norm: {}'.format(self.Norms[-1]))

        # Iterations
        while self.runAgain:
            self.oneIteration()
            if self.iteration<self.iterations:
                self.runAgain = True

        # Solved
        self.massive.solved = True

        # All done
        return

    def oneIteration(self):
        '''
        Do one iteration of the solver.
        '''

        # Talk To Me
        self.PETSc.Sys.Print('      Iteration {} / {}'.format(self.iteration,self.iterations))

        # Calculate the residuals
        self.PETSc.Sys.Print('          Calculate Residuals')
        self.computeResiduals(factor=-1.)

        # Do the convolution of the residuals
        self.PETSc.Sys.Print('          Convolve Residuals')
        self.convolveResiduals()

        # Compute steepestAscent
        self.PETSc.Sys.Print('          Go to the model space')
        self.createsteepestAscent()

        # Do the convolution of the models
        self.PETSc.Sys.Print('          Convolve in the model space')
        self.convolveModels()

        # Update m
        self.PETSc.Sys.Print('          Update model vector')
        self.updatem()

        # Update Norms and iteration
        self.Norms.append(self.computeNorm())
        self.PETSc.Sys.Print('          Unweighted Residual Norm: {}'.format(self.Norms[-1]))
        self.iteration += 1

        # All done
        return

    def computeNorm(self):
        '''
        Compute the l2 norm of the residuals (unweighted).
        '''

        # Compute the residuals
        self.computeResiduals(factor=1.)

        # Get norm
        norm = self.residuals.norm()

        # all done
        return norm

    def computeResiduals(self, factor=1.):
        '''
        Computes the residuals (d-Gm)
        '''

        # Create a residual vector if needed
        if not hasattr(self, 'residuals'):
            self.residuals = self.PETSc.Vec().createMPI(self.massive.Nl, comm=self.Com)

        # Assemble 
        self.residuals.assemble()

        # Compute the prediction
        self.G.mult(self.m, self.residuals)

        # Compute the residuals
        self.residuals.axpy(-1., self.d)

        # Multiply
        self.residuals *= factor

        # All done
        return

    def createsteepestAscent(self):
        '''
        Compute the dot product (G.T residuals)
        '''

        # Create a vector if it does not exist
        if not hasattr(self, 'Gtr'):
            self.steepestAscent = self.PETSc.Vec().createMPI(self.massive.Nc, comm=self.Com)

        # Assemble 
        self.steepestAscent.assemble()

        # Do the muliplication
        self.G.multTranspose(self.residuals, self.steepestAscent)

        # All done
        return

    def finalizesteepestAscent(self):
        
        # Add m to steepestAscent
        self.steepestAscent.axpy(1., self.m)
        self.steepestAscent.assemble()

        # Remove mprior to steepestAscent
        self.steepestAscent.axpy(-1., self.mprior)
        self.steepestAscent.assemble()

        # All done
        return 

    def updatem(self):
        '''
        Compute the in-place update of m
        '''

        # finalize steepest ascent
        self.finalizesteepestAscent()

        # multiply by mu
        self.steepestAscent *= self.mu

        # update m
        self.m.axpy(-1.0, self.steepestAscent)

        # all done
        return

    def convolveResiduals(self):
        '''
        Do the convolution of the residuals.
        '''

        # Get the residual interferograms
        Residuals = self.getResiduals()

        # Do the convolutions
        for residual in Residuals:
            # Get Lambda and Sigma
            covariance = self.Cd[residual[4]]
            # Do the convolution
            conv = self.expConvolution(residual[3],   # Image
                                       residual[0],   # X coord
                                       residual[1],   # Y coord
                                       self.massive.dx, 
                                       self.massive.dy,
                                       covariance,
                                       inverse=True)
            # Set the results
            residual[3] = conv

        # Send back the data
        self.setbackResiduals(Residuals)

        # All done 
        return

    def getResiduals(self):
        '''
        Get the residuals from the self.residuals PETSc vector and send them to workers.
        Each worker will receive a number of residual image so they can work on them.
        '''

        # Who am I
        me = self.Com.Get_rank()

        # Create the list of which ifg goes on which worker
        Nifg = self.massive.Nifg
        ifgsWanted = self._split_seq(range(Nifg), self.Com.Get_size())

        # 1. Send the residuals to the workers who are going to work on them
        Packages = []   # In case nothing is sent here
        # Iterate over the workers
        for worker in range(self.Com.Get_size()):

            # Create the list of things to send
            ToSend = []

            # Iterate over the ifgs this worker takes care of 
            for ifg in ifgsWanted[worker]:
                
                # Find the lines corresponding to that interfero
                ii = np.flatnonzero(self.pixels[:,2] == ifg)

                # Get the coordinates and lines
                indx = self.pixels[ii,0]        # X coordinates of the pixels
                indy = self.pixels[ii,1]        # Y coordinates of the pixels
                indo = self.pixels[ii,3]        # Which lines are they in residuals

                # Get the values
                Values = self.residuals.getValues(indo.astype(np.int32))

                # Make a package to send (x, y, data, ifg, worker-wher-it-comes-from)
                if len(Values)>0:
                    ToSend.append([indx, indy, Values, ifg, me])

            # Send the package
            Received = self.Com.gather(ToSend, root=worker)
            # If I am the worker concerned, store it as a flat list
            if worker==me: 
                Packages = list(itertools.chain.from_iterable(Received))
                del Received

        # Wait (doesn't cost much and make sure things go accordingly)
        self.Com.Barrier()

        # 2. When they have all been sent, collect and order as interferograms

        # Which ifgs do I have to take care of
        Ifgs = np.array([package[3] for package in Packages])
    
        # Create a list to store the thing
        Residuals = []
        for ifg in np.unique(Ifgs):
            # Find the good packages
            packs = np.flatnonzero(Ifgs==ifg)
            # Create a holder
            residual = [[] for i in range(5)]
            # Iterate over these packages
            for p in packs:
                x, y, val, ifg, worker = Packages[p]
                residual[0].append(x)
                residual[1].append(y)
                residual[2].append(np.ones(x.shape)*worker)
                residual[3].append(val)
                residual[4].append(ifg)
            # Concatenate what's needed
            residual[0] = np.concatenate(residual[0]).astype(int)
            residual[1] = np.concatenate(residual[1]).astype(int)
            residual[2] = np.concatenate(residual[2]).astype(int)
            residual[3] = np.concatenate(residual[3]).astype(float)
            residual[4] = np.unique(residual[4])[0]
            # Set residual in Residuals
            Residuals.append(residual)

        # All done
        return Residuals
        
    def setbackResiduals(self, Residuals):
        '''
        Sends the Residuals by package to the workers and put them back in self.residuals
        '''

        # Who am I
        me = self.Com.Get_rank()

        # 1. Iterate over the residuals and send to workers
        Packages = []   # In case nothing is sent here
        for worker in range(self.Com.Get_size()):
            # Create the package to send
            ToSend = []
            # Iterate over the residuals
            for residual in Residuals:
                ii = np.flatnonzero(residual[2]==worker)
                if len(ii)>0:
                    x = residual[0][ii]
                    y = residual[1][ii]
                    v = residual[3][ii]
                    i = residual[4]
                    ToSend.append([x, y, v, i])
            # Send the thing
            Received = self.Com.gather(ToSend, root=worker)
            # If I am the worker concerned by this package, store it
            if worker==me: 
                Packages = list(itertools.chain.from_iterable(Received))
                del Received

        # Wait (doesn't cost much and make sure things go accordingly)
        self.Com.Barrier()

        # 2. Take things and put them back in residuals
        indo = []; values = []
        for package in Packages:
            ifg = package[3]
            for x, y, v in zip(package[0], package[1], package[2]):
                o = np.flatnonzero(np.logical_and.reduce((self.pixels[:,0]==x,
                                                          self.pixels[:,1]==y,
                                                          self.pixels[:,2]==ifg)))
                assert len(o)>0, 'Problem broadcasting back pixel {},{} of interferogram {}'.format(x, y, ifg)
                indo.append(o[0])
                values.append(v)
        
        # 3. Set values in residuals
        self.residuals.setValues(indo, values, self.massive.INS)

        # All done
        return

    def convolveModels(self):
        '''
        Do the convolution in the Model space.
        '''

        # Models
        Models = self.getModels()
        Orbits, Indexes, Workers = self.getOrbits()

        # Convolve models
        for model in Models:
            # Get Lambda and Sigma
            covariance = self.Cm[model[4]]
            # Do the convolution
            conv = self.expConvolution(model[3],
                                       model[0],
                                       model[1],
                                       self.massive.dx,
                                       self.massive.dy,
                                       covariance,
                                       inverse=False)
            # Set the results
            model[3] = conv

        # Multiply orbits by variance
        for orbit,variance in zip(Orbits,self.oCm):
            orbit *= variance

        # Put back the model vector
        self.setbackModels(Models)
        self.setbackOrbits([Orbits, Indexes, Workers])

        # All done
        return

    def getModels(self, vector='steepestAscent'):
        '''
        Get the model parameters as images to different workers so workers can work on them.

        Args (developper mode):
            * vector        : Which vector is going to be used (default is steepestAscent).
        '''

        # Get the vector we are working on
        model = self.__getattribute__(vector)

        # Who am I 
        me = self.Com.Get_rank()

        # Create the list of which parameter goes on which worker
        nParams = self.massive.nParams
        parWanted = self._split_seq(range(nParams), self.Com.Get_size())

        # 1. Send the models to the workers who are going to work on them
        # Iterate over the workers
        Packages = []
        for worker in range(self.Com.Get_size()):

            # Create a package to send
            ToSend = []

            # Iterate over the parameters
            for par in parWanted[worker]:

                # Find the columns of that parameter
                cols = np.flatnonzero(self.params[:,2]==par)

                # Get the coordinates and lines
                indx = self.params[cols,0]
                indy = self.params[cols,1]
                indo = self.params[cols,3].tolist()

                # Get the values
                Values = model.getValues(indo)

                # Make a package to send
                if len(Values)>0:
                    ToSend.append([indx, indy, Values, par, me])

            # Send the packages
            Received = self.Com.gather(ToSend, root=worker)
            # If I am the worker concerned, store it as a flat list
            if worker==me: 
                Packages = list(itertools.chain.from_iterable(Received))
                del Received

        # Wait (doesn't cost much and make sure things go accordingly)
        self.Com.Barrier()

        # 2. When all have been sent, collect and order 
        
        # Which parameters do I have to take care of
        Pars = np.array([package[3] for package in Packages])

        # Create a list to store things
        Parameters = []
        for par in np.unique(Pars):
            # Find the packages with this parameter
            packs = np.flatnonzero(Pars==par)
            # Create a holder for that parameter
            parameter = [[] for i in range(5)]
            # Iterate over these packages
            for p in packs:
                x, y, val, Par, worker = Packages[p]
                parameter[0].append(x)
                parameter[1].append(y)
                parameter[2].append(np.ones(x.shape)*worker)
                parameter[3].append(val)
                parameter[4].append(Par)
            # Concatenate what needs to be concatenated
            parameter[0] = np.concatenate(parameter[0]).astype(int)
            parameter[1] = np.concatenate(parameter[1]).astype(int)
            parameter[2] = np.concatenate(parameter[2]).astype(int)
            parameter[3] = np.concatenate(parameter[3]).astype(float)
            parameter[4] = np.unique(parameter[4])
            # Set parameter in Parameters
            Parameters.append(parameter)

        # All done
        return Parameters

    def getOrbits(self, worker=0, vector='steepestAscent'):
        '''
        Returns all the orbital parameters onto a worker.
        Returns also the where it came from and the index of the orbital parameter.
        
        Args:
            * worker    : Rank of the worker that is going to receive the orbits.
            * vector    : Which vector is holding the orbit parameters (default: steepestAscent).
        '''

        # Get the appropriate vector
        m = self.__getattribute__(vector)

        # Get the index of the orbits
        Os = self.massive.OrbShape
        nOrb = self.massive.nOrb
        Nc = self.massive.Nc
        indexes = np.array(range(Nc - nOrb*Os, Nc))

        # What do I own
        I = self.m.getOwnershipRange()

        # What can I get from here
        indexes = indexes[np.logical_and(indexes>=I[0], indexes<I[1])].tolist()

        # Get the values
        orbits = m.getValues(indexes)

        # Gather these onto node worker
        recOrb = self.Com.gather(orbits, root=worker)
        recInd = self.Com.gather(indexes, root=worker)
        recWor = self.Com.gather((np.ones(orbits.shape)*self.Com.Get_rank()).astype(int), root=worker)

        # If I am worker 'worker', flatten Orbit and indexes and return
        if self.Com.Get_rank()==worker:
            Orbits = list(itertools.chain.from_iterable(recOrb))
            Indexes = list(itertools.chain.from_iterable(recInd))
            Workers = list(itertools.chain.from_iterable(recWor))
            return Orbits, Indexes, Workers
        else:
            return [], [], []

        # All done

    def setbackModels(self, Models, vector='steepestAscent'):
        '''
        Sends the model parameters to the workers and put them back into vector.
        '''

        # Who am I
        me = self.Com.Get_rank()

        # 1. Iterate over the workers and send them what they want
        Packages = [] # In case nothing is sent here
        for worker in range(self.Com.Get_size()):
            # Create the package to send
            ToSend = []
            # Iterate over the models
            for model in Models:
                ii = np.flatnonzero(model[2]==worker)
                x = model[0][ii]
                y = model[1][ii]
                v = model[3][ii]
                p = model[4]
                if len(p)>0:
                    ToSend.append([x, y, v, p])
            # Send this
            Received = self.Com.gather(ToSend, root=worker)
            # If I am the worker concerned by this package, store it
            if worker==me: 
                Packages = list(itertools.chain.from_iterable(Received))
                del Received
        
        # Wait (doesn't cost much and make sure things go accordingly)
        self.Com.Barrier()

        # 2. Take things and put them back in the model vector

        # Which model vector do we work on?
        m = self.__getattribute__(vector)

        # Create lists
        indi = []; values = []

        # iterate over the packages
        for package in Packages:
            model = package[3]
            for x, y, v in zip(package[0], package[1], package[2]):
                o = np.flatnonzero(np.logical_and.reduce((self.params[:,0]==x,
                                                          self.params[:,1]==y,
                                                          self.params[:,2]==model)))
                assert len(o)>0, 'Problem broadcasting back pixel {},{} of model {}'.format(x, y, model)
                indi.append(o)
                values.append(v)

        # Set values
        m.setValues(indi, values, self.massive.INS)

        # All done
        return

    def setbackOrbits(self, orblist, vector='steepestAscent'):
        '''
        Sends the orbit parameters to the workers and put them back in the vector.
        '''

        # Which model vector do we work on 
        m = self.__getattribute__(vector)

        # Who am I 
        me = self.Com.Get_rank()

        # Get what needs to be sent
        Orbits, Indexes, Workers = orblist

        # Iterate over the workers
        for worker in range(self.Com.Get_size()):
            # Find the stuff to send
            ii = np.flatnonzero(np.array(Workers)==worker)
            # If there is something
            i = [Indexes[u] for u in ii]
            v = [Orbits[u] for u in ii]
            # Create the package to send
            ToSend = [v,i]
            # Send this 
            Received = self.Com.gather(ToSend, root=worker)
            # If I am the worker concerned by this package, store it in m
            if worker==me and Received is not None:
                val = []
                ind = []
                for received in Received:
                    val += received[0]
                    ind += received[1]
                m.setValues(ind, val, self.massive.INS)

        # All done
        return

    def expConvolution(self, image, xm, ym, Lambda, Sigma, inverse=False):
        '''
        Convolve an image of the size of the interferogram with an exponential (or inverse exponential)
        function.
        The function is of the form:
            f(x1,x2) = Sigma^2 exp(-||x1,x2||/Lambda)
            where ||x1,x2|| is the distance between pixels x1 and x2.
        Returns the convoluted function.
        
        Args:
            * image     : 1d array containing the data
            * xm        : 1d array the size of image containing the x coordinates
            * ym        : 1d array the size of the 
            * Lambda    : Float, Correlation length
            * Sigma     : Float, Amplitude of the correlation function  
        '''

        # Get X and Y
        dx, dy = self.massive.dx, self.massive.dy
        x, y = self.x, self.y

        # Interpolate
        inter = sciint.Rbf(xm, ym, image, method='multiquadratic')        

        # Do the FFT
        fm = fftpack.fft2(inter(x.astype(float),y.astype(float)))
        u = fftpack.fftfreq(x.shape[1], d=dx)
        v = fftpack.fftfreq(y.shape[0], d=dy)
        u,v = np.meshgrid(u,v)

        # Select the convolution function
        if inverse:
            H = self._expInvF
        else:
            H = self._expF

        # Convolve with the function
        dfm = H(u,v,Lambda,Sigma)*fm
        dm = np.real(fftpack.ifft2(dfm))/np.sqrt(2.)
    
        # all done
        return dm[ym, xm]

    def _expInvF(self, u, v, lam, sig):
        return ((1 + (lam*u*2*np.pi)**2 + (lam*v*2*np.pi)**2)**(1.5)) \
                 /(sig*sig*lam*lam*2*np.pi)

    def _expF(self, u, v, lam, sig):
        return (sig*sig*lam*lam*2*np.pi)/ \
                ((1 + (lam*u*2*np.pi)**2 + (lam*v*2*np.pi)**2)**(1.5))

    def _split_seq(self, seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq

# EOF
