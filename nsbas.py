'''
Class that implements the nsbas version of the massive Time Series.

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

# Externals
import numpy as np
import sys, gc
import h5py
import tsinsar as ts
import itertools

# Internals
from .massive import tsmassive
from . import utils

class nsbas(tsmassive):

    def __init__(self, name, massiveObject=None):
        '''
        Initializes the class
        Args:
            * name          : Name of the project.
        '''

        # Just initializes the super class
        super(nsbas,self).__init__(name, massiveObject=massiveObject)

        # Initialize the nsbas weight parameter
        self.gamma = None
        self.theta = None

        # All done
        return

    def buildDesignAndConstraint(self, rep, gamma=1e-4, theta=1.):
        '''
        Build the constrain matrix for a full pixel.
        Args:
            * rep       : Functional representation of the constrained matrix.
            * gamma     : Weight of the constrain section
            * theta     : Weight of the SBAS section
        '''

        # Functional parametrization
        self.buildTimeMatrix(rep, createh5repo=True)
        self.Cons = self.tMatrix
        self.NCons = self.Cons.shape[1]

        # Diagonal matrix
        self.LT = -1.0*np.eye(self.Nsar)

        # Design
        self.Gg = self.Jmat
        if self.masterind is not None:
            self.Gg[:,self.masterind] = 0.0

        # Store gamma
        self.gamma = gamma
        self.theta = theta
        self.nParams = self.Gg.shape[1] + self.NCons

        #All done
        return

    def buildPixStartStop(self):
        '''
        Builds the list of start/end of lines for each pixel
        and the starting and ending column in G local
        '''

        self.PixStartStop = []
        self.imagesInG = [[], [], [], []]
        Lst = 0
        Cst = 0
        for p in self.PixList:
            # Get some indexes
            Led1 = Lst + p[2] 
            Led = Led1 + p[3]
            Ced = Cst + p[3] + self.NCons
            # Append to PixStartStop
            self.PixStartStop.append([Lst, Led1, Led, Cst, Ced])
            # Get some indexes
            images = range(self.Nsar)
            x = np.ones((self.Nsar,)).astype(int)*p[0]
            y = np.ones((self.Nsar,)).astype(int)*p[1]
            lines = range(Led1, Led)
            # Append to imagesInG
            self.imagesInG[0] += x.tolist()
            self.imagesInG[1] += y.tolist()
            self.imagesInG[2] += images
            self.imagesInG[3] += lines
            # Update Starting column and line
            Cst = Ced
            Lst = Led

        # Save
        self.imagesInG = np.array(self.imagesInG).T
        self.PixStartStop = np.array(self.PixStartStop)
        
        # All done
        return

    def filld(self):
        '''
        Fill d. This method superseeds the one in massive.py
        '''

        # Run the main filld
        super(nsbas,self).filld()

        # Assemble
        self.d.assemble()

        # Multiply by theta
        self.d *= self.theta

        # All done
        return

    def fillG(self):
        '''
        Fill the G matrix with appropriate numbers. This method runs the main method in massive.py
        and updates for the pixels that haven't been dealt with correclty.
        '''

        # Run the main fillG method
        super(nsbas,self).fillG()

        # Get the ownership
        I = list(self.G.getOwnershipRange())

        # Where do we start? 
        us, xs, ys, nis = self.line2pix(I[0]) 
        # Where do we end?
        ue, xe, ye, nie = self.line2pix(I[1]-1)  

        # First pixel
        if nis>0:                    # If first pixel is incomplete, deal with its lines 
            nli = self.PixStartStop[us,2] - I[0]  
            for i in range(I[0],I[0]+nli):
                # Get the pixel position and the number of the line we want
                u, x, y, ni = self.line2pix(i) 
                # Get the corresponding line
                dline, indc, oline, indo = self.getGline(u,ni) 
                # Fill G
                self.G.setValues(i, indc, dline, self.INS)
                if self.orbit and (indo is not None):
                    self.G.setValues(i, indo, oline, self.INS) 

        # Last Pixel
        if nie<self.PixStartStop[ue,2]-1:  # If last pixel is incomplete, deal with its lines 
            for i in range(I[1]-(nie+1),I[1]):
                # Get the pixel position and the number of the line we want
                u, x, y, ni = self.line2pix(i) 
                # Get the corresponding line
                dline, indc, oline, indo = self.getGline(u,ni) 
                # Fill G
                self.G.setValues(i, indc, dline, self.INS)
                if self.orbit and (indo is not None):
                    self.G.setValues(i, indo, oline, self.INS) 

        # Make imagesInG local
        I = self.G.getOwnershipRange()
        uu = np.flatnonzero(np.logical_and(self.imagesInG[:,3]>=I[0], 
                                           self.imagesInG[:,3]<I[1]))
        self.imagesInG = self.imagesInG[uu,:]

        # All done
        return

    def getG(self,u):
        '''
        Set up the whole G matrix for one full pixel, the number u in PixList.
        returns G and the orbit matrix
        '''

        # Size of the Matrix
        x = self.PixList[u,0]
        y = self.PixList[u,1]
        Nd = self.PixList[u,2]
        Ni = self.PixList[u,3]
        Nl = Nd + Ni
        Nc = Ni + self.Cons.shape[1]

        # Get indexes to remove
        rmt = self.IfgToDelete[u]

        # Deal with the G matrix
        G = np.zeros((Nl, Nc))
        G[:Nd,:Ni] = np.delete(self.Gg,rmt,axis=0)*self.theta
        G[Nd:,:Ni] = self.LT
        G[Nd:,Ni:] = self.Cons
        G[Nd:,:] *= self.gamma

        # Create the index vectors
        iGr = range(self.PixStartStop[u,0], self.PixStartStop[u,2])
        iGc = range(self.PixStartStop[u,3], self.PixStartStop[u,4])

        # Orbit
        O = np.delete(self.Orb, rmt, axis=0)
        if self.xRef is None: 
            xRef = 0
        else:
            xRef = self.xRef
        if self.yRef is None: 
            yRef = 0
        else:
            yRef = self.yRef
        O[:,:self.Nsar] *= (np.float(x)-np.float(xRef))/np.float(self.Nx)
        O[:,self.Nsar:2*(self.Nsar)] *= (np.float(y)-np.float(yRef))/np.float(self.Ny)
        iOr = iGr[:Nd]

        # Create the index vector
        iOc = range(self.Nc-self.OrbShape*self.nOrb-self.Nifg,self.Nc)

        # All done
        return G, iGr, iGc, O, iOr, iOc

    def getFullSize(self, nonzerosfactor=10, verbose=True, orbit=True):
        '''
        Determine the full size problem dimension.
        '''
    
        # Number of lines 
        self.Nl = np.cumsum(self.PixList[:,2]) + np.cumsum(self.PixList[:,3])
        self.Nl = self.Nl[-1]

        # This is needed by the solver
        self.Ndata = self.Nifg + self.Nsar

        # Number of columns 
        self.Nc = np.cumsum(self.PixList[:,3] + self.NCons)[-1] 
        if orbit:
            self.Nc += self.nOrb*self.OrbShape + self.Nifg
        self.Npar = np.cumsum(self.PixList[:,3] + self.NCons)[-1]

        # Non zero factor
        self.nzfactor = nonzerosfactor

        # Get some matrix
        G = self.Gg
        if orbit:
            O = self.Orb

        # Number of non-zero elements on the diagonal (10* is conservative)
        self.d_nz = nonzerosfactor*np.max([np.flatnonzero(G[i,:]!=0.).shape[0]\
                for i in range(G.shape[0])]) 
        # Nmber of off-diagonal, non-zero, elements  (10* is conservative)
        if orbit:
            self.o_nz = nonzerosfactor*np.max([np.flatnonzero(O[i,:]!=0.).shape[0]\
                    for i in range(O.shape[0])])
        else:
            self.o_nz = 100

        # Orbit yes/no
        self.orbit = orbit
        # If orbit is False, still build it, so that the other routines work fine
        if not self.orbit:
            self.buildOrbitMatrix()

        # Print
        if verbose:
            self.PETSc.Sys.Print('-------------------------------------------------------')
            self.PETSc.Sys.Print('-------------------------------------------------------')
            self.PETSc.Sys.Print('System Sizes:')
            self.PETSc.Sys.Print('Main Matrix size: {} {}'.format(self.Nl, self.Nc))
            self.PETSc.Sys.Print('Non Zeros: {} {}'.format(self.d_nz, self.o_nz))
            self.PETSc.Sys.Print('Number of pixels: {}'.format(self.Npix))
            self.PETSc.Sys.Print('-------------------------------------------------------')
            self.PETSc.Sys.Print('-------------------------------------------------------')

        # All done
        return

    def getGline(self, u, ni):
        '''
        For the pixel No u in PixList, return the Design line ni and the 
        Orbit line ni.
        Args:
            * u         : Number of the pixel
            * ni        : Which line do we want
        Returns:
            * dline     : Line of the design matrix
            * indd      : Column indexes for that line
            * oline     : Line of the orbit
            * indo      : Column indexes for that line
        '''

        # Compute the G local matrix
        G, iGr, iGc, O, iOr, iOc = self.getG(u)

        # Get the line
        dline = G[ni,:]
        indd = range(self.PixStartStop[u,3],self.PixStartStop[u,4])

        # Orbit Section
        if ni<self.PixStartStop[u,1]-self.PixStartStop[u,0]:
            oline = O[ni,:]
            indo = range(self.Nc - self.OrbShape*self.nOrb - self.Nifg, self.Nc)
        else:
            oline, indo = None, None

        # All done
        return dline, indd, oline, indo

    def writeModel2File(self, talktome=False, name='parms'):
        '''
        Once m has been solved for, this routine stores the parameters included in the constraint function.
        '''
    
        # Check something
        if not hasattr(self, 'parsInG'):
            self.mIndex2ParamsPixels()

        # How many parameters
        nCons = self.Cons.shape[1]
        nPhase = self.Nsar

        # Create the variable in the h5 file
        pout = self.hdfout.create_dataset(name, shape=(self.Ny, self.Nx, nCons))
        pout.attrs['help'] = 'Model Parameters'
        self.Barrier()

        # Print stuff
        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Writing Functions to file')

        # Create a numpy variable
        mm = np.zeros((self.Ny, self.Nx))

        # Get the parameters
        if talktome:
            self.PETSc.Sys.Print('          Get the model parameters')
        Models = self.getModelSpace(vector='m')

        # Iterate and write
        if talktome:
            self.PETSc.Sys.Print('          Write the model parameters')

        for model in Models:
            x, y, worker, values, par = model
            if par>=self.Nsar:
                mm[:,:] = np.nan
                mm[y,x] = values
                if self.xRef is not None:
                    mm[:,:] -= mm[self.yRef, self.xRef]
                pout[:, :, par-self.Nsar] = mm

        # Clean the screen 
        self.PETSc.Sys.Print(' ')

        # All done
        return

    def getDataSpace(self, vector='d'):
        ''' 
        Get the data from the main PETSc vector and send them to workers.
        Eahc worker will receive a number of images so they can work on them.
        This function superseeds the main function in massive.py
        It returns the residuals of the function fit in the second part of 
        the main G matrix as well
        '''

        # Who am I
        me = self.Com.Get_rank()

        # Run the main method
        dataSpace = super(nsbas,self).getDataSpace(vector=vector)

        # Which vector do I want
        if type(vector) is str:
            dataSpaceVector = self.__getattribute__(vector)
        else:
            dataSpaceVector = vector

        # Create the list of which images goes on which worker
        imagesWanted = utils._split_seq(range(self.Nsar), self.Com.Get_size())

        # Send the image to the workers who are going to work on them
        Packages = []

        # Iterate over the workers
        for worker in range(self.Com.Get_size()):

            # Create the list of things to send
            ToSend = []

            # Iterate over the things this worker has
            for image in imagesWanted[worker]:

                # Find the lines corresponding to that image:
                ii = np.flatnonzero(self.imagesInG[:,2] == image)

                # Get the coordinates and the lines
                indx = self.imagesInG[ii,0]
                indy = self.imagesInG[ii,1]
                indo = self.imagesInG[ii,3]

                # Get the values
                Values = dataSpaceVector.getValues(indo.astype(np.int32))

                # Make a package. The self.Nifg is to differentiate this image
                # from the interferograms
                if len(Values)>0:
                    ToSend.append([indx, indy, Values, image+self.Nifg, me, indo])

            # Send the Package
            Received = self.Com.gather(ToSend, root=worker)
            # If I am the worker concerned, store it as a flat list
            if worker==me:
                Packages = list(itertools.chain.from_iterable(Received))
                del Received

        # Wait
        self.Com.Barrier()

        # When they all have been sent, collect and order

        # Which Images do I have to take care of
        Images = np.array([package[3] for package in Packages])

        # Iterate over the Images
        for image in np.unique(Images):
            # Find the good packages
            packs = np.flatnonzero(Images==image)
            # Create a holder
            data = [[] for i in range(6)]
            # Iterate over these
            for p in packs:
                x, y, val, ifg, worker, lines = Packages[p]
                data[0].append(x)
                data[1].append(y)
                data[2].append(np.ones(x.shape)*worker)
                data[3].append(val)
                data[4].append(image)
                data[5].append(lines)
            # Concatenate what's needed
            data[0] = np.concatenate(data[0]).astype(int)
            data[1] = np.concatenate(data[1]).astype(int)
            data[2] = np.concatenate(data[2]).astype(int)
            data[3] = np.concatenate(data[3]).astype(float)
            data[4] = np.unique(data[4])[0]
            data[5] = np.concatenate(data[5]).astype(int)
            # Set these in the dataSpace list
            dataSpace.append(data)

        # All done
        return dataSpace

    def getPhaseEvolution(self):
        '''
        Return the Phase evolution
        '''

        # Get the phase from the model vector
        All = self.getModelSpace(vector='m')

        # Iterate 
        Phases = []
        for phase in All:
            if phase[4]<self.Nsar:
        #        if phase[4]>=self.masterind: phase[4]+=1
                Phases.append(phase)

        # All done
        return Phases
        
    def computePhaseEvolution(self):
        '''
        Nothing to do here
        '''

        # All done
        return

    def col2pix(self, col):
        '''
        From the index of a column in the global G, returns the indexes of the 
        corresponding pixel and the corresponding parameter estimated.
        Args:
            * col               : Column of the global G.
        '''

        # Check if orbit
        if col>=self.Npar:
            return None, None

        # Find pixel
        p = np.flatnonzero((self.PixStartStop[:,3]<=col) & (self.PixStartStop[:,4]>col))
        assert len(p)==1, 'Problem with col number {} in classic.col2pix'.format(col)
        p = p[0]

        # What parameter is that?
        u = np.mod(col, self.nParams)

        # All done
        return p, u

    def line2pix(self, line, check_main_block=False, generalIfgNum=False):
        '''
        From the index of the line in the Global G, returns the 
        indexes of the corresponding pixel and the line number of the local G
        Args:
            * line              : Line of the Global G.
            * check_main_block  : Check if line is in the design matrix section or not.
            * generalIfgNum     : Returns the ifg number in the initial list.
        '''

        # Find where in PixList
        p = np.flatnonzero( (self.PixStartStop[:,0]<=line) & (self.PixStartStop[:,2]>line) )[0]

        # Set the pixel position
        x = self.PixList[p,0]
        y = self.PixList[p,1]

        # number of the interferogram, non-nan
        ni = line - self.PixStartStop[p,0]

        # Add the ifg number, not only the line in local G
        if generalIfgNum:
            # Get the list of ifg deleted
            ifgdeleted = self.IfgToDelete[p]
            # How many are under ni
            nrem = np.flatnonzero(ifgdeleted<ni)
            ni += nrem.size
            ifgdeleted = np.delete(ifgdeleted, nrem)
            # Loop until we are goo
            while nrem.size>0:
                nrem = np.flatnonzero(ifgdeleted<ni)
                ni += nrem.size
                ifgdeleted = np.delete(ifgdeleted, nrem)
            # If it has been removed return none
            if ni in self.IfgToDelete[p].tolist():
                return None, None, None, None

        # Check
        if check_main_block:
            if ni is not None:
                if ni>=self.PixList[p,2]:
                    return None, None, None, None

        # All done
        return p, x, y, ni

    def pix2line(self, x, y, ifg):
        '''
        From the position along x and y and the index of the ifg, 
        returns the line in G, if it exists, or None.
        '''

        # Which is this pixel
        u = np.flatnonzero( (self.PixList[:,0] == x) & (self.PixList[:,1] == y) )

        # check if the pixel is in the list
        if len(u) == 0:
            return None

        # Get the list of Ifg to delete
        ifgtodelete = self.IfgToDelete[u]

        # check if the ifg has to be kept or is nan
        if ifg in ifgtodelete:
            return None

        # find the position in G
        nrem = np.flatnonzero(ifgtodelete<ifg)
        add = ifg - nrem.size
        pos = np.int(self.PixStartStop[u,0] + add)

        # All done
        return pos

    def getModelSpace(self, vector='m', target=None):
        '''
        Get the model parameters as images to different workers so workers can work on them.

        Args (developper mode):
            * vector        : Which vector is going to be used (default is m).
            * target        : Send model space vectors to a special worker (default is all)
        '''

        # Check 
        if not hasattr(self, 'parsInG'):
            self.mIndex2ParamsPixels()

        # Get the vector we are working on
        if type(vector) is str:
            model = self.__getattribute__(vector)
        else:
            model = vector

        # Who am I 
        me = self.Com.Get_rank()

        # Create the list of which parameter goes on which worker
        nParams = self.NCons + self.Nsar
        if target is None:
            parWanted = utils._split_seq(range(nParams), self.Com.Get_size())
        else:
            assert type(target) is int, 'If specified, target must be an integer...'
            assert 0<=target<self.Com.Get_size(), 'If specified, target must be\
                                between 0 and {}...'.format(self.Com.Get_size())
            parWanted = [[] for i in range(self.Com.Get_size())]
            parWanted[target] = range(nParams)

        # 1. Send the models to the workers who are going to work on them
        # Iterate over the workers
        Packages = []
        for worker in range(self.Com.Get_size()):

            # Create a package to send
            ToSend = []

            # Iterate over the parameters
            for par in parWanted[worker]:

                # Find the columns of that parameter
                cols = np.flatnonzero(self.parsInG[:,2]==par)

                # Get the coordinates and lines
                indx = self.parsInG[cols,0]
                indy = self.parsInG[cols,1]
                indo = self.parsInG[cols,3].tolist()

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
            parameter[4] = np.unique(parameter[4])[0]
            # Set parameter in Parameters
            Parameters.append(parameter)

        # All done
        return Parameters

    def setbackModelSpace(self, Models, vector='m'):
        '''
        Sends the model parameters to the workers and put them back into vector.
        '''

        # Check
        if not hasattr(self, 'parsInG'):
            self.mIndex2ParamsPixels()

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
                if len(ii)>0:
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
        if type(vector) is str:
            m = self.__getattribute__(vector)
        else:
            m = vector

        # Create lists
        indi = []; values = []

        # iterate over the packages
        for package in Packages:
            model = package[3]
            for x, y, v in zip(package[0], package[1], package[2]):
                o = np.flatnonzero(np.logical_and.reduce((self.parsInG[:,0]==x,
                                                          self.parsInG[:,1]==y,
                                                          self.parsInG[:,2]==model)))
                assert len(o)>0, 'Problem broadcasting back pixel {},{} of model {}'.format(x, y, model)
                indi.append(self.parsInG[o[0],3])
                values.append(v)

        # Set values
        m.setValues(indi, values, self.INS)
        m.assemble()

        # All done
        return

    def setmodel(self, inputModel, vector='m'):
        '''
        Takes a dictionary and affect it to m.
        Args:
            * inputModel      : Dictionary
                    {0: an array the size of the decimated data,
                     2: same thing,
                     ...
                     n: same thing}
                     If index is not in the argument, it is set to 0 in the model vector.
        '''

        # Get model space
        Models = self.getModelSpace(vector=vector)

        # Iterate 
        for model in Models:
            if model[4] in inputModel.keys():
                # Get data
                ifg = inputModel[model[4]]
                # Check
                assert ifg.shape==(self.Ny, self.Nx), 'Provided array for model {} is not \n\
                        of the good shape: {} (should be {})'.format(model[4], ifg.shape, 
                                                                     (self.Ny, self.Nx))
                model[3] = ifg[model[1], model[0]]

        # Put back the model vector
        self.setbackModelSpace(Models, vector=vector)

        # All done
        return

    def setModelFromTimefn(self, timefn):
        '''
        Sets the values of the m vector from a timefn object.
        The parameterization (functions) should be the same in both
        objects. This will not be checked, so weird things can happen 
        if you do not pay attention.

        Args:
            * timefn        : Timefn instance.
        '''

        # Who am i 
        me = self.Com.Get_rank()

        # Make sure phase Evolution exists
        if not hasattr(timefn, 'Phi'):
            timefn.computePhaseEvolution()

        # Get them
        Phase = timefn.getPhaseEvolution()
        Params = timefn.getModelSpace()

        # Alter the parameter number in Params
        for param in Params:
            param[4] += self.Nsar

        # Merge both lists into one
        TimeFn = Phase+Params

        # Get the model space 
        Models = self.getModelSpace()

        # which models do I have
        iModels = [model[-1] for model in Models]
        iTimefn = [timef[-1] for timef in TimeFn]

        # Check modified
        modifieds = []

        # Iterate over the models
        for imodel in range(self.nParams):
                
            # Do I have the model
            mcheck = imodel in iModels
            tcheck = imodel in iTimefn

            # Gather that info on all workers
            mChecks = self.Com.allgather(mcheck)
            tChecks = self.Com.allgather(tcheck)
            whogott = np.flatnonzero(np.array(tChecks))
            whogotm = np.flatnonzero(np.array(mChecks))

            # Write stuff
            self.PETSc.Sys.Print('------------ Model {}'.format(imodel))

            # If I have the field, send it to all
            if tcheck:
                tosend = TimeFn[np.flatnonzero(np.array(iTimefn)==imodel)[0]]
                phase = self.Com.bcast(tosend, root=me)
            else:
                tosend = None
                phase = self.Com.bcast(tosend, root=whogott)

            # If I have the authority, modify the model
            if not mcheck:
                del phase
            else:
                # Get it
                model = Models[np.flatnonzero(np.array(iModels)==imodel)[0]]
                # Build a field with phase
                field = np.zeros((self.Ny, self.Nx))
                field[:,:] = np.nan
                field[phase[1], phase[0]] = phase[3]
                # Check that no nan is in there
                assert not np.isnan(field[model[1], model[0]]).any(), \
                    'Some NaNs are in the field... Parameter {}'.format(imodel)
                # Alter
                model[3] = field[model[1], model[0]]
                # Update modified
                modifieds.append(model[-1])
                # Delete 
                del phase

            # Clean up
            gc.collect()

            # Wait 
            self.Barrier()

        # Everything should be ok          
        modifieds = self.Com.allgather(modifieds)
        modifieds = list(itertools.chain.from_iterable(modifieds))
        self.PETSc.Sys.Print('These parameters have been altered: {}'.format(modifieds))
                
        # Set Back
        self.setbackModelSpace(Models)

        # Orbits
        Orbits, Indexes, Workers = self.getOrbits(target=0)
        newOrbits, tfin, tfwo = timefn.getOrbits(target=0)
        self.setbackOrbits([newOrbits, Indexes, Workers])

        # All done
        return

    def setModelFromClassicAndTimefn(self, classic, timefn):
        '''
        Sets the values of the m vector from a timefn object.
        The parameterization (functions) should be the same in both
        objects. This will not be checked, so weird things can happen 
        if you do not pay attention.

        Args:
            * classic       : Classic instance.
            * timefn        : Timefn instance.
        '''

        # Who am i 
        me = self.Com.Get_rank()

        # Make sure phase Evolution exists
        if not hasattr(timefn, 'Phi'):
            timefn.computePhaseEvolution()

        # Get them
        Phase = classic.getPhaseEvolution()
        Params = timefn.getModelSpace()

        # MasterInd phase
        TimeFn = timefn.getPhaseEvolution()
        itm = np.array([tf[4] for tf in TimeFn])
        if self.masterind in itm:
            Phase.append(TimeFn[np.flatnonzero(itm==self.masterind)[0]])

        # Alter the parameter number in Params
        for param in Params:
            param[4] += self.Nsar

        # Merge both lists into one
        All = Phase+Params

        # Get the model space 
        Models = self.getModelSpace()

        # which models do I have
        iModels = [model[-1] for model in Models]
        iAll = [timef[-1] for timef in All]

        # Check modified
        modifieds = []

        # Iterate over the models
        for imodel in range(self.nParams):
                
            # Do I have the model
            mcheck = imodel in iModels
            tcheck = imodel in iAll

            # Gather that info on all workers
            mChecks = self.Com.allgather(mcheck)
            tChecks = self.Com.allgather(tcheck)
            whogott = np.flatnonzero(np.array(tChecks))
            whogotm = np.flatnonzero(np.array(mChecks))

            # Write stuff
            self.PETSc.Sys.Print('------------ Model {}'.format(imodel))

            # If I have the field, send it to all
            if tcheck:
                tosend = All[np.flatnonzero(np.array(iAll)==imodel)[0]]
                phase = self.Com.bcast(tosend, root=me)
            else:
                tosend = None
                phase = self.Com.bcast(tosend, root=whogott)

            # If I have the authority, modify the model
            if not mcheck:
                del phase
            else:
                # Get it
                model = Models[np.flatnonzero(np.array(iModels)==imodel)[0]]
                # Build a field with phase
                field = np.zeros((self.Ny, self.Nx))
                field[:,:] = np.nan
                field[phase[1], phase[0]] = phase[3]
                # Check that no nan is in there
                assert not np.isnan(field[model[1], model[0]]).any(), \
                    'Some NaNs are in the field... Parameter {}'.format(imodel)
                # Alter
                model[3] = field[model[1], model[0]]
                # Update modified
                modifieds.append(model[-1])
                # Delete 
                del phase

            # Clean up
            gc.collect()

            # Wait 
            self.Barrier()

        # Everything should be ok          
        modifieds = self.Com.allgather(modifieds)
        modifieds = list(itertools.chain.from_iterable(modifieds))
        self.PETSc.Sys.Print('These parameters have been altered: {}'.format(modifieds))
                
        # Set Back 
        self.setbackModelSpace(Models)

        # Orbits
        Orbits, Indexes, Workers = self.getOrbits(target=0)
        newOrbits, tfin, tfwo = timefn.getOrbits(target=0)
        self.setbackOrbits([newOrbits, Indexes, Workers])

        # All done
        return

