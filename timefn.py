'''
Class that implements the TimeFn version of the massive Time Series.

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
import sys
import scipy.linalg as lm
import scipy.io as sio
import h5py
import datetime as dt
import tsinsar as ts
import os
import matplotlib.pyplot as plt
import itertools
import copy

# Internals
from .massive import tsmassive
from . import utils

class timefn(tsmassive):

    def __init__(self, name, massiveObject=None):
        '''
        Initializes the class
        Args:
            * name          : Name of the project.
        '''

        # Just initializes the super class
        super(timefn,self).__init__(name, massiveObject=massiveObject)

        # All done
        return

    def setGlocals(self):
        '''
        Sets the full G matrix for a full pixel.
        '''

        # This routine in this mode is a bit useless
        self.Gglocal = self.Gg

        # Orbits
        if not hasattr(self, 'Orb'):
            self.buildOrbitMatrix()

        # All done
        return

    def buildPixStartStop(self):
        '''
        Builds the list of start/end of lines for each pixel
        and the starting and ending column in Glocal
        '''

        # Since we are doing a fit, we have a constant number of parameters per pixel
        for pix in self.PixList:
            pix[3] = self.nParams

        # Create and fill PixStartStop
        self.PixStartStop = []
        Lst = 0
        Cst = 0
        for p in self.PixList:
            Led = Lst + p[2] 
            Ced = Cst + p[3] 
            self.PixStartStop.append([Lst, Led, Cst, Ced])
            Cst = Ced
            Lst = Led

        # Save
        self.PixStartStop = np.array(self.PixStartStop).astype(int)
        
        # All done
        return

    def getG(self,u):
        '''
        Set up the whole G matrix for one full pixel, the number u in PixList.
        returnes G and the orbit matrix
        '''

        # Size of the Matrix
        x = self.PixList[u,0]
        y = self.PixList[u,1]
        Nl = self.PixList[u,2]
        Nc = self.PixList[u,3]

        # Get indexes to remove
        rmt = self.IfgToDelete[u]

        # Deal with the G matrix
        G = np.delete(self.Gg,rmt,axis=0)
        assert G.shape==(Nl, Nc)

        # Create the index vectors
        iGr = range(self.PixStartStop[u,0], self.PixStartStop[u,1])
        iGc = range(self.PixStartStop[u,2], self.PixStartStop[u,3])

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
        O[:,:self.OrbShape] *= (float(x)-float(xRef))/float(self.Nx)
        O[:,self.OrbShape:2*(self.OrbShape)] *= (float(y)-float(yRef))/float(self.Ny)
        iOr = iGr

        # Create the index vector
        iOc = range(self.Nc-self.nOrb*self.OrbShape-self.Nifg,self.Nc)

        # All done
        return G, iGr, iGc, O, iOr, iOc

    def getFullSize(self, nonzerosfactor=10, verbose=True, orbit=True, minimizeorbits=None):
        '''
        Determine the full size problem dimension.
        '''
    
        # Glocals
        self.setGlocals()

        # Number of lines (Number of data/pixel)
        self.Nl = np.cumsum(self.PixList[:,2])[-1]

        # This is needed by the solver
        self.Ndata = self.Nifg

        # Number of columns (Number of parameters/pixel + nOrb params
        self.Nc = np.cumsum(self.PixList[:,3]) 
        self.Nc = self.Nc[-1]
        self.Npar = copy.deepcopy(self.Nc)
        if orbit:
            self.Nc += self.nOrb*self.OrbShape + self.Nifg

        # Get some matrices
        G = self.Gglocal
        if orbit:
            O = self.Orb

        # Non-zeros
        self.nzfactor = nonzerosfactor

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

        # Minimize Orbits?
        if minimizeorbits is not None and self.orbit:
            self.orbitMinWeight = minimizeorbits
            self.Nl += self.nOrb

        # Add constraints to the orbits
        if self.orbitConstraints is not None:
            self.Nl += len(self.orbitConstraints)*self.nOrb

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

    def getGline(self,u,ni):
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

        # get the full matrix
        G, iGr, iGc, O, iOr, iOc = self.getG(u)

        # Get the line
        dline = G[ni,:]
        indd = iGc

        # Orbits
        oline = O[ni,:]
        indo = iOc

        # All done
        return dline, indd, oline, indo

    def writeModel2File(self, talktome=False, name=None):
        '''
        Once m has been solved for, this routine stores the parameters that 
        have been solved for in an hdf5 file. Pixels that have been masked 
        will be NANs.
        '''

        # How many parameters
        n = self.tMatrix.shape[1]
    
        # Assert we have the good keyword
        if name is None:
            name = 'parms'
        pout = self.hdfout.create_dataset(name, shape=(self.Ny, self.Nx, n))
        pout.attrs['help'] = 'The parameters of the function'
        self.Barrier()

        # Create a numpy variable
        mm = np.zeros((self.Ny, self.Nx))

        # Print stuff
        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Writing Functions to file')

        # Get the parameters
        if talktome:
            self.PETSc.Sys.Print('          Get the model parameters')
        Models = self.getModelSpace(vector='m')

        # Iterate and write
        if talktome:
            self.PETSc.Sys.Print('          Write the model parameters')
        for model in Models:
            x, y, worker, values, par = model
            mm[:,:] = np.nan
            mm[y,x] = values
            if self.xRef is not None:
                mm[:,:] -= mm[self.xRef,self.yRef]
            pout[:, :, par] = mm

        # Clean the screen 
        self.PETSc.Sys.Print(' ')

        # All done
        return

    def getPhaseEvolution(self, target=None, images=None):
        '''
        Returns the phase map in self.Phi to the workers
        '''

        # Get the transfer index list
        self.Phi2ImgPix(onmyown=True)

        # Get the owner ship
        I = self.Phi.getOwnershipRange()

        # Who am I
        me = self.Com.Get_rank()

        # Which images do we want
        nSar = self.Nsar
        dates = range(nSar)
        if images is None:
            images = dates
        if type(images) is not list:
            images = [images]

        # Create the list of which date goes on which worker
        if target is None:
            dateWanted = utils._split_seq(dates, self.Com.Get_size())
        else:
            assert type(target) is int, 'Target must be an integer'
            assert 0<=target<self.Com.Get_size(), 'Target must be between 0 and {}'.format(self.Com.Get_size())
            dateWanted = [[] for i in range(self.Com.Get_size())]
            dateWanted[target] = dates

        # Create Packages and send them to the workers
        Packages = []
        for worker in range(self.Com.Get_size()):

            # Create a package to send
            ToSend = []

            # Iterate over the dates
            for date in dateWanted[worker]:

                # Find the indexes of this date
                ii = np.flatnonzero(self.imgsInPhi[:,2]==date)
                indx = self.imgsInPhi[ii,0]
                indy = self.imgsInPhi[ii,1]
                indo = self.imgsInPhi[ii,3].tolist()

                # Get the values
                Values = self.Phi.getValues(indo)

                # Make a package to send
                if len(Values)>0 and (date in images):
                    ToSend.append([indx, indy, Values, date, me])

            # Send the Packages
            Received = self.Com.gather(ToSend, root=worker)
            
            # If I am the worker concerned, store this in a flat list
            if worker==me:
                Packages = list(itertools.chain.from_iterable(Received))
                del Received

        # Wait
        self.Com.Barrier()

        # Now collect and order
        Images = np.array([package[3] for package in Packages])

        # Create a list to store what's needed
        Dates = []
        for image in np.unique(Images):
            # Find the good packages
            packs = np.flatnonzero(Images==image)
            # Create a holder for that image
            date = [[] for i in range(5)]
            # Iterate over the packages
            for p in packs:
                x, y, val, dat, worker = Packages[p]
                date[0].append(x)
                date[1].append(y)
                date[2].append(np.ones(x.shape)*worker)
                date[3].append(val)
                date[4].append(dat)
            # Concatenate
            date[0] = np.concatenate(date[0]).astype(int)
            date[1] = np.concatenate(date[1]).astype(int)
            date[2] = np.concatenate(date[2]).astype(int)
            date[3] = np.concatenate(date[3]).astype(float)
            date[4] = np.unique(date[4])[0]
            # Set in Dates
            Dates.append(date)

        # All done
        return Dates

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
                # Build arrays
                x = np.arange(self.Nx).astype(int)
                y = np.arange(self.Ny).astype(int)
                x,y = np.meshgrid(x,y)
                x = x[np.isfinite(ifg)]; y = y[np.isfinite(ifg)]
                ifgs = ifg[np.isfinite(ifg)]
                # Interpolate
                intifg, xmin, ymin = utils._linearInterp(ifgs, x, y)
                # Create a holder
                holder = np.zeros(ifg.shape)
                ii = np.flatnonzero(model[0]<intifg.shape[1])
                jj = np.flatnonzero(model[1]<intifg.shape[0])
                vv = np.intersect1d(ii,jj)
                x = model[0][vv]; y = model[1][vv]
                holder[y,x] = intifg[y-ymin, x-xmin]
                assert np.isfinite(holder).all(), 'NaNs in the holder (cdsolver.setmodel)'
                # Put it back
                model[3] = holder[model[1], model[0]]

        # Put back the model vector
        self.setbackModelSpace(Models, vector=vector)

        # All done
        return

    def getModelSpace(self, vector='m', target=None, images=None):
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

        # Which images do we want
        dates = range(self.nParams)

        # Create the list of which parameter goes on which worker
        nParams = self.nParams
        if target is None:
            parWanted = utils._split_seq(dates, self.Com.Get_size())
        else:
            assert type(target) is int, 'If specified, target must be an integer...'
            assert 0<=target<self.Com.Get_size(), 'If specified, target must be\
                                between 0 and {}...'.format(self.Com.Get_size())
            parWanted = [[] for i in range(self.Com.Get_size())]
            parWanted[target] = dates

        # 1. Send the models to the workers who are going to work on them
        # Iterate over the workers
        Packages = []
        for worker in range(self.Com.Get_size()):

            # Create a package to send
            ToSend = []

            # Iterate over the parameters
            for par in parWanted[worker]:

                if len(self.parsInG)>0:

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

                else:
                    # Send nothing
                    ToSend = []

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

    def computePhaseEvolution(self):
        '''
        Computes the evolution of the phase at each pixel for a solved problem
        '''

        # Check that the problem has been solved
        assert self.solved, 'Need to solve the problem first'

        # Gather things into a Param vector
        self.createParamVector()

        # Create the transfer matrix
        self.createParam2PhiTransfer()
        nPhi = self.Param2Phi.getSize()[0]

        # Create the phase evolution
        self.Phi = self.PETSc.Vec().createMPI(nPhi, comm=self.Com)
        self.Phi.setOption(self.ZER, 1)

        # Compute the phase values
        self.Param2Phi.mult(self.Param, self.Phi)

        # All done
        return

    def createParamVector(self):
        '''
        Gathers parameter estimates from self.m into self.Param
        '''

        # Check that the problem has been solved
        assert self.solved, 'Need to solve the problem first'
        
        # What indexes are relevant
        ist = 0
        ied = self.PixStartStop[-1][-1]
        indexes = range(ist, ied)
        indl = len(indexes)

        # Create the vector
        self.Param = self.PETSc.Vec().createMPI(indl, comm=self.Com)
        self.Param.setOption(self.ZER, 1)

        # Create the index objects
        iTo = self.PETSc.IS().createGeneral(range(indl), comm=self.Com)
        iFrom = self.PETSc.IS().createGeneral(indexes, comm=self.Com)

        # Create a Scattering object
        Scatter = self.PETSc.Scatter().create(self.m, iFrom, self.Param, iTo)

        # Scatter
        Scatter.begin(self.m, self.Param)
        Scatter.end(self.m, self.Param)

        # All done
        return

    def createParam2PhiTransfer(self):
        '''
        Creates the matrix that will allow to compute phase evolution from 
        parameters estimated.
        '''

        # Get time matrix
        tMatrix = self.tMatrix

        # How many pixels? -> How many lines/columns in the transfer matrix
        nPixel = len(self.PixList)
        nl = tMatrix.shape[0]*nPixel
        nc = tMatrix.shape[1]*nPixel

        # Allocate a giant matrix to calculate the phase evolution
        self.Param2Phi = self.PETSc.Mat().createAIJ([nl, nc], 
                comm=self.Com, nnz = [self.nzfactor*self.Nsar, self.nzfactor*self.Nsar])
        self.Param2Phi.setOption(self.ZER, 1)

        # Get ownership
        I = self.Param2Phi.getOwnershipRange()
        ipStart = np.int(np.floor(I[0]/tMatrix.shape[0]))
        ipEnd = np.int(np.floor(I[1]/tMatrix.shape[0]))

        # How many lines to fill before the end of the first pixel we deal with?
        if I[0]>0:
            ns = tMatrix.shape[0] - np.mod(I[0], tMatrix.shape[0])
        else:
            ns = 0

        # How many lines to fill after the beginning of the last pixel we deal with?
        ne = np.mod(I[1], tMatrix.shape[0])

        # Check if first pixel is full or not
        if self.Nsar>ns>0:
            # Get the values
            values = tMatrix[-ns:,:]
            # Indexes
            indr = range(I[0], I[0]+ns)
            indc = range(ipStart*tMatrix.shape[1], (ipStart+1)*tMatrix.shape[1])
            # Set values
            self.Param2Phi.setValues(indr, indc, values.flatten(), self.INS)
            # upgrade ipStart
            ipStart += 1

        # Check if the last pixel is full or not
        if self.Nsar>ne>0:
            # Get the values
            values = tMatrix[:ne,:]
            # Indexes 
            indr = range(I[1]-ne, I[1])
            indc = range(ipEnd*tMatrix.shape[1], (ipEnd+1)*tMatrix.shape[1])
            # Set values
            self.Param2Phi.setValues(indr, indc, values.flatten(), self.INS)
            # upgrade ipEnd
            #ipEnd -= 1

        # Deal with the other pixels
        for i in range(ipStart, ipEnd):
            # Indexes
            indr = range(i*tMatrix.shape[0], (i+1)*tMatrix.shape[0])
            indc = range(i*tMatrix.shape[1], (i+1)*tMatrix.shape[1])
            # Set values
            self.Param2Phi.setValues(indr, indc, tMatrix.flatten(), self.INS)

        # Matrix ready, assemble it
        self.Param2Phi.assemblyBegin()
        self.Param2Phi.assemblyEnd()

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
        p = np.flatnonzero((self.PixStartStop[:,2]<=col) & (self.PixStartStop[:,3]>col))
        assert len(p)==1, 'Problem with col number {} in timefn.col2pix'.format(col)
        p = p[0]

        # What parameter is that?
        u = np.mod(col, self.nParams)

        # All done
        return p, u

    def line2pix(self, line, generalIfgNum=False):
        '''
        From the index of the line in the Global G, returns the 
        indexes of the corresponding pixel and the line number of the local G
        Args:
            * line              : Line of the Global G.
            * generalIfgNum     : Returns the ifg number in the initial list.
        '''

        # Find where in PixList
        p = np.flatnonzero((self.PixStartStop[:,0]<=line) & (self.PixStartStop[:,1]>line))
        assert len(p)==1, 'Problem with line number {} in timefn.line2pix'.format(line)
        p = p[0]

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
        u = u[0]

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

    def phiLine2pix(self, line):
        '''
        From a line of self.Phi, returns the pixel number, its coordinates and the image number
        Args:
            * line  : Line of self.Phi
        '''

        # Which pixel are we talking about
        p = int(np.floor(line/self.tMatrix.shape[0]))

        # Coordinates
        x = self.PixList[p,0]
        y = self.PixList[p,1]

        # Image number
        ni = line - p*self.tMatrix.shape[0]

        # All done
        return p, x, y, ni

#    def reProcessOrbits(self, tol=None, stepBetweenNetworks=None):
#        '''
#        Fits the functions onto the evolution of the orbital parameters to 
#        make sure these are quite random. This is done at the PETSc level.
#        Args:
#            tol                 : [rtol, atol, diverging threshold, max iterations]
#            stepBetweenNetworks : If there is disconnected networks, this should be a list
#                                  of lists of indices of the time steps in each network.
#                  ex: [ [0, 1, 2, 3,    5,    7,    9],                     # ERS 1 Satellite
#                        [            4,    6,    8   ],                     # Envisat Satellite
#                        [                             10, 11, 12, 13, 14]   # ALOS Satellite   
#                      ]
#        '''
#
#        self.PETSc.Sys.Print('-------------------------------------------------------')
#        self.PETSc.Sys.Print(' ')
#        self.PETSc.Sys.Print(' Re-Processing Orbits')
#
#        # 1. Build a matrix to fit with a set of temporal functions
#        #    Add subnetwork offsets
#        # 1.1 Get the temporal matrix
#        iGlocal = copy.deepcopy(self.tMatrix)
#        # 1.2 If there is some subNetworks, add columns
#        if stepBetweenNetworks is not None:
#            for stepIs in stepBetweenNetworks:
#                line = np.zeros((iGlocal.shape[0], 1))
#                line[stepIs] = 1.0
#                iGlocal = np.append(iGlocal, line, axis=1)
#        # 1.3 Assemble iGlocal into the Glocal
#        nl, nc = iGlocal.shape
#        Glocal = np.zeros((nl*self.nOrb, nc*self.nOrb))
#        for i in range(self.nOrb):
#            Glocal[i*nl:(i+1)*nl,i*nc:(i+1)*nc] = iGlocal
#        # 1.4 Create the general matrix
#        Gorb = self.PETSc.Mat().createAIJ(Glocal.shape, comm=self.Com,
#                nnz=np.count_nonzero(Glocal))
#        Gorb.setOption(self.ZER, 1)
#        # 1.5 Get what is needed
#        I = Gorb.getOwnershipRange()
#        # 1.6 Fill Gorb
#        iOr = range(I[0], I[1])
#        iOc = range(nc*self.nOrb)
#        Gorb.setValues(iOr, iOc, Glocal[I[0]:I[1],:].flatten().tolist(), self.INS)
#        # 1.7 Create the parameter vector
#        morb = self.PETSc.Vec().createMPI(Glocal.shape[1], comm=self.Com)
#        # 1.8 Assemble all these people 
#        morb.assemble()
#        Gorb.assemble()
#
#        # 2. Get the subvector of the orbits
#        # 2.1 Which elements do we want on this worker
#        st,ed = Gorb.getOwnershipRange()
#        ifrom = range(self.Npar+st, self.Npar+ed)
#        iFrom = self.PETSc.IS().createGeneral(ifrom)
#        # 2.2 Get the subvector
#        dorb = self.m.getSubVector(iFrom)
#
#        # 3. Fit the orbits using a lsqr solver
#        # 3.1 Create a solver
#        OrbSolver = self.PETSc.KSP().create(comm=self.Com)
#        OrbSolver.setType('lsqr')
#        OrbSolver.pc.setType('none')
#        OrbSolver.setNormType(self.PETSc.NormType.FROBENIUS)
#        OrbSolver.setFromOptions()
#        OrbSolver.setInitialGuessNonzero(0)
#        if tol is None:
#            tol = [1e-30, 1e-30, 10000, 10000]
#        OrbSolver.setTolerances(rtol=tol[0], atol=tol[1], divtol=tol[2], max_it=tol[3])
#        OrbSolver.setOperators(Gorb)
#        # 3.2 Solve
#        OrbSolver(dorb, morb)
#        # 3.3 Infos
#        self.PETSc.Sys.Print(' ')
#        self.PETSc.Sys.Print(' Converged in %d iterations '%(OrbSolver.getIterationNumber()))
#        self.PETSc.Sys.Print(' Tolerance Asked: %e %e %d %d'%(OrbSolver.getTolerances()))
#        self.PETSc.Sys.Print(' Residual Norm: {}'.format(OrbSolver.getResidualNorm()))
#        self.PETSc.Sys.Print(' Converged Reason: %s'\
#                %(self.convergedReason(OrbSolver.getConvergedReason())))
#        self.PETSc.Sys.Print(' ')
#
#
#        # 4. Correct the self.m vector
#        # 4.1 Predict the orbits
#        orbpred = self.PETSc.Vec().createMPI(self.Nc-self.Npar, comm=self.Com)
#        Gorb.mult(morb, orbpred)
#        # 4.2 Correct the orbits 
#        dorb.axpy(-1.0, orbpred)
#        # 4.3 Send these values to the right place in self.m
#        st, ed = dorb.getOwnershipRange()
#        ito = range(st, ed)
#        iTo = self.PETSc.IS().createGeneral(ito)
#        Scat = self.PETSc.Scatter().create(dorb, iTo, self.m, iFrom)
#        Scat.begin(dorb, self.m); Scat.end(dorb, self.m); Scat.destroy()
#
#        # 5. For each parameter, compute the corresponding change per pixel
#        # 5.1 Create the General X and Y vectors
#        X = self.PETSc.Vec().createMPI(self.Npar, comm=self.Com)
#        Y = self.PETSc.Vec().createMPI(self.Npar, comm=self.Com)
#        C = self.PETSc.Vec().createMPI(self.Npar, comm=self.Com)
#        # 5.2 For each parameter, create the list of elements to fill in X and Y 
#        st, ed = X.getOwnershipRange()
#        ed = min(ed, self.Npar)
#        xvalues = []; yvalues = []; indexes = []
#        if st<self.Npar:
#            indexes = range(st,ed)
#            pst, ust = self.col2pix(st)
#            ped, ued = self.col2pix(ed-1)
#            if ust>0:
#                xvalues = [(float(self.PixList[pst][0]) - float(self.xRef))/float(self.Nx)] \
#                        * (self.nParams-ust)
#                yvalues = [(float(self.PixList[pst][1]) - float(self.yRef))/float(self.Ny)] \
#                        * (self.nParams-ust)
#                pst += 1
#            for p in range(pst,ped):
#                xvalues += [(float(self.PixList[p][0]) - float(self.xRef))/float(self.Nx)] \
#                        * self.nParams
#                yvalues += [(float(self.PixList[p][1]) - float(self.yRef))/float(self.Ny)] \
#                        * self.nParams
#            xvalues += [(float(self.PixList[ped][0]) - float(self.xRef))/float(self.Nx)] * \
#                    (ued+1)
#            yvalues += [(float(self.PixList[ped][1]) - float(self.yRef))/float(self.Ny)] * \
#                    (ued+1)
#        # 5.3 Fill X, Y and C
#        if len(indexes)>0:
#            X.setValues(indexes, xvalues, self.INS)
#            Y.setValues(indexes, yvalues, self.INS)
#            C.setValues(indexes, np.ones((ed-st,)), self.INS)
#        # 5.4 Make a list of the vectors
#        Vecs = [X, Y, C]
#        Vecs = Vecs[:self.nOrb]
#        # 5.5 Create a vector that will hold the correction
#        msub = self.PETSc.Vec().createMPI(self.Npar, comm=self.Com)
#
#        # 5.6 Re-arrange morb 
#        for vec, i in zip(Vecs, range(self.nOrb)):
#            mo = self.PETSc.Vec().createMPI(self.Npar, comm=self.Com)
#            ifrom = []; ito = []
#            ist = self.nParams 
#            if stepBetweenNetworks is not None:
#                ist += len(stepBetweenNetworks)
#            ist *= i
#            for j in range(self.nParams):
#                ifrom += [ist+j]*len(self.PixList)
#                ito += range(j, self.Npar, self.nParams)
#            iFrom = self.PETSc.IS().createGeneral(ifrom)
#            iTo = self.PETSc.IS().createGeneral(ito)
#            Scat = self.PETSc.Scatter().create(morb, iFrom, mo, iTo)
#            Scat.begin(morb, mo); Scat.end(morb, mo); Scat.destroy()
#            mox = self.PETSc.Vec().createMPI(self.Npar, comm=self.Com)  
#            mox.pointwiseMult(mo, vec)
#            msub.axpy(1.0, mox)
#            mox.destroy()
#        # 5.7 Set these values into a vector of the right size
#        mcor = self.PETSc.Vec().createMPI(self.Nc, comm=self.Com)
#        mcor.setValues(range(self.Npar, self.Nc), np.zeros((self.Nc-self.Npar,)), self.INS)
#        st, ed = msub.getOwnershipRange()
#        iFrom = self.PETSc.IS().createGeneral(range(st, ed))
#        iTo = self.PETSc.IS().createGeneral(range(st, ed))
#        Scat = self.PETSc.Scatter().create(msub, iFrom, mcor, iTo)
#        Scat.begin(msub, mcor); Scat.end(msub, mcor); Scat.destroy()
#        msub.destroy()
#        # 5.8 Add correction to self.m
#        self.m.axpy(1.0, mcor)
#        mcor.destroy()
#
#        # All done
#        return
#
#    def postProcessOrbits(self, rcond=-1, stepBetweenNetworks=None):
#        '''
#        Fits the functions onto the evolution of the orbital parameters to 
#        make sure these are quite random.
#        Args:
#            rcond               : Conditioning the fit (see man page of numpy.linalg.lstsq)
#            stepBetweenNetworks : If there is disconnected networks, this should be a list
#                                  of lists of indices of the time steps in each network.
#                  ex: [ [0, 1, 2, 3,    5,    7,    9],                     # ERS 1 Satellite
#                        [            4,    6,    8   ],                     # Envisat Satellite
#                        [                             10, 11, 12, 13, 14]   # ALOS Satellite   
#                      ]
#        '''
#
#        self.PETSc.Sys.Print('-------------------------------------------------------')
#        self.PETSc.Sys.Print(' ')
#        self.PETSc.Sys.Print(' Post-Processing Orbits')
#
#        # Only rank 0 works
#        if self.rank>0:
#            # All done
#            return
#
#        # Save the previous guys
#        self.hdfout.create_dataset('raworbits', data=self.hdfout['orbits'])
#        self.hdfout.create_dataset('rawrecons', data=self.hdfout['recons'])
#        self.hdfout.create_dataset('rawparms', data=self.hdfout['parms'])
#
#        # Get the time matrix
#        tMat = self.tMatrix
#        self.PETSc.Sys.Print(' Using the former time function representation')
#
#        # Add offsets 
#        G = copy.deepcopy(tMat)
#        nReal = G.shape[1]
#        if stepBetweenNetworks is not None:
#            for stepIs in stepBetweenNetworks:
#                line = np.zeros((G.shape[0],1))
#                line[stepIs] = 1.0
#                G = np.append(G, line, axis=1)
#
#        # Build a set of x and y coordinates
#        xx = [(p[0]-float(self.xRef))/float(self.Nx) for p in self.PixList]
#        yy = [(p[1]-float(self.yRef))/float(self.Ny) for p in self.PixList]
#
#        # Fit the X term and correct the orbit
#        ox = self.hdfout['orbits'][:,0]
#        mx, resx, rankx, sx = np.linalg.lstsq(G, ox, rcond=rcond)
#        self.hdfout['orbits'][:,0] -= np.dot(G, mx)
#
#        # Fit the Y term
#        oy = self.hdfout['orbits'][:,1]
#        my, resy, ranky, sy = np.linalg.lstsq(G, oy, rcond=rcond)
#        self.hdfout['orbits'][:,1] -= np.dot(G, my)
#
#        # Fit the C term, if needed
#        if self.nOrb>2:
#            oc = self.hdfout['orbits'][:,2]
#            mc, resc, rankc, sc = np.linalg.lstsq(G, oc, rcond=rcond)
#            self.hdfout['orbits'][:,2] -= np.dot(G, mc)
#        else:
#            mc = np.zeros(my.shape)
#
#        # Compute what to add to the parameters
#        addPars = mx[None, None, :nReal]*xx[:,:,None] + my[None, None, :nReal]*yy[:,:,None] \
#            + mc[None, None, :nReal]
#        self.hdfout['parms'][:,:,:] += addPars
#
#        # Compute what to add to the time series
#        tx = np.dot(tMat, mx[:nReal])
#        ty = np.dot(tMat, my[:nReal])
#        tc = np.dot(tMat, mc[:nReal])
#        addRecons = tx[:,None,None]*xx[None,:,:] + ty[:,None,None]*yy[None,:,:] \
#                + tc[:,None,None]
#        self.hdfout['recons'][:,:,:] += addRecons
#
#        # All done
#        return


