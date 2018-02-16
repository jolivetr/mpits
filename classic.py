'''
Class that implements the TimeFn version of the massive Time Series.
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

class classic(tsmassive):

    def __init__(self, name, massiveObject=None):
        '''
        Initializes the class
        Args:
            * name          : Name of the project.
        '''

        # Just initializes the super class
        super(classic,self).__init__(name, massiveObject=massiveObject)

        # All done
        return

    def buildDesign(self):
        '''
        Builds the design matrix.
        '''

        # Store
        self.Gg = np.delete(self.Jmat, self.masterind, axis=1)

        # We should have self.Gg.shape[1] parameyters
        self.nParams = self.Gg.shape[1]

        # All done
        return

    def buildPixStartStop(self):
        '''
        Builds the list of start/end of lines for each pixel
        and the starting and ending column in G
        '''

        # Create and fill PixStartStop
        self.PixStartStop = []
        Lst = 0
        Cst = 0
        for p in self.PixList:
            Led = Lst + p[2] 
            Ced = Cst + p[3] - 1 
            self.PixStartStop.append([Lst, Led, Cst, Ced])
            Cst = Ced
            Lst = Led

        # Save
        self.PixStartStop = np.array(self.PixStartStop).astype(int)
        
        # All done
        return

    def getG(self,u):
        '''
        Returns the design matrix and the orbit matrix for a pixel u
        '''

        # Size of the Matrix
        x = self.PixList[u,0]
        y = self.PixList[u,1]
        Nl = self.PixList[u,2]
        Nc = self.PixList[u,3]

        # Get indexes to remove
        rmint = self.IfgToDelete[u]
        #rmima = self.ImagesToDelete[u]

        # Deal with the G matrix
        #G = np.delete(np.delete(self.Gg, rmint, axis=0),
        #              rmima, axis=1)
        G = np.delete(self.Gg, rmint, axis=0)

        # Create the index vectors
        iGr = range(self.PixStartStop[u,0], self.PixStartStop[u,1])
        iGc = range(self.PixStartStop[u,2], self.PixStartStop[u,3])

        # Orbit
        O = np.delete(self.Orb, rmint, axis=0)
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
    
        # Number of lines (Number of data/pixel)
        self.Nl = np.cumsum(self.PixList[:,2])[-1]

        # This is needed by the solver
        self.Ndata = self.Nifg

        # Number of columns (Number of parameters/pixel + nOrb params
        self.Nc = np.cumsum(self.PixList[:,3]-1)[-1]
        self.Npar = np.cumsum(self.PixList[:,3]-1)[-1]
        if orbit:
            self.Nc += self.nOrb*self.OrbShape + self.Nifg

        # Get some matrices
        G = self.Gg
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

        # Compute the G local matrix
        G, iGr, iGc, O, iOr, iOc = self.getG(u)

        # Get the lines
        dline = G[ni,:]
        oline = O[ni,:]                                                        

        # All done
        return dline, iGc, oline, iOc

    def writeModel2File(self, talktome=False, name='parms'):
        '''
        Once m has been solved for, this routine stores the parameters that 
        have been solved for in an hdf5 file. Pixels that have been masked 
        will be NANs.
        '''

        # How many parameters
        n = self.Jmat.shape[1]
    
        # Assert we have the good keyword
        pout = self.hdfout.create_dataset(name, shape=(self.Ny, self.Nx, n))
        pout.attrs['help'] = 'The Phase Evolution through time'
        self.Barrier()

        # Create a numpy variable
        mm = np.zeros((self.Ny, self.Nx))

        # Print stuff
        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Writing Phase to file')

        # Get the parameters
        if talktome:
            self.PETSc.Sys.Print('          Get the model parameters')
        Models = self.getModelSpace(vector='m')

        # Iterate and write
        if talktome:
            self.PETSc.Sys.Print('          Write the model parameters')
        for model in Models:
            x, y, worker, values, par = model
            if par>=self.masterind: par = par + 1
            mm[:,:] = np.nan
            mm[y,x] = values
            if self.xRef is not None:
                mm[:,:] -= mm[self.yRef, self.xRef]
            pout[:, :, par] = mm

        # Clean the screen 
        self.PETSc.Sys.Print(' ')

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
        nParams = self.nParams
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

    def getPhaseEvolution(self):
        '''
        Return the phase evolution
        '''
        
        # Get the phase from the model vector
        Phases = self.getModelSpace(vector='m')

        # Iterate over the phases to adjust the index
        for phase in Phases:
            if phase[4]>=self.masterind: phase[4]+=1

        # All done
        return Phases

    def computePhaseEvolution(self):
        '''
        There is nothing to do here.
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
        p = np.flatnonzero((self.PixStartStop[:,2]<=col) & (self.PixStartStop[:,3]>col))
        assert len(p)==1, 'Problem with col number {} in classic.col2pix'.format(col)
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
        assert len(p)==1, 'Problem with line number {} in classic.line2pix'.format(line)
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
        TimeFn = timefn.getPhaseEvolution()

        # Reference
        iRef = np.flatnonzero(np.array([timef[-1] for timef in TimeFn])==self.masterind)
        if len(iRef)>0:
            del TimeFn[iRef[0]]
        for timef in TimeFn:
            if timef[-1]>self.masterind:
                timef[-1] -= 1

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

#EOF
