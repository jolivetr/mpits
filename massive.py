'''
Class that deals with a massive TS inversion (all pixels at the same time).

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
import itertools
import matplotlib.pyplot as plt

from . import utils

class tsmassive(object):

    def __init__(self, name, massiveObject=None):
        '''
        Initializes the class
        Args:
            * name          : Name of the project.
        '''

        # Stores the name somewhere
        self.name = name

        # Store if the problem has been solved
        self.solved = False

        # Initialize stuffs
        self.orbitMinWeight = None

        if massiveObject is None:
            
            # Import a bunch of stuff to initialize the mpi communicator
            import mpi4py
            from mpi4py import MPI

            # Stores the communicator
            self.MPI = MPI
            self.Com = MPI.COMM_WORLD

            # Initialize the rank
            self.rank = self.Com.Get_rank()

            # Import a bunch of stuff to initialize the petsc business
            import petsc4py
            petsc4py.init(sys.argv,comm=self.Com)
            from petsc4py import PETSc

            # Store PETSc into self
            self.PETSc = PETSc

        else:
            self.PETSc = massiveObject.PETSc
            self.MPI = massiveObject.MPI
            self.Com = massiveObject.Com
            self.rank = massiveObject.Com.Get_rank()

        # Ignore zero entries in mat() and vec()
        self.ZER = self.PETSc.Mat.Option.IGNORE_ZERO_ENTRIES
        # Insert, rather than Add
        self.INS = self.PETSc.InsertMode.INSERT_VALUES
        # Add, rather than insert
        self.ADD = self.PETSc.InsertMode.ADD_VALUES                       
        # PETSc options
        self.Opt = self.PETSc.Options()                                   

        # Print stuff to show it has been initialized
        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print('         Massive Time Series Inversion System          ')
        self.PETSc.Sys.Print(' ') 
        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print('-------------------------------------------------------')

        # Set masterindex to 0
        self.masterind = 0

        # All done
        return

    def Finalize(self):
        '''
        Finalize the MPI and PETSc dudes.
        '''

        # Finalize PETSc
        self.PETSc._finalize()

        # Finalize MPI
        self.MPI.Finalize()

        # All done
        return

    def Barrier(self):
        ''' 
        Implements a MPI Barrier.
        '''

        # Barrier
        self.Com.Barrier()

        # All done
        return

    def setMasterInd(self, masterind):
        '''
        Set the index of the master date.
        Args:
            * masterind     : Index of the master date.
        '''

        # Set it
        self.masterind = masterind

        # Move the time vector
        if hasattr(self, 'ti'):
            self.masterTime = copy.deepcopy(self.ti[masterind])
            self.ti -= self.ti[masterind]

        # All done
        return

    def Limits(self, stay, endy, stax, endx, stey=1, stex=1, dx=1., dy=1.):
        '''
        Sets the limits for the domain to compute on the interferogram.
        Args:
            * stay          : First line to include.
            * endy          : Last line to include.
            * stax          : First column to include.
            * endx          : Last column to include.
            * dx            : Size of the pixels along range (same unit as Lambda)
            * dy            : Size of the pixels along azimuth (same unit as Lambda)
        '''

        self.stay = stay
        self.endy = endy
        self.stax = stax
        self.endx = endx
        self.stey = stey
        self.stex = stex
        self.dx = dx*stex
        self.dy = dy*stey

        # All done
        return

    def HDF5open(self, inputname, outputname):
        '''
        Opens the input HDF5 file to read the data.
        Args:
            * inputname : Name of the hdf5 file to open.
            * outputname: Name of the output hdf5 file.
        '''

        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Open input (%s) and output (%s) HDF5 files '%(inputname, outputname))
        self.PETSc.Sys.Print(' ')

        # Open the file
        self.hdfin = h5py.File(inputname, 'r', driver='mpio', comm=self.Com) 
        self.hdfout = h5py.File(outputname, 'w', driver='mpio', comm=self.Com)

        # All done
        return

    def buildTimeMatrix(self, representation, createh5repo=True):
        '''
        Build the constrain matrix for a full pixel.
        Args:
            * representation: Functional representation of the constrained matrix.
            * createh5repo  : Create the dataset in the h5 file for the parameters.
        '''
        
        # Get the time
        time = self.ti
        dates = self.da

        # Transform the dates into time 
        referenceTime = dt.datetime.fromordinal(int(self.da[self.masterind]))
        for f, func in enumerate(representation):
            for a, args in enumerate(func):
                for s, spec in enumerate(args):
                    if type(spec)==dt.datetime:
                        spec = (spec - referenceTime).days/365.25
                        representation[f][a][s] = spec

        # Functional parametrization
        self.rep = representation
        tMatrix,mName,regF = ts.Timefn(representation,time)
        Jmat = copy.deepcopy(self.Jmat)
        if hasattr(self, 'masterind'):
            Jmat[:,self.masterind] = 0.0
        Gg = np.dot(Jmat, tMatrix)

        # Size
        self.nParams = tMatrix.shape[1]

        # Store 
        self.tMatrix = tMatrix
        self.Gg = Gg
        self.mName = mName

        # Store
        if createh5repo:
            self.hdfout.create_dataset('mName', data=self.mName)

        #All done
        return

    def HDF5close(self):
        '''
        Close the 2 HDF5 files open.
        '''

        self.hdfin.close()
        self.hdfout.close()

        # All done
        return

    def LoadDataFromHDF5(self, referencePixel=None, dataStorage='igram'):
        '''
        Loads Stuffs from the Design Matrix
        '''

        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Get Design Matrix and Data From the h5file')
        self.PETSc.Sys.Print(' ')

        # Get the files
        fin = self.hdfin
        fout = self.hdfout

        # Get the design matrix
        self.Jmat = fin['Jmat'].value

        # Get the numbers 
        self.Nifg = self.Jmat.shape[0]
        self.Nsar = self.Jmat.shape[1]

        # Get the dates
        self.da = fin['dates'].value
        self.ti = fin['tims'].value 
        fout.create_dataset('dates',data=self.da)
        fout.create_dataset('tims',data=self.ti)
        if hasattr(self, 'masterind'):
            self.masterTime = copy.deepcopy(self.ti[self.masterind])
            self.ti -= fin['tims'][self.masterind]
            fout.create_dataset('masterind',data=self.masterind)
        else:
            self.masterTime = None

        # Create sub data holder
        igram = fin[dataStorage]
        
        # Check something
        assert igram.shape[1]>=self.endy, \
                'More lines than available asked... {} / {}'.format(igram.shape[1], 
                                                                    self.endy)
        assert igram.shape[2]>=self.endx, \
                'More columns than available asked... {} / {}'.format(igram.shape[2],
                                                                      self.endx)

        # Some size things
        lines = range(self.stay, self.endy, self.stey)
        self.Ny = len(lines)
        self.Nx = len(range(self.stax, self.endx, self.stex))
        self.igramsub = fout.create_dataset('Data',(self.Nifg,self.Ny,self.Nx),'f')

        # Read and copy sub data 
        isublines = utils._split_seq(lines, self.Com.Get_size())[self.Com.Get_rank()]
        osublines = utils._split_seq(range(len(lines)), self.Com.Get_size())[self.Com.Get_rank()]
        self.igramsub[:,osublines,:] = igram[:,isublines,self.stax:self.endx:self.stex]

        # Wait until all has been copied
        self.Barrier()

        # Reference (only on a fraction of interfero)
        if referencePixel is not None:
            self.xRef, self.yRef = referencePixel
            for i in utils._split_seq(range(self.igramsub.shape[0]), self.Com.Get_size())[self.Com.Get_rank()]:
                assert np.isfinite(self.igramsub[i, self.yRef, self.xRef]), 'Reference is not finite on interferogram {}'.format(i)
                self.igramsub[i,:,:] -= self.igramsub[i, self.yRef, self.xRef]
        else:
            self.xRef = None
            self.yRef = None
 
        # build arrays with that
        self.da = np.array(self.da)

        # All done
        return

    def makeMask(self, MinInt=None, MinIma=None, debug=False, minPix=10):
        '''
        Builds a list of x and y coordinates of the valid pixels, 
        together with the number of interferograms and images valid per pixel. 
        Args:
            * MinInt        : Minimum number of coherent Interferograms to accept pixel.
            * MinIma        : Minimum number of coherent Images to accept pixel.
            * debug         : If True, shows the minimum number of ifg and images requested
                              and the number ifg and image available per pixel
        '''

        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Determine the mask and the number of data')
        self.PETSc.Sys.Print(' ')

        # Limits
        if MinInt is None:
            Ndmin = self.Nifg
        else:
            Ndmin = MinInt
        if MinIma is None:
            Nimin = self.Nsar
        else:
            Nimin = MinIma

        # Create a storage in the h5 file
        images = self.hdfout.create_dataset('Number of Images', (self.Ny, self.Nx), dtype='f')
        interferograms = self.hdfout.create_dataset('Number of Interferograms', (self.Ny, self.Nx), dtype='f')

        # Allocate a list
        PixList = []
        IfgToDelete = []
        ImagesToDelete = [] 
   
        # Get the lines we are going to work on
        me = self.Com.Get_rank()
        size = self.Com.Get_size()
        columns = utils._split_seq(range(self.Nx), size)[me]

        # Loop on the pixels
        for c in columns:
            for l in range(self.Ny):

                # Get the data
                d = self.igramsub[:, l, c]

                # Get the number of non-nan pixels (i.e. interferograms)
                Nd = len(d[np.isfinite(d)])

                # Check Number of interferos
                J = np.delete(self.Jmat, np.where(np.isnan(d)), axis=0)
                a = np.array([np.count_nonzero(J[:,i]) for i in range(self.Nsar)])

                # Check Masterind things
                if hasattr(self, 'masterind'):
                    if a[self.masterind] > 0:
                        a = np.delete(a, self.masterind)

                # Number of parameters
                Ni = self.Nsar-1 
                if hasattr(self, 'masterind'): Ni += 1
                
                # Get the number of images
                Nit = np.count_nonzero(a)
                if hasattr(self, 'masterind'): Nit += 1

                # Store that
                images[l, c] = Nit
                interferograms[l,c] = Nd

                # Store these
                if debug:
                    print('Col/Line: {}/{}   Ndata, min: {}/{}   Ni, min: {}/{}'.format(c,l,
                                                                Nd, Ndmin, Ni, Nimin))
                if (Nd>=Ndmin) and (Nit>=Nimin):
                    ima = np.flatnonzero(a==0)
                    ifg = np.flatnonzero(np.isnan(d))
                    PixList.append([c, l, Nd, Ni])
                    IfgToDelete.append(ifg)
                    ImagesToDelete.append(ima)

        # Reshape
        PixList = np.array(PixList).reshape((len(PixList),4))

        # Share lists to everyone
        self.PixList = np.concatenate(self.Com.allgather(PixList)).astype(int)
        self.IfgToDelete = list(itertools.chain.from_iterable(self.Com.allgather(IfgToDelete)))
        self.ImagesToDelete = list(itertools.chain.from_iterable(self.Com.allgather(ImagesToDelete)))

        # Number of pixels
        self.Npix = self.PixList.shape[0]
        assert self.Npix>minPix, 'Number of acceptable pixel is too low...'

        # Compute the starting and ending lines of each pixel and the starting and ending column of Glocal
        self.buildPixStartStop()

        # All done
        return

    def fillG(self):
        ''' 
        Fill the G matrix with appropriate numbers.
        '''

        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Fill the G matrix')
        self.PETSc.Sys.Print(' ')

        # Get the ownerships
        I = list(self.G.getOwnershipRange())

        # Do not handle the last lines if minimizeorbit==True
        if self.orbitMinWeight is not None and I[-1]==self.Nl:
            I[-1] -= self.nOrb

        # Do not handle the last line if orbitConstraints is not None
        if self.orbitConstraints is not None and I[-1]==self.Nl:
            I[-1] -= self.nOrb*len(self.orbitConstraints)

        # Where do we start?
        us, xs, ys, nis = self.line2pix(I[0])
        # Where do we end?
        ue, xe, ye, nie = self.line2pix(I[1]-1)

        # First pixel
        if nis>0:                    # If first pixel is incomplete, deal with its lines
            nli = self.PixStartStop[us,1] - I[0]
            for i in range(I[0],I[0]+nli):
                # Get the pixel position and the number of the line we want
                u, x, y, ni = self.line2pix(i) 
                # Get the corresponding line
                dline, indc, oline, indo = self.getGline(u,ni) 
                # Fill G
                self.G.setValues(i, indc, dline, self.INS)
                if self.orbit and (oline is not None):
                    self.G.setValues(i, indo, oline, self.INS) 
            # Update the first pixel to deal with
            us += 1

        # Last pixel
        if (nie<self.PixList[ue,2]-1):  # If last pixel is incomplete, deal with its lines
            # We deal line by line
            for i in range(I[1]-(nie+1),I[1]): 
                # Get the pixel position and the number of the line we want
                u, x, y, ni = self.line2pix(i) 
                # Get the corresponding line
                dline, indc, oline, indo = self.getGline(u,ni) 
                # Fill G                       
                self.G.setValues(i, indc, dline, self.INS) 
                if self.orbit and (oline is not None):
                    self.G.setValues(i, indo, oline, self.INS)
            # Update the last pixel to deal with
            ue -= 1

        # Other complete pixels
        for u in range(us,ue+1):
         
            # Get G
            G,iGr,iGc,O,iOr,iOc = self.getG(u)
            if len(iGc)*len(iGr)!=np.prod(G.shape):
                print('Soucis Worker {} Pixel {} {} \n\
                       iGc: {} - {} ({}) \n\
                       iGr: {} - {} ({})\n\
                       G: {}'.format(self.Com.Get_rank(), self.PixList[u,0], 
                                     self.PixList[u,1], iGc[0], iGc[-1], len(iGc),
                                     iGr[0], iGr[-1], len(iGr), G.shape))
                assert False, 'Die here'

            # Set the values
            self.G.setValues(iGr, iGc, G.flatten(), self.INS)
            if self.orbit:
                self.G.setValues(iOr, iOc, O.flatten(), self.INS)

        # Minimize orbits?
        if self.orbitMinWeight is not None:
            line = self.orbitMinWeight*np.ones((self.OrbShape,))
            for i in range(self.nOrb):
                iOr = self.Nl-(i+1)
                iOc = range(self.Nc-(i+1)*self.OrbShape, self.Nc-i*self.OrbShape)
                self.G.setValues(iOr, iOc, line.tolist(), self.INS)

        # Additional Constraints
        if self.orbitConstraints is not None:
            I = self.G.getOwnershipRange()
            nl = self.Nl - self.nOrb*len(self.orbitConstraints) - 1
            for orbit in self.orbitConstraints:
                nl += 1
                weight = self.orbitConstraints[orbit]['Weight']
                if I[0]<=nl<I[1]: self.G.setValue(nl, self.Npar+orbit, weight, self.INS)
                nl += 1
                if I[0]<=nl<I[1]: self.G.setValue(nl, self.Npar+orbit+self.Nsar, weight, self.INS)
                if self.nOrb>2:
                    nl += 1
                    if I[0]<=nl<I[1]: self.G.setValue(nl, self.Npar+orbit+2*self.Nsar, weight, self.INS)

        # All done
        return

    def filld(self):
        '''
        Fill the vector d with the appropriate numbers.
        '''

        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Fill the data vector')
        self.PETSc.Sys.Print(' ')

        # Who am I
        me = self.Com.Get_rank()

        # Get the ownership range
        Istart, Iend = self.d.getOwnershipRange()
        
        # Do not handle last line orbitConstraints is not None
        if self.orbitConstraints is not None and Iend==self.Nl:
            Iend -= self.nOrb*len(self.orbitConstraints)

        # Create a ifgsInG list
        self.ifgsInG = [[],[],[],[]]

        # Get ownerships
        I = self.d.getOwnershipRanges()
        
        # Do not handle last line orbitConstraints is not None
        if self.orbitConstraints is not None and I[-1]==self.Nl:
            I[-1] -= self.nOrb*len(self.orbitConstraints)

        # Create a dictionary of data/worker and a dictionary of index/worker
        d = {}
        d['data'] = []
        d['x'] = []
        d['y'] = []
        d['interfero'] = []
        d['lines'] = []

        # Store for each pixel into the appropriate data/worker vector
        for pix, pss in zip(self.PixList, self.PixStartStop):
            
            # Where do the data for this pixel start and stop
            Pst = pss[0]
            Ped = pss[1] 

            # On which worker are the starting and ending lines of this pixel
            Wst = self.line2rank(Pst)
            Wed = self.line2rank(Ped-1)

            # Do something only if I own that pixel
            if Wst==me or Wed==me:

                # Get x and y
                x = pix[0]
                y = pix[1]

                # Get the data
                data = self.igramsub[:,y,x]
                ilocal = np.where(np.isfinite(data))
                dlocal = data[ilocal].tolist()

                # Which interferograms are concerned
                ifglocal = (np.arange(self.Nifg)[ilocal]).astype(int).tolist()
                xlocal = (np.ones((len(ifglocal),))*x).astype(int).tolist()
                ylocal = (np.ones((len(ifglocal),))*y).astype(int).tolist()

                # 2 Cases:
                if Wst==Wed==me:                        # Both start and end are on the same worker
                    assert len(dlocal)==Ped-Pst, 'Whhhooooaaaaa.... {} {} '.format(x,y)
                    d['data'].extend(dlocal)
                    d['x'].extend(xlocal)
                    d['y'].extend(ylocal)
                    d['interfero'].extend(ifglocal)
                    d['lines'].extend(range(Pst,Ped))
                    #print('Worker {:2d}: {:4d} {:4d}          Full Pixel ({},{})'.format(me, len(dlocal), len(range(Pst,Ped)), x,y))
                elif Wst==me:                           # I own the starting point
                    ed = I[Wed] - Pst
                    assert len(dlocal[:ed])==I[Wed]-Pst, 'Whhhooooaaaaaaa... {} {} Starting Error'.format(x,y)
                    d['data'].extend(dlocal[:ed])
                    d['x'].extend(xlocal[:ed])
                    d['y'].extend(ylocal[:ed])
                    d['interfero'].extend(ifglocal[:ed])
                    d['lines'].extend(range(Pst,I[Wed]))
                    #print('Worker {:2d}: {:4d} {:4d}          Starting Point Owned'.format(me, len(dlocal[:ed]), len(range(Pst,I[Wed]))))
                elif Wed==me:                           # I own the ending point
                    st = I[Wed] - Pst
                    assert len(dlocal[st:])==Ped-I[Wed], 'Whhhooooaaaaaaa... {} {} Starting Error'.format(x,y)
                    d['data'].extend(dlocal[st:])
                    d['x'].extend(xlocal[st:])
                    d['y'].extend(ylocal[st:])
                    d['interfero'].extend(ifglocal[st:])
                    d['lines'].extend(range(I[Wed],Ped))
                    #print('Worker {:2d}: {:4d} {:4d}          Ending Point Owned'.format(me, len(dlocal[st:]), len(range(I[Wed],Ped))))

        # Store values in d
        self.d.setValues(d['lines'], d['data'], self.INS)

        # Save x, y, ifg and lines in ifgsInG
        self.ifgsInG = [d['x'], d['y'], d['interfero'], d['lines']]

        # Finalize ifgsInG
        self.ifgsInG = np.array(self.ifgsInG).T.astype(int)

        # Add constraints if needed 
        if self.orbitConstraints is not None:
            I = self.d.getOwnershipRange()
            nl = self.Nl - self.nOrb*len(self.orbitConstraints) - 1
            for orbit in self.orbitConstraints:
                weight = self.orbitConstraints[orbit]['Weight']
                x = self.orbitConstraints[orbit]['X ramp']
                nl += 1
                if I[0]<=nl<I[1]: self.d.setValue(nl, x*weight, self.INS)
                y = self.orbitConstraints[orbit]['Y ramp']
                nl += 1
                if I[0]<=nl<I[1]: self.d.setValue(nl, y*weight, self.INS)
                if self.nOrb>2:
                    c = self.orbitConstraints[orbit]['Constant']
                    nl += 1
                    if I[0]<=nl<I[1]: self.d.setValue(nl, c*weight, self.INS)

        # All done
        return

    def buildOrbitMatrix(self, includeConstant=True, looseConstraints=None, strongConstraints=None):
        '''
        Create the Full Orbit Matrix.
        The equation is orb = ax + by (+ c, if includeConstant is True).
        This will also include the reference for all interferograms (they can be shifted)
        orbits. 
            Args:

                    * includeConstant       : Estimates a constant term
                    * looseConstraints      : Adds constraints on the estimation of the orbits
                                              This just adds lines to G
                                              Default is None
                                              example: {0: {'X ramp': 1.2,
                                                            'Y ramp': 2.0,
                                                            'Constant': 0.0,
                                                            'Weight': 100.0},
                                                        22: {'X ramp': 1.3,
                                                             'Y ramp': 2.4,
                                                             'Constant': 0.0,
                                                             'Weight': 100.0}}
                    * strongConstraints     : Adds strong Constraints on the wanted parameters.
                                              The parameters are taken out of the inversion (equal to 0)
                                              Default is None

        '''
        
        # Stack them up
        Orb = copy.deepcopy(self.Jmat)

        # Strong Constraints
        self.strongConstraints = strongConstraints
        if strongConstraints is not None:
            assert type(strongConstraints) in (int,list), 'strongConstraints should be of Type int or list...'
            Orb = np.delete(Orb, strongConstraints, axis=1)

        # Add the referencing term
        refO = np.eye(self.Nifg) 

        # Save it
        self.OrbShape = Orb.shape[1]
        if includeConstant:
            self.Orb = np.hstack((Orb,Orb,Orb,refO))
            self.nOrb = 3
        else:
            self.Orb = np.hstack((Orb,Orb,refO))
            self.nOrb = 2

        # Save the orbital constraint
        self.orbitConstraints = looseConstraints
        self.strongConstraints = strongConstraints

        # All done
        return

    def AllocateGmd(self, dryanddie=False, orbit=True, nonzerosfactor=10):
        '''
        Allocates the PETSc matrices.
        '''

        # Get the size of the matrix
        self.getFullSize(orbit=orbit, nonzerosfactor=nonzerosfactor)

        # Print some things
        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Allocate G, m and d')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Data vector size: {}'.format(self.Nl))
        self.PETSc.Sys.Print(' Model parameter size: {}'.format(self.Nc))
        self.PETSc.Sys.Print(' Non Zero values in the diagonal blocks: {}'.format(self.d_nz))
        self.PETSc.Sys.Print(' Non Zero values in the off-diagonal part: {}'.format(self.o_nz))
        if dryanddie:
            self.PETSc.Sys.Print(' Dry run...')
            self.Finalize()
            self.HDF5close()
            sys.exit()

        # Data vector
        self.Allocated()

        # Model vector
        self.Allocatem()

        # Theory matrix
        self.AllocateG()

        # Wait 
        self.Barrier()

        # All done
        return

    def Allocated(self):
        '''
        Allocates the data vector.
        '''
        
        # Check if it exists
        if hasattr(self, 'd'):
            self.d.destroy()

        self.d = self.PETSc.Vec().createMPI(self.Nl, comm=self.Com)
        self.d.setOption(self.ZER,1) # Ignore zero entries on d

        # All done
        return
    
    def Allocatem(self):
        '''
        Allocates the model vector.
        '''

        # Check if it exists
        if hasattr(self, 'm'):
            self.m.destroy()

        self.PETSc.Sys.Print(' ')
        self.m = self.PETSc.Vec().createMPI(self.Nc, comm=self.Com)

        # All done
        return
    
    def AllocateG(self):
        '''
        Allocate the design matrix.
        '''

        # Check if it exists
        if hasattr(self, 'G'):
            self.G.destroy()

        self.G = self.PETSc.Mat().createAIJ([self.Nl, self.Nc], nnz = [self.d_nz, self.o_nz], 
                comm=self.Com)
        self.G.setOption(self.ZER,1)    # Ignore zero entrie on G

        # All done
        return

    def AssembleGmd(self):
        '''
        Assembles the vector d, m and the matrix G for PETSc.
        '''

        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Assembles G, m and d')
        self.PETSc.Sys.Print(' ')

        # Assemble d
        self.Assembled()

        # Assemble m
        self.Assemblem()

        # Assemble G
        self.AssembleG()

        # All done
        return

    def Assemblem(self):
        '''
        Assembles the model vector.
        '''

        self.m.assemble()

        # All done
        return

    def AssembleG(self):
        '''
        Assembles the Design matrix.
        '''

        self.G.assemble()

        # All done
        return

    def Assembled(self):
        '''
        Assembles the data vector.
        '''

        self.d.assemble()

        # All done
        return

    def line2rank(self, line):
        '''
        From the index of the line in the Global G, returns the
        worker affected to that line.
        Args:   
            * line      : Line of the Global G.
        '''

        # Get ownerships
        I = self.d.getOwnershipRanges()

        # find the index of the lowest line number above line in I
        up = np.flatnonzero(I>line)[0] - 1

        # return
        return up

    def Phi2ImgPix(self, onmyown=True):
        '''
        Builds an array (self.imgsInPhi) with 4 columns:
        [x, y, Img#, id] where id is the index of an element of self.Phi
        Args:
            * onmyown   : If True, will only do that over the lines owned by the mpi worker
        '''

        # Initialize 
        imgsInPhi = []

        # Loop over the columns
        if onmyown:
            I = self.Phi.getOwnershipRange()
            indexes = range(I[0], I[1])
        else:
            indexes = range(self.Nc)

        # Loop 
        for index in indexes:
            p, x, y, ni = self.phiLine2pix(index)
            if p is not None:
                imgsInPhi.append([x, y, ni, index])

        # Array-ize
        self.imgsInPhi = np.array(imgsInPhi).astype(int)

        # All done
        return

    def Glines2IfgPixels(self, onmyown=True):
        '''
        Creates an array (self.ifgsInG) with 4 columns:
        [x, y, ifg#, line] where line is the number of the line in global G.
        Args:
            * onmyown   : If True, will only do that over the lines owned by the mpi worker
        '''

        # Initialize a list
        ifgsInG = []

        # Loop over the lines of G (all or just what I own)
        if onmyown:
            I = self.d.getOwnershipRange()
            Lines = range(I[0], I[1])
        else:
            Lines = range(self.Nl)

        # Check minimize
        if Lines[-1]==self.Nl and self.orbitMinWeight is not None:
            Lines[-1] -= self.nOrb
        
        # Loop 
        for l in Lines:
            p, x, y, ni = self.line2pix(l, generalIfgNum=True)
            if p is not None:
                ifgsInG.append([x, y, ni, l])

        # Array-ize
        self.ifgsInG = np.array(ifgsInG).astype(int)

        # All done
        return

    def mIndex2ParamsPixels(self, onmyown=True):
        '''
        Creates an array (self.parsInG) with 4 columns:
        [x, y, par#, column], where line is the number of the line in Global G.
        Args:
            * onmyown   : If True, will onle do that over the columns owned by the worker.
        '''

        # Initialize a list
        parsInG = []

        # Loop over the indexes of m
        if onmyown:
            I = self.m.getOwnershipRange()
            Columns = range(I[0], I[1])
        else:
            Columns = range(self.Nc)

        # Loop
        for column in Columns:
            u, p = self.col2pix(column)
            if u is not None:
                x, y = self.PixList[u,:2]
                parsInG.append([x, y, p, column])

        # Save
        self.parsInG = np.array(parsInG)

        # All done
        return

    def whereisXYTinPhi(self, x, y, i):
        '''
        From the position of a pixel (x,y) and the date index, i, get the line in self.Phi. 
        '''

        # Which one is this pixel
        u = np.where( (self.PixList[:,0] == x) & (self.PixList[:,1] == y) )[0]

        # Check if the pixel is in the list
        if len(u) == 0:
            return None

        # Find the position in Phi
        pos = np.int(self.PixSSInc2Phi[u,0])

        # All done
        return pos

    def getDataSpace(self, vector='d'):
        '''
        Get the vector from the self.vector PETSc vector and send them to workers.
        Each worker will receive a number of residual image so they can work on them.
        '''

        # Which vector do I want
        if type(vector) is str:
            dataSpaceVector = self.__getattribute__(vector)
        else:
            dataSpaceVector = vector

        # Who am I
        me = self.Com.Get_rank()

        # Create the list of which ifg goes on which worker
        ifgsWanted = utils._split_seq(range(self.Nifg), self.Com.Get_size())

        # 1. Send the residuals to the workers who are going to work on them
        Packages = []   # In case nothing is sent here
        # Iterate over the workers
        for worker in range(self.Com.Get_size()):

            # Create the list of things to send
            ToSend = []
            # Iterate over the ifgs this worker takes care of 
            for ifg in ifgsWanted[worker]:
                
                # Find the lines corresponding to that interfero
                ii = np.flatnonzero(self.ifgsInG[:,2] == ifg)

                # Get the coordinates and lines
                indx = self.ifgsInG[ii,0]        # X coordinates of the pixels
                indy = self.ifgsInG[ii,1]        # Y coordinates of the pixels
                indo = self.ifgsInG[ii,3]        # Which lines are they in residuals

                # Get the values
                Values = dataSpaceVector.getValues(indo.astype(np.int32))

                # Make a package to send (x, y, data, ifg, worker-wher-it-comes-from, lines)
                if len(Values)>0:
                    ToSend.append([indx, indy, Values, ifg, me, indo])

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
        dataSpace = []
        for ifg in np.unique(Ifgs):
            # Find the good packages
            packs = np.flatnonzero(Ifgs==ifg)
            # Create a holder
            data = [[] for i in range(6)]
            # Iterate over these packages
            for p in packs:
                x, y, val, ifg, worker, lines = Packages[p]
                data[0].append(x)
                data[1].append(y)
                data[2].append(np.ones(x.shape)*worker)
                data[3].append(val)
                data[4].append(ifg)
                data[5].append(lines)
            # Concatenate what's needed
            data[0] = np.concatenate(data[0]).astype(int)
            data[1] = np.concatenate(data[1]).astype(int)
            data[2] = np.concatenate(data[2]).astype(int)
            data[3] = np.concatenate(data[3]).astype(float)
            data[4] = np.unique(data[4])[0]
            data[5] = np.concatenate(data[5]).astype(int)
            # Set residual in Residuals
            dataSpace.append(data)

        # All done
        return dataSpace

    def setbackDataSpace(self, dataSpace, vector):
        '''
        Sends the dataSpace vector by package to the worker and put them back in self.vector
        '''

        # Which vector am I working on 
        if type(vector) is str:
            sharedData = self.__getattribute__(vector)
        else:
            sharedData = vector

        # Who am I
        me = self.Com.Get_rank()

        # 1. Iterate over the residuals and send to workers
        Packages = []   # In case nothing is sent here
        for worker in range(self.Com.Get_size()):
            # Create the package to send
            ToSend = []
            # Iterate over the residuals
            for data in dataSpace:
                ii = np.flatnonzero(data[2]==worker)
                if len(ii)>0:
                    v = data[3][ii]
                    l = data[5][ii]
                    ToSend.append([v, l])
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
            values += package[0].tolist()
            indo += package[1].tolist()
        
        # 3. Set values in residuals
        sharedData.setValues(indo, values, self.INS)
        sharedData.assemble()

        # All done
        return

    def computePredicted(self):
        '''
        Computes the predicted d vector.
        '''

        # Allocates a predicted
        self.dpred = self.PETSc.Vec().createMPI(self.Nl, comm=self.Com)

        # Multiply G by m
        self.G.mult(self.m, self.dpred)

        # All done
        return

    def writePredicted2File(self, talktome=False):
        '''
        Writes the predicted data into a hdf5 file.
        Args:
            * talktome  : If True, prints stuffs to screen
        '''

        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Writing Predicted Ifgs to file')

        # Compute the predicted
        if talktome:
            self.PETSc.Sys.Print('          Computing Predicted data')
        self.computePredicted()

        # Create a numpy holder
        mm = np.zeros((self.Ny, self.Nx))

        # Check if the output directory exists
        mmout = self.hdfout.create_dataset('Predicted', shape=(self.Ndata, self.Ny, self.Nx), dtype='f')
        mmout.attrs['help'] = 'Predicted interferograms'
        self.Barrier()

        # Get the predicted data
        if talktome:
            self.PETSc.Sys.Print('          Getting Predicted data')
        Predicted = self.getDataSpace(vector='dpred')

        # Write the predictions
        if talktome:
            self.PETSc.Sys.Print('          Writing Predicted data')
        for predicted in Predicted:
            x, y, worker, values, ifg, lines = predicted
            mm[:, :] = np.nan; mm[y,x] = values
            if self.xRef is not None:
                mm[:,:] -= mm[self.yRef, self.xRef]
            mmout[ifg, :, :] = mm

        # End up 
        self.PETSc.Sys.Print(' ')

        # All done
        return

    def writeOrbits2File(self):
        '''
        Once m has been solved for, this routine stores the orbit parameters 
        in the h5 output file
        '''
        
        # Print stuff
        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Write Orbits to file ')
        
        # who am I
        me = self.Com.Get_rank()

        # Create the data set
        pout = self.hdfout.create_dataset('orbits', shape=(self.Nsar, 3))
        pout.attrs['help'] = 'Orbital functions, ax+by+c'

        rout = self.hdfout.create_dataset('reference', shape=(self.Nifg,))

        # Get Orbits
        Orbits, Indexes, Workers = self.getOrbits(vector='m', target=0)

        # Check if I have something
        if me==0:

            # Save them
            Orbits = np.array(Orbits)

            # Get the number of strongConstraints
            if self.strongConstraints is not None:
                if type(self.strongConstraints) is int:
                    ls = 1
                else:
                    ls = len(self.strongConstraints)
            else:
                ls = 0
    
            # Get the orbital terms
            x = Orbits[:self.Nsar-ls]
            y = Orbits[self.Nsar-ls:2*(self.Nsar-ls)]
            if self.nOrb==3:
                c = Orbits[2*(self.Nsar-ls):3*(self.Nsar-ls)]
            else:
                c = np.zeros((self.Nsar,))
            reference = Orbits[-self.Nifg:]

            # Expand from the strongConstraints
            if self.strongConstraints is not None:
                x = np.insert(x, self.strongConstraints, 0.)
                y = np.insert(y, self.strongConstraints, 0.)
                if self.nOrb==3:
                    c = np.insert(c, self.strongConstraints, 0.)

            # Set 
            pout[:,:] = np.vstack((x, y, c)).T
            rout[:] = reference

        # Clean screen 
        self.PETSc.Sys.Print(' ')

        # All done
        return

    def getOrbits(self, vector='m', target=None):
        '''
        Returns all the orbital parameters onto a worker.
        Returns also the where it came from and the index of the orbital parameter.
        
        Args:
            * target    : Rank of the worker that is going to receive the orbits.
            * vector    : Which vector is holding the orbit parameters (default: m).
        '''

        # Check
        if not self.orbit:
            return [],[],[]

        # Get the appropriate vector
        if type(vector) is str:
            m = self.__getattribute__(vector)
        else:
            m = vector

        # Get the index of the orbits
        Os = self.OrbShape
        nOrb = self.nOrb
        Nc = self.Nc
        indexes = np.array(range(Nc - nOrb*Os - self.Nifg, Nc))

        # What do I own
        I = m.getOwnershipRange()

        # What can I get from here
        indexes = indexes[np.logical_and(indexes>=I[0], indexes<I[1])].tolist()

        # Get the values
        orbits = m.getValues(indexes)

        # Gather these onto node worker
        recOrb = self.Com.gather(orbits, root=target)
        recInd = self.Com.gather(indexes, root=target)
        recWor = self.Com.gather((np.ones(orbits.shape)*self.Com.Get_rank()).astype(int), root=target)

        # If I am worker 'worker', flatten Orbit and indexes and return
        if self.Com.Get_rank()==target:
            Orbits = list(itertools.chain.from_iterable(recOrb))
            Indexes = list(itertools.chain.from_iterable(recInd))
            Workers = list(itertools.chain.from_iterable(recWor))
            return Orbits, Indexes, Workers
        else:
            return [], [], []

        # All done
        return

    def setbackOrbits(self, orblist, vector='m'):
        '''
        Sends the orbit parameters to the workers and put them back in the vector.
        '''

        # Which model vector do we work on 
        if type(vector) is str:
            m = self.__getattribute__(vector)
        else:
            m = vector

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
                m.setValues(ind, val, self.INS)

        # Assemble
        m.assemble()

        # All done
        return

    def writePhaseEvolution2File(self, talktome=False, name='recons', Jmat=True):
        '''
        Writes the phase evolution to a hdf5 file'
        '''

        # Who am I
        me = self.Com.Get_rank()

        self.PETSc.Sys.Print('-------------------------------------------------------')
        self.PETSc.Sys.Print(' ')
        self.PETSc.Sys.Print(' Writing Reconstructed Phase values to file')

        # Compute the phase evolution
        if talktome:
            self.PETSc.Sys.Print('          Compute Phase evolution')
        self.computePhaseEvolution()

        # Get datasets
        pout = self.hdfout.create_dataset(name, 
                                          (self.Nsar, self.Ny, self.Nx), 
                                          'f')
        pout.attrs['help'] = 'Reconstructed phase evolution'

        # Do the storage dates by dates
        if talktome:
            self.PETSc.Sys.Print('          Get and Write Phase Evolution')
        iDates = utils._split_seq(range(self.Nsar), self.Com.Get_size())

        # Get model space
        Phases = self.getPhaseEvolution()

        for phase in Phases:

            if talktome:
                print('Phase map #{} by worker {}'.format(phase[-1], me))

            # Get values
            Indx, Indy, Dateval, iIfg = phase[0], phase[1], phase[3], phase[4]

            # Prepare array
            pp = np.zeros((self.Ny, self.Nx))
            pp[:,:] = np.nan
            pp[Indy, Indx] = Dateval

            if self.xRef is not None:
                pp[:,:] -= pp[self.yRef,self.xRef]
            pout[iIfg,:,:] = pp

        # Set the masterind date to 0
        if talktome:
            self.PETSc.Sys.Print('          Reference to the master date')
        if me==0:
            pp = np.zeros((self.Ny, self.Nx))
            pp[:,:] = 0.
            imask, jmask = np.where(np.isnan(np.sum(pout, axis=0)))
            pp[imask, jmask] = np.nan
            pout[self.masterind, :, :] += pp
 
        # Wait here
        self.Barrier()

        # Put NaNs where there should be 
        if talktome:
            self.PETSc.Sys.Print('          Add NaNs where they should be')
        indexes = utils._split_seq(range(len(self.PixList)), 
                self.Com.Get_size())[me]
        for ind in indexes:
            pp = np.zeros((self.Nsar,))
            pp[:] = 0.
            x,y = self.PixList[ind,:2]
            images = self.ImagesToDelete[ind]
            pp[images] = np.nan
            pout[:,y,x] += pp

        # Create the Jmat matrix corresponding to the solved problem
        if Jmat:
            if talktome:
                self.PETSc.Sys.Print('          Deal with Jmat')
            Jmat = np.zeros((self.Nsar, self.Nsar))
            Jmat[:,self.masterind] = 1.
            Jmat += -1.0*np.eye(self.Nsar)
            Jmat[:,:self.masterind] *= -1.
            pout = self.hdfout.create_dataset('Jmat', data=Jmat)
            pout.attrs['help'] = 'Connectivity matrix [-1,1,0]'

        # Clean the line
        self.PETSc.Sys.Print(' ')

        # All done
        return

    def _cleanUp(self):
        '''
        Clean up the whole set of PETSc vectors and matrices.
        '''

        # These are the main guys to kill
        self.d.destroy()
        self.G.destroy()
        self.m.destroy()

        # Delete them
        del self.d
        del self.G
        del self.m

        # All done
        return

#########################
# For testing purposes

    def writeGtoFile(self,outfile='G.dat', verbose=True):
        '''
        Writes G to a file, binary.
        '''

        # Print some info
        if verbose:
            self.PETSc.Sys.Print('Writing G matrix to file {}'.format(outfile))
            self.PETSc.Sys.Print('Matrix size: {} Lines'.format(self.Nl))
            self.PETSc.Sys.Print('             {} Columns'.format(self.Nc))

        # Hou, dirty... and maybe wrong...
        #G = self.G.getDenseLocalMatrix()
        #ls, le = G.getOwnershipRange()
        #cs, ce = 0, self.Nc
        #G.getValues(range(ls,le), range(cs, ce)).tofile('{:03d}_{}'.format(self.rank, outfile))
        #self.Barrier()
        #if self.rank==0:
        #    nProc = self.Com.Get_size()
        #    toWrite = np.zeros((self.Nl, self.Nc))
        #    for i in range(nProc):
        #        ls, le = G.getOwnershipRanges()[i:i+2]
        #        cs, ce = 0, self.Nc
        #        toWrite[ls:le, cs:ce] = np.fromfile('{:03d}_{}'.format(self.rank, \
        #            outfile)).reshape((le-ls, ce-cs))
        #    toWrite = np.array(toWrite).flatten()
        #    toWrite.tofile(outfile)

        # Smart way but I do not understand the formats
        # Create a viewer
        fout = self.PETSc.Viewer()
        fout.createBinary('PETSc_'+outfile, mode=self.PETSc.Viewer.Mode.WRITE, format=self.PETSc.Viewer.Format.NATIVE, comm=self.Com)

        # View G
        self.G.view(viewer=fout)

        # Destroy the viewer
        fout.destroy()

        # All done
        return

    def writeIncVector2File(self, outfile='Inc.dat'):
        '''
        Writes Inc to a file, binary.
        '''

        # Create a viewer
        fout = self.PETSc.Viewer()
        fout.createASCII(outfile, mode=self.PETSc.Viewer.Mode.WRITE, format=self.PETSc.Viewer.Format.ASCII_DENSE, comm=self.Com)

        # View vector
        self.Inc.view(viewer=fout)

        # Destroy viewer
        fout.destroy()

        # All done
        return

    def writeDataVector2File(self, outfile='Data.dat'):
        '''
        Writes the d vector to a binary file.
        '''

        # Create a viewer
        fout = self.PETSc.Viewer()
        fout.createASCII(outfile, mode=self.PETSc.Viewer.Mode.WRITE, format=self.PETSc.Viewer.Format.ASCII_DENSE, comm=self.Com)

        # View vector
        self.d.view(viewer=fout)

        # Destroy viewer
        fout.destroy()

        # All done 
        return

    def writeModelVector2File(self, outfile='Model.dat'):
        '''
        Writes the m vector to a binary file.
        '''

        # Create a viewer
        fout = self.PETSc.Viewer()
        fout.createASCII(outfile, mode=self.PETSc.Viewer.Mode.WRITE, format=self.PETSc.Viewer.Format.ASCII_DENSE, comm=self.Com)

        # View vector
        self.m.view(viewer=fout)

        # Destroy viewer
        fout.destroy()

        # All done 
        return

#########################
# Will not be used for now

    def MapGPSin2InSAR(self, gps, latfile, lonfile, los, distance=1):
        '''
        Converts the lon and lat arrays of the gps into the radar geometry.
        Args:
            * gps       : gps structure from StaticInv. Can be gpsrates or gpstimesseries.
            * latfile   : file containing the latitude of each pixel.
            * lonfile   : file containing the longitude of each pixel.
            * los       : array (len=3) with the LOS unit vector.
            * distance  : distance of pixel averaging for GPS in km.
        '''

        if self.rank == 0:

            # Read the Lat and Lon files
            lat = np.fromfile(latfile, dtype=np.float32)
            lat = lat[self.stay:self.endy:self.stey,self.stax:self.endx:self.stex]
            lon = np.fromfile(lonfile, dtype=np.float32)
            lon = lon[self.stay:self.endy:self.stey,self.stax:self.endx:self.stex].flatten()

            # Compute the utm coordinates of the pixels
            sarx, sary = gps.lonlat2xy(lon, lat)

            # Create lists of points
            pp = np.vstack((x.flatten(), y.flatten())).T.tolist()

            # import shapely
            import shapely.geometry as geom
            
            # Build a list of point objects
            Pl = []
            for p in pp:
                Pl.append(geom.Point(p))

            # Create a storage location
            self.GPS = {}
            self.GPS['Number of Pixels'] = []
            self.GPS['Line'] = []
            self.GPS['Column'] = []

            # Loop over the GPSs
            for g in range(len(gps.lon)):
                # Initialize
                Np = 0
                l = None
                c = None
                # Create the gps point
                P1 = geom.Point([gps.x[g], gps.y[g]])
                # Get the points that are close to the GPS
                u = np.flatnonzero(np.array([P1.distance(p) for p in Pl]) < distance)
                # Get the number of points
                Np = len(u)
                # Get the indexes
                l,c = np.where((lon==lon.item(u)) & (lat==lat.item(u)))
                # Store those
                self.GPS['Number of Pixels'].append(Np)
                self.GPS['Line'].append(l)
                self.GPS['Column'].append(c)

        else:

            self.GPS = None

        # Broadcast
        self.GPS = self.Com.bcast(self.GPS, root=0)

        # All done
        return

