import os, copy
import numpy
from scipy import sparse
from scipy.sparse.linalg.dsolve import factorized
#from pyspline import pySpline

class pyGeo():

    def _readIges(self, fileName):
        """Load a Iges file and create the splines to go with each patch
        Parameters
        ----------
        fileName : str
            Name of file to load.
        """
        f = open(fileName, 'r')
        Ifile = []
        for line in f:
            line = line.replace(';', ',')  #This is a bit of a hack...
            Ifile.append(line)
        f.close()

        start_lines   = int((Ifile[-1][1:8]))
        general_lines = int((Ifile[-1][9:16]))
        directory_lines = int((Ifile[-1][17:24]))
        parameter_lines = int((Ifile[-1][25:32]))

        # Now we know how many lines we have to deal with
        dir_offset  = start_lines + general_lines
        para_offset = dir_offset + directory_lines

        surf_list = []
        # Directory lines is ALWAYS a multiple of 2
        for i in range(directory_lines//2):
            # 128 is bspline surface type
            if int(Ifile[2*i + dir_offset][0:8]) == 128:
                start = int(Ifile[2*i + dir_offset][8:16])
                num_lines = int(Ifile[2*i + 1 + dir_offset][24:32])
                surf_list.append([start, num_lines])

        self.nSurf = 1 #

        print('Found %d surfaces in Iges File.'%(len(surf_list)))

        self.surfs = []

        for isurf in range(self.nSurf):  # Loop over our patches
            print(isurf)
            data = []
            # Create a list of all data
            # -1 is for conversion from 1 based (iges) to python
            para_offset = surf_list[isurf][0]+dir_offset+directory_lines-1

            for i in range(surf_list[isurf][1]):
                aux = Ifile[i+para_offset][0:69].split(',')
                for j in range(len(aux)-1):
                    data.append(float(aux[j]))

            # Now we extract what we need
            Nctlu = int(data[1]+1)
            Nctlv = int(data[2]+1)
            ku    = int(data[3]+1)
            kv    = int(data[4]+1)

            counter = 10
            tu = data[counter:counter+Nctlu+ku]
            counter += (Nctlu + ku)

            tv = data[counter:counter+Nctlv+kv]
            counter += (Nctlv + kv)
            #print('tu',tu)
            #print('tv',tv)
            weights = data[counter:counter+Nctlu*Nctlv]
            weights = numpy.array(weights)
            if weights.all() != 1:
                print('WARNING: Not all weight in B-spline surface are\
 1. A NURBS surface CANNOT be replicated exactly')
            counter += Nctlu*Nctlv

            coef = numpy.zeros([Nctlu, Nctlv, 3])
            for j in range(Nctlv):
                for i in range(Nctlu):
                    coef[i, j, :] = data[counter:counter +3]
                    counter += 3

            # Last we need the ranges
            prange = numpy.zeros(4)

            prange[0] = data[counter    ]
            prange[1] = data[counter + 1]
            prange[2] = data[counter + 2]
            prange[3] = data[counter + 3]

            # Re-scale the knot vectors in case the upper bound is not 1
            tu = numpy.array(tu)
            tv = numpy.array(tv)
            '''
            if not tu[-1] == 1.0:
                tu /= tu[-1]

            if not tv[-1] == 1.0:
                tv /= tv[-1]
            '''
        return tu, tv, Nctlu, Nctlv, ku, kv #coef, 

iges = pyGeo()
tu, tv, Nctlu, Nctlv, ku, kv = iges._readIges('CAD_new/uCRM-9_wingbox.igs')
print(tu)
print(tv)
print(tu.shape)
print(tv.shape)
print('Nctlu',Nctlu, 'Nctlv', Nctlv, 'ku', ku, 'kv', kv)
#print(numpy.shape(iges1))
#iges1 = iges1.reshape((13 * 28, 3))
#print(numpy.shape(iges1))
#numpy.savetxt("cps_0.txt",iges1,fmt='%12.4f') 