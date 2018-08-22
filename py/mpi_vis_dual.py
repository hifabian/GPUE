'''
vis.py - GPUE: Split Operator based GPU solver for Nonlinear
Schrodinger Equation, Copyright (C) 2011-2015, Lee J. O'Riordan
<loriordan@gmail.com>, Tadhg Morgan, Neil Crowley. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import sys, os
from mpi4py import MPI
CPUs = 12#os.environ['SLURM_JOB_CPUS_PER_NODE']
from numpy import genfromtxt
import math as m
import matplotlib as mpl
import matplotlib.tri as tri
import numpy as np
import scipy as sp
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy.matlib
mpl.use('Agg')
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects
import ConfigParser
import random as r
from decimal import *
#import stats
#import hist3d

dat1 = sys.argv[1]
dat2 = sys.argv[2]

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank
print "Rank %d/%d initialised"%(rank,size-1)

if rank==0:
    if os.path.exists('dual'):
        import shutil
        shutil.rmtree('dual')
    os.mkdir('dual')
comm.Barrier()

getcontext().prec = 4
c = ConfigParser.ConfigParser()
getcontext().prec = 4
c = ConfigParser.ConfigParser()
c.readfp(open(dat1 + '/Params.dat'))

xDim = int(c.getfloat('Params','xDim'))
yDim = int(c.getfloat('Params','yDim'))
gndMaxVal = int(c.getfloat('Params','gsteps'))
evMaxVal = int(c.getfloat('Params','esteps'))
incr = int(c.getfloat('Params','printSteps'))
sep = (c.getfloat('Params','dx'))
dx = (c.getfloat('Params','dx'))
dy = (c.getfloat('Params','dy'))
dt = (c.getfloat('Params','dt'))
xMax = (c.getfloat('Params','xMax'))
yMax = (c.getfloat('Params','yMax'))
num_vort = 0#int(c.getfloat('Params','Num_vort'))

data = numpy.ndarray(shape=(xDim,yDim))

def delaunay(dataName,dataType,value):
	v_arr=genfromtxt(dataName + str(value) + dataType,delimiter=',' )
	data = np.array([[row[0],row[1]] for row in v_arr])
	dln = sp.spatial.Delaunay(data)
	plt.triplot(data[:,0],data[:,1],dln.simplices.copy(),linewidth=0.5,color='b',marker='.')
	plt.xlim(300,700);plt.ylim(300,700);
	plt.savefig('delaun_' + str(value) + '.png',dpi=200)
	print 'Saved Delaunay @ t=' + str(value)

def voronoi(dataName,dataType,value):
	v_arr=genfromtxt(dataName + str(value) + dataType,delimiter=',' )
	data = [[row[0],row[1]] for row in v_arr]
	vor = Voronoi(data)
	voronoi_plot_2d(vor)
	plt.xlim(300,700);plt.ylim(300,700);
	plt.savefig('voronoi_' + str(value) + '.png',dpi=200)
	print 'Saved Voronoi @ t=' + str(value)

def laplacian(density,name,imgdpi):
	gx,gy = np.gradient(density)
	g2x,gxgy = np.gradient(gx)
	gygx,g2y = np.gradient(gy)
	fig, ax = plt.subplots()
	#f = plt.quiver(gx,gy)
	f = plt.imshow((g2x**2 + g2y**2),cmap=plt.get_cmap('spectral'))
	cbar = fig.colorbar(f)
	plt.savefig(name + "_laplacian.png",dpi=imgdpi)
	plt.close()
	f = plt.imshow((gxgy - gygx),cmap=plt.get_cmap('spectral'))
	cbar = fig.colorbar(f)
	plt.savefig(name + "_dxdy.png",dpi=imgdpi)
	plt.close()

def struct_fact(density,name,imgdpi):
	fig, ax = plt.subplots()
	#f = plt.quiver(gx,gy)
	f = plt.imshow((np.abs(np.fft.fftshift(np.fft.fft2(density)))),cmap=plt.get_cmap('prism'))
	cbar = fig.colorbar(f)
	cbar.set_clim(1e6,1e11)
	plt.jet()
	plt.savefig(name + "_struct_log10.png",dpi=imgdpi)
	plt.close()

def opPot(dataName,imgdpi):
	data = open(dataName).read().splitlines()
	a = numpy.asanyarray(data,dtype='f8')
	b = np.reshape(a,(xDim,yDim))
	fig, ax = plt.subplots()
	f = plt.imshow((b))
	plt.gca().invert_yaxis()
	cbar = fig.colorbar(f)
	plt.jet()
	plt.savefig(dataName + ".png",dpi=imgdpi)
	plt.close()

def hist_gen(name,value,num_bins):
	v_arr=genfromtxt('vort_arr_' + str(value),delimiter=',' )
	H=[]
	count=0

	for i1 in range(0,v_arr.size/2):
		for i2 in range(i1,v_arr.size/2):
			H.append(m.sqrt( abs(v_arr[i1][0]*sep - v_arr[i2][0]*sep)**2  +  abs(v_arr[i1][1]*sep - v_arr[i2][1]*sep)**2 ))
			count = count + 1
	plt.title('Vortex lattice @ t=' + str(value*dt))
	plt.ticklabel_format(style='scientific')
	plt.ticklabel_format(style='scientific',axis='x', scilimits=(0,0))
	h = plt.hist(H, bins=num_bins)
	plt.savefig(name + "_" + str(value) + ".pdf")
	plt.close()

def image_gen(dataName, initValue, finalValue, increment,imgdpi):
	for i in range(initValue,finalValue,increment):
		if not os.path.exists(dataName+"r_"+str(i)+"_abspsi2.png"):
			real=open(dataName + '_' + str(i)).read().splitlines()
			img=open(dataName + 'i_' + str(i)).read().splitlines()
			a_r = numpy.asanyarray(real,dtype='f8') #64-bit double
			a_i = numpy.asanyarray(img,dtype='f8') #64-bit double
			a = a_r[:] + 1j*a_i[:]
			b = np.reshape(a,(xDim,yDim))
			f = plt.imshow(abs(b)**2)
			plt.jet()
			plt.gca().invert_yaxis()
			plt.savefig(dataName+"r_"+str(i)+"_abspsi2.png",dpi=imgdpi)
			plt.close()
			g = plt.imshow(np.angle(b))
			plt.gca().invert_yaxis()
			plt.savefig(dataName+"r_"+str(i)+"_phi.png",dpi=imgdpi)
			plt.close()
			f = plt.imshow(abs(np.fft.fftshift(np.fft.fft2(b)))**2)
			plt.gca().invert_yaxis()
			plt.jet()
			plt.savefig(dataName+"p_"+str(i)+"_abspsi2.png",dpi=imgdpi)
			plt.close()
			g = plt.imshow(np.angle(np.fft.fftshift(np.fft.fft2(b))))
			plt.gca().invert_yaxis()
			plt.savefig(dataName+"p_"+str(i)+"_phi.png",dpi=imgdpi)
			plt.close()
			print "Saved figure: " + str(i) + ".png"
			plt.close()
		else:
			print "File(s) " + str(i) +".png already exist."

def image_gen_single(dataName, value, imgdpi,opmode, x_dat, dat1, dat2, cbarOn=True, plot_vtx=False):
    real1=open(dat1 + '/' + dataName + '_' + str(0)).read().splitlines()
    img1=open(dat1 + '/' + dataName + 'i_' + str(0)).read().splitlines()
    a1_r = numpy.asanyarray(real1,dtype='f8') #128-bit complex
    a1_i = numpy.asanyarray(img1,dtype='f8') #128-bit complex
    a1 = a1_r[:] + 1j*a1_i[:]
    b1 = np.reshape(a1,(xDim,yDim))

    real2=open(dat2 + '/' + dataName + '_' + str(0)).read().splitlines()
    img2=open(dat2 + '/' + dataName + 'i_' + str(0)).read().splitlines()
    a2_r = numpy.asanyarray(real2,dtype='f8') #128-bit complex
    a2_i = numpy.asanyarray(img2,dtype='f8') #128-bit complex
    a2 = a2_r[:] + 1j*a2_i[:]
    b2 = np.reshape(a2,(xDim,yDim))

    m_val=np.max(np.abs(b2)**2)
    if not os.path.exists(dataName+"r_"+str(value)+"_abspsi2.png"):
        real1=open(dat1 + '/' + dataName + '_' + str(value)).read().splitlines()
        img1=open(dat1 + '/' + dataName + 'i_' + str(value)).read().splitlines()
        a1_r = numpy.asanyarray(real1,dtype='f8') #128-bit complex
        a1_i = numpy.asanyarray(img1,dtype='f8') #128-bit complex
        a1 = a1_r[:] + 1j*a1_i[:]
        b1 = np.transpose(np.reshape(a1,(xDim,yDim)))

        real2=open(dat2 + '/' + dataName + '_' + str(value)).read().splitlines()
        img2=open(dat2 + '/' + dataName + 'i_' + str(value)).read().splitlines()
        a2_r = numpy.asanyarray(real2,dtype='f8') #128-bit complex
        a2_i = numpy.asanyarray(img2,dtype='f8') #128-bit complex
        a2 = a2_r[:] + 1j*a2_i[:]
        b2 = np.transpose(np.reshape(a2,(xDim,yDim))) #Transpose to match matlab plots

        startBit = 0x00

        try:
            vorts1 = np.loadtxt(dat1 + '/' + 'vort_ord_' + str(value) + '.csv', delimiter=',', unpack=True)
            vorts2 = np.loadtxt(dat2 + '/' + 'vort_ord_' + str(value) + '.csv', delimiter=',', unpack=True)
        except Exception as e:
            print "Failed to load the required data file: %s"%e
            print "Please run vort.py before plotting the density, if you wih to have correctly numbered vortices"
            vorts1 = np.loadtxt(dat1 + '/' + 'vort_arr_' + str(value), delimiter=',', unpack=True, skiprows=1)
            vorts2 = np.loadtxt(dat2 + '/' + 'vort_arr_' + str(value), delimiter=',', unpack=True, skiprows=1)
            startBit=0x01

        if opmode & 0b100000 > 0:
            nameStr = dataName+"r_"+str(value)

            fig = plt.figure(figsize=(8, 4.5))

            #Plot of dat1 data
            ax1 = fig.add_subplot(121)
            f1 = ax1.imshow( (abs(b1)**2),
                                cmap='hot', vmin=0, vmax=5.4e7,
                                interpolation='none',
                                extent=[-xMax, xMax, -xMax, xMax])

            #plt.axis('equal')
            plt.axis([-1.25e-4, 1.25e-4, -1.25e-4, 1.25e-4])

            plt.axis('off')
            plt.gca().invert_yaxis()

            ax2 = fig.add_subplot(122)
            f2 = ax2.imshow( (abs(b2)**2),
                                cmap='hot', vmin=0, vmax=5.4e7,
                                interpolation='none',
                                extent=[-xMax, xMax, -xMax, xMax])
            #plt.axis('equal')
            plt.axis([-1.25e-4,1.25e-4,-1.25e-4,1.25e-4])

            plt.axis('off')
            plt.gca().invert_yaxis()

            tstr = "%.2f" % (value*dt)
            plt.suptitle('t=' + tstr + " s", fontsize=18, y=0.9)

            if cbarOn==True:
                cb_ax = fig.add_axes([0.93, 0.18, 0.02, 0.62])
                cbar = fig.colorbar(f2, cax=cb_ax)

            if plot_vtx==True:
                if startBit==0x00:
                    zVort1 = zip(vorts1[0,:],vorts1[1,:], vorts1[3,:])
                    zVort2 = zip(vorts2[0,:],vorts2[1,:], vorts2[3,:])
                else:
                    zVort1 = zip(vorts1[1,:], vorts1[3,:], [0, 1, 2, 3])
                    zVort2 = zip(vorts2[1,:], vorts2[3,:], [0, 1, 2, 3])
                for r1, r2 in zip(zVort1, zVort2):

                    if r1[2]==0:
                        txt1 = ax1.text((r1[0]-xDim/2)*dx, -(r1[1]-yDim/2)*dy, str(int(r1[2])), color='#379696', fontsize=6, alpha=0.7)
                        txt1.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='#B9EA56')])
                    else:
                        txt1 = ax1.text((r1[0]-xDim/2)*dx, -(r1[1]-yDim/2)*dy, str(int(r1[2])), color='#B9EA56', fontsize=6, alpha=0.7)
                        txt1.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='#379696')])

                    if r2[2]==0:
                        txt2 = ax2.text((r2[0]-xDim/2)*dx, -(r2[1]-yDim/2)*dy, str(int(r2[2])), color='#379696', fontsize=6, alpha=0.7)
                        txt2.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='#B9EA56')])
                    else:
                        txt2 = ax2.text((r2[0]-xDim/2)*dx, -(r2[1]-yDim/2)*dy, str(int(r2[2])), color='#B9EA56', fontsize=6, alpha=0.7)
                        txt2.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='#379696')])
            if plot_vtx==True:
                plt.savefig('dual/' + dataName+"r_"+str(value)+"_abspsi2_num.png",dpi=imgdpi, bbox_inches='tight')
            else:
                plt.savefig('dual/' + dataName+"r_"+str(value)+"_abspsi2_nonum.png",dpi=imgdpi, bbox_inches='tight')
            plt.close()

        if opmode & 0b010000 > 0:
            fig, ax = plt.subplots()
            g = plt.imshow(np.angle(b))
            cbar = fig.colorbar(g)
            plt.gca().invert_yaxis()
            plt.title('theta(r) @ t=' + str(value*dt))
            plt.savefig(dataName+"r_"+str(value)+"_phi.png",dpi=imgdpi)
            plt.close()

        if opmode & 0b001000 > 0:
            fig, ax = plt.subplots()
            f = plt.imshow(abs(np.fft.fftshift(np.fft.fft2(b)))**2)
            cbar = fig.colorbar(f)
            plt.gca().invert_yaxis()
            plt.jet()
            plt.title('rho(p) @ t=' + str(value*dt))
            plt.savefig(dataName+"p_"+str(value)+"_abspsi2.png",dpi=imgdpi)
            plt.close()

        if opmode & 0b000100 > 0:
            fig, ax = plt.subplots()
            g = plt.imshow(np.angle(np.fft.fftshift(np.fft.fft2(b))))
            cbar = fig.colorbar(g)
            plt.gca().invert_yaxis()
            plt.title('theta(p) @ t=' + str(value*dt))
            plt.savefig(dataName+"p_"+str(value)+"_phi.png",dpi=imgdpi)
            plt.close()

        if opmode & 0b000010 > 0:
            struct_fact(abs(b)**2,dataName+"_" + str(value),imgdpi)

        if opmode & 0b000001 > 0:
		    laplacian(abs(b)**2,dataName+"_" + str(value),imgdpi)

        print "Saved figure: " + str(value) + ".png"
        plt.close()
    else:
        print "File(s) " + str(value) +".png already exist."

def vort_traj(name,imgdpi):
	evMaxVal_l = evMaxVal
	H=genfromtxt('vort_arr_0',delimiter=',' )
	count=0
	for i1 in range(incr,evMaxVal_l,incr):
		try:
			v_arr=genfromtxt('vort_lsq_' + str(i1) + '.csv',delimiter=',' )
			H=np.column_stack((H,v_arr))
		except:
			evMaxVal_l = i1
			break
	X=np.zeros((evMaxVal_l/incr),dtype=np.float64)
	Y=np.zeros((evMaxVal_l/incr),dtype=np.float64)
	H=np.reshape(H,([num_vort,2,evMaxVal_l/incr]),order='F')
	for i1 in range(0, num_vort):
		for i2 in range(0,evMaxVal_l/incr):
			X[i2]=(H[i1,0,i2]*dx) - xMax
			Y[i2]=(H[i1,1,i2]*dx) - yMax
		h = plt.plot(X,Y,color=(r.random(),r.random(),r.random(),0.85),linewidth=0.1)
	plt.axis('equal')
	plt.title('Vort(x,y) from t=0 to t='+str(evMaxVal_l*dt)+" s")

	plt.axis((-xMax/2.0, xMax/2.0, -yMax/2.0, yMax/2.0))
	plt.ticklabel_format(style='scientific')
	plt.ticklabel_format(style='scientific',axis='x', scilimits=(0,0))
	plt.ticklabel_format(style='scientific',axis='y', scilimits=(0,0))
	plt.savefig(name +".pdf")
	plt.close()
	print "Trajectories plotted."

def scaleAxis(data,dataName,label,value,imgdpi):
	fig, ax = plt.subplots()
	ax.xaxis.set_major_locator(ScaledLocator(dx=dx))
	ax.xaxis.set_major_formatter(ScaledLocator(dx=dx))
	f = plt.imshow(abs(data)**2)
	cbar = fig.colorbar(f)
	plt.gca().invert_yaxis()
	plt.jet()
	plt.savefig(dataName+"r_"+str(value)+"_"+label +".png",dpi=imgdpi)
	plt.close()

def overlap(dataName, initValue, finalValue, increment):
	real=open(dataName + '_' + str(0)).read().splitlines()
	img=open(dataName + 'i_' + str(0)).read().splitlines()
	a_r = numpy.asanyarray(real,dtype='f8') #128-bit complex
	a_i = numpy.asanyarray(img,dtype='f8') #128-bit complex
	wfc0 = a_r[:] + 1j*a_i[:]
	for i in range(initValue,finalValue,increment):
		real=open(dataName + '_' + str(value)).read().splitlines()
		img=open(dataName + 'i_' + str(value)).read().splitlines()
		a_r = numpy.asanyarray(real,dtype='f8') #128-bit complex
		a_i = numpy.asanyarray(img,dtype='f8') #128-bit complex
		a = a_r[:] + 1j*a_i[:]
		b = np.dot(wfc0,a)
		print i, np.sum(b)

if __name__ == '__main__':
    import sys, os
    cbarOn = sys.argv[3].lower() == 'true'
    plot_vtx = sys.argv[4].lower() == 'true'

    gndImgList=[]
    evImgList=[]
    x_coord = np.loadtxt(dat1 + '/x_0', unpack=True)

    arrG = np.array_split( xrange(0,gndMaxVal,incr), size)
    arrE = np.array_split( xrange(0,evMaxVal,incr), size)

    for i in arrG[rank]:
        gndImgList.append(i)
    for i in arrE[rank]:
        evImgList.append(i)

	while gndImgList:
		i=gndImgList.pop()
		image_gen_single("wfc_0_ramp",i,300,0b110000)
		image_gen_single("wfc_0_const",i,300,0b110000)
    while evImgList:
        print "Processing data index=%d on rank=%d"%(i,rank)
        i=evImgList.pop()
        image_gen_single("wfc_ev", i, 600, 0b100000, x_coord, dat1, dat2, cbarOn, plot_vtx)
        print "Processed data index=%d on rank=%d"%(i,rank)
