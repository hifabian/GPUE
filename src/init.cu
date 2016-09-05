/*
* init.cu - GPUE: Split Operator based GPU solver for Nonlinear 
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
*/

#include "../include/init.h"

template <typename O>
int initialise(O &opr, Cuda &cupar, Grid &par, Wave &wave){

    // Setting functions for operators
    opr.set_fns();

    // Re-establishing variables from parsed Grid class
    // Initializes uninitialized variables to 0 values
    std::string data_dir = par.sval("data_dir");
    int N = par.ival("atoms");
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    int threads;
    unsigned int gSize = xDim*yDim;
    double omega = par.dval("omega");
    double gdt = par.dval("gdt");
    double dt = par.dval("dt");
    double omegaX = par.dval("omegaX");
    double omegaY = par.dval("omegaY");
    double omegaZ = par.dval("omegaZ");
    double gammaY = par.dval("gammaY"); //Aspect ratio of trapping geometry.
    double l = par.dval("winding");
    double *x;
    double *y;
    double *xp;
    double *yp;
    double *Energy;
    double *r;
    double *V;
    double *V_opt;
    double *Phi;
    double *Phi_gpu;
    double *K;
    double *xPy;
    double *yPx;
    double *xPy_gpu;
    double *yPx_gpu;
    double *Energy_gpu;
    cufftDoubleComplex *wfc;
    cufftDoubleComplex *V_gpu;
    cufftDoubleComplex *EV_opt;
    cufftDoubleComplex *wfc_backup;
    cufftDoubleComplex *GK;
    cufftDoubleComplex *GV;
    cufftDoubleComplex *EV;
    cufftDoubleComplex *EK;
    cufftDoubleComplex *ExPy;
    cufftDoubleComplex *EyPx;
    cufftDoubleComplex *EappliedField; 
    cufftDoubleComplex *wfc_gpu;
    cufftDoubleComplex *K_gpu;
    cufftDoubleComplex *par_sum;

    std::cout << omegaX << '\t' << omegaY << '\n';
    std::cout << "xDim is: " << xDim << '\t' <<  "yDim is: " << yDim << '\n';

    cufftResult result = cupar.cufftResultval("result");
    cufftHandle plan_1d = cupar.cufftHandleval("plan_1d");
    cufftHandle plan_2d = cupar.cufftHandleval("plan_2d");

    dim3 grid = cupar.dim3val("grid");

    std::string buffer;
    double Rxy; //Condensate scaling factor.
    double a0x, a0y; //Harmonic oscillator length in x and y directions

    unsigned int xD=1,yD=1,zD=1;
    threads = 128;

    // number of blocks in simulation
    unsigned int b = xDim*yDim/threads;

    // largest number of elements
    unsigned long long maxElements = 65536*65536ULL; 

    if( b < (1<<16) ){
        xD = b;
    }
    else if( (b >= (1<<16) ) && (b <= (maxElements)) ){
        int t1 = log(b)/log(2);
        float t2 = (float) t1/2;
        t1 = (int) t2;
        if(t2 > (float) t1){
            xD <<= t1;
            yD <<= (t1 + 1);
        }
        else if(t2 == (float) t1){
            xD <<= t1;
            yD <<= t1;
        }
    }
    else{
        printf("Outside range of supported indexing");
        exit(-1);
    }
    printf("Compute grid dimensions chosen as X=%d    Y=%d\n",xD,yD);
    
    grid.x=xD; 
    grid.y=yD; 
    grid.z=zD; 
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    
    int i,j; //Used in for-loops for indexing
    
/*
    double xOffset, yOffset;
    xOffset=0.0;//5.0e-6;
    yOffset=0.0;//5.0e-6;
*/
    
    double mass = 1.4431607e-25; //Rb 87 mass, kg
    par.store("mass",mass);
    double a_s = 4.76e-9;
    par.store("a_s",a_s);

    double sum = 0.0;

    a0x = sqrt(HBAR/(2*mass*omegaX));
    a0y = sqrt(HBAR/(2*mass*omegaY));
    par.store("a0x",a0x);
    par.store("a0y",a0y);

    std::cout << "a0x and y are: " << a0x << '\t' << a0y << '\n';

    std::cout << N << '\t' << a_s << '\t' << mass << '\t' << omegaZ << '\n';
    
    Rxy = pow(15,0.2)*pow(N*a_s*sqrt(mass*omegaZ/HBAR),0.2);
    par.store("Rxy",Rxy);
    double bec_length = sqrt( HBAR/(mass*sqrt( omegaX*omegaX * 
                                               ( 1 - omega*omega) ) ));

    std::cout << "Rxy is: " << Rxy << '\n';
    double xMax = 6*Rxy*a0x; //10*bec_length; //6*Rxy*a0x;
    double yMax = 6*Rxy*a0y; //10*bec_length;
    par.store("xMax",xMax);
    par.store("yMax",yMax);

    double pxMax, pyMax;
    pxMax = (PI/xMax)*(xDim>>1);
    pyMax = (PI/yMax)*(yDim>>1);
    par.store("pyMax",pyMax);
    par.store("pxMax",pxMax);
    
    double dx = xMax/(xDim>>1);
    double dy = yMax/(yDim>>1);
    par.store("dx",dx);
    par.store("dy",dy);
    
    double dpx, dpy;
    dpx = PI/(xMax);
    dpy = PI/(yMax);
    std::cout << "yMax is: " << yMax << '\t' << "xMax is: " << xMax << '\n';
    std::cout << "dpx and dpy are:" << '\n';
    std::cout << dpx << '\t' << dpy << '\n';
    par.store("dpx",dpx);
    par.store("dpy",dpy);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    
    //double *x,*y,*xp,*yp;
    x = (double *) malloc(sizeof(double) * xDim);
    y = (double *) malloc(sizeof(double) * yDim);
    xp = (double *) malloc(sizeof(double) * xDim);
    yp = (double *) malloc(sizeof(double) * yDim);

    /*
     * R-space and K-space grids
     */
    std::cout << "dx and dy are: " << '\n';
    std::cout << dx << '\t' << dy << '\n';
    for(i=0; i<xDim/2; ++i){
        x[i] = -xMax + i*dx;        
        x[i + (xDim/2)] = i*dx;
        
        y[i] = -yMax + i*dy;        
        y[i + (yDim/2)] = i*dy;
        
        xp[i] = i*dpx;
        xp[i + (xDim/2)] = -pxMax + i*dpx;
        
        yp[i] = i*dpy;
        yp[i + (yDim/2)] = -pyMax + i*dpy;

        //std::cout << x[i] << '\t' << y[i] << '\t' << xp[i] << '\t' << yp[i] << '\n';
        //std::cout << x[i+xDim/2] << '\t' << y[i+xDim/2] << '\t' << xp[i+xDim/2] << '\t' << yp[i+xDim/2] << '\n';
    }

    par.store("x", x);
    par.store("y", y);
    par.store("xp", xp);
    par.store("yp", yp);
    

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    
    /* Initialise wavefunction, momentum, position, angular momentum, 
       imaginary and real-time evolution operators . */
    Energy = (double*) malloc(sizeof(double) * gSize);
    r = (double *) malloc(sizeof(double) * gSize);
    Phi = (double *) malloc(sizeof(double) * gSize);
    wfc = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    wfc_backup = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * 
                                               (gSize/threads));
    K = (double *) malloc(sizeof(double) * gSize);
    V = (double *) malloc(sizeof(double) * gSize);
    V_opt = (double *) malloc(sizeof(double) * gSize);
    GK = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    GV = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    EK = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    EV = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    EV_opt = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    xPy = (double *) malloc(sizeof(double) * gSize);
    yPx = (double *) malloc(sizeof(double) * gSize);
    ExPy = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    EyPx = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * gSize);
    EappliedField = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * 
                                                         gSize);
    
    /* Initialise wfc, EKp, and EVr buffers on GPU */
    cudaMalloc((void**) &Energy_gpu, sizeof(double) * gSize);
    cudaMalloc((void**) &wfc_gpu, sizeof(cufftDoubleComplex) * gSize);
    cudaMalloc((void**) &Phi_gpu, sizeof(double) * gSize);
    cudaMalloc((void**) &K_gpu, sizeof(cufftDoubleComplex) * gSize);
    cudaMalloc((void**) &V_gpu, sizeof(cufftDoubleComplex) * gSize);
    cudaMalloc((void**) &xPy_gpu, sizeof(cufftDoubleComplex) * gSize);
    cudaMalloc((void**) &yPx_gpu, sizeof(cufftDoubleComplex) * gSize);
    cudaMalloc((void**) &par_sum, sizeof(cufftDoubleComplex) * (gSize/threads));
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    std::cout << "all variables malloc'd" << '\n';

    #ifdef __linux
    int cores = omp_get_num_procs();
    par.store("Cores_Total",cores);

    // Assuming dev system specifics (Xeon with HT -> cores detected / 2)
    par.store("Cores_Max",cores/2);
    omp_set_num_threads(cores/2);
    std::cout << "GAMMAY IS: " << gammaY << '\n';
    #pragma omp parallel for private(j)
    #endif
    for( i=0; i < xDim; i++ ){
        for( j=0; j < yDim; j++ ){
            Phi[(i*yDim + j)] = fmod(l*atan2(y[j], x[i]),2*PI);
            
            if (par.bval("unit_test")){
                wfc[(i*yDim + j)].x =  (1/sqrt(2))*pow(1/PI,0.5) 
                    * exp( -0.5*( x[i]*x[i] + y[j]*y[j] ) )*(1+2*x[i]/sqrt(2));
                wfc[(i*yDim + j)].y = 0;
            }
            else{
                wfc[(i*yDim + j)].x = exp(-( pow((x[i])/(Rxy*a0x),2) + 
                                             pow((y[j])/(Rxy*a0y),2) ) ) *
                                      cos(Phi[(i*xDim + j)]);
                wfc[(i*yDim + j)].y = -exp(-( pow((x[i])/(Rxy*a0x),2) + 
                                              pow((y[j])/(Rxy*a0y),2) ) ) *
                                          sin(Phi[(i*xDim + j)]);
            }
                
            //V[(i*yDim + j)] = 0.5*mass*( pow(omegaX*(x[i]+xOffset),2) + 
            //                             pow(gammaY*omegaY*(y[j]+yOffset),2));
            V[(i*yDim + j)] = opr.V_fn(par.Vfn)(par, i, j, 0);
            //V[(i*yDim + j)] = 0;
            //K[(i*yDim + j)] = (HBAR*HBAR/(2*mass))*(xp[i]*xp[i]+yp[j]*yp[j]);
            K[(i*yDim + j)] = opr.K_fn(par.Kfn)(par, i, j, 0);
            // We want something like...
            // K[(i*yDim + j)] = opr.K_at(i,j);
            //K[(i*yDim + j)] = 0;

            GV[(i*yDim + j)].x = exp( -V[(i*xDim + j)]*(gdt/(2*HBAR)));
            GK[(i*yDim + j)].x = exp( -K[(i*xDim + j)]*(gdt/HBAR));
            GV[(i*yDim + j)].y = 0.0;
            GK[(i*yDim + j)].y = 0.0;
            
            xPy[(i*yDim + j)] = x[i]*yp[j];
            yPx[(i*yDim + j)] = -y[j]*xp[i];
            
            EV[(i*yDim + j)].x=cos( -V[(i*xDim + j)]*(dt/(2*HBAR)));
            EV[(i*yDim + j)].y=sin( -V[(i*xDim + j)]*(dt/(2*HBAR)));
            EK[(i*yDim + j)].x=cos( -K[(i*xDim + j)]*(dt/HBAR));
            EK[(i*yDim + j)].y=sin( -K[(i*xDim + j)]*(dt/HBAR));
            
            ExPy[(i*yDim + j)].x=cos(-omega*omegaX*xPy[(i*xDim + j)]*dt);
            ExPy[(i*yDim + j)].y=sin(-omega*omegaX*xPy[(i*xDim + j)]*dt);
            EyPx[(i*yDim + j)].x=cos(-omega*omegaX*yPx[(i*xDim + j)]*dt);
            EyPx[(i*yDim + j)].y=sin(-omega*omegaX*yPx[(i*xDim + j)]*dt);
    
            sum+=sqrt(wfc[(i*xDim + j)].x*wfc[(i*xDim + j)].x + 
                      wfc[(i*xDim + j)].y*wfc[(i*xDim + j)].y);
        }
    }

    std::cout << "writing initial variables to file..." << '\n';
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    //hdfWriteDouble(xDim, V, 0, "V_0"); //HDF COMING SOON!
    //hdfWriteComplex(xDim, wfc, 0, "wfc_0");
    //FileIO::writeOutDouble(buffer, data_dir + "V_opt",V_opt,xDim*yDim,0);
    FileIO::writeOutDouble(buffer, data_dir + "V",V,xDim*yDim,0);
    FileIO::writeOutDouble(buffer, data_dir + "K",K,xDim*yDim,0);
    FileIO::writeOutDouble(buffer, data_dir + "xPy",xPy,xDim*yDim,0);
    FileIO::writeOutDouble(buffer, data_dir + "yPx",yPx,xDim*yDim,0);
    FileIO::writeOut(buffer, data_dir + "WFC",wfc,xDim*yDim,0);
    FileIO::writeOut(buffer, data_dir + "ExPy",ExPy,xDim*yDim,0);
    FileIO::writeOut(buffer, data_dir + "EyPx",EyPx,xDim*yDim,0);
    FileIO::writeOutDouble(buffer, data_dir + "Phi",Phi,xDim*yDim,0);
    FileIO::writeOutDouble(buffer, data_dir + "r",r,xDim*yDim,0);
    FileIO::writeOutDouble(buffer, data_dir + "x",x,xDim,0);
    FileIO::writeOutDouble(buffer, data_dir + "y",y,yDim,0);
    FileIO::writeOutDouble(buffer, data_dir + "px",xp,xDim,0);
    FileIO::writeOutDouble(buffer, data_dir + "py",yp,yDim,0);
    FileIO::writeOut(buffer, data_dir + "GK",GK,xDim*yDim,0);
    FileIO::writeOut(buffer, data_dir + "GV",GV,xDim*yDim,0);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    std::cout << "wrote initial variables" << '\n';

    //free(V); 
    free(K); free(r); free(Phi);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    sum=sqrt(sum*dx*dy);
    //#pragma omp parallel for reduction(+:sum) private(j)
    for (i = 0; i < xDim; i++){
        for (j = 0; j < yDim; j++){
            wfc[(i*yDim + j)].x = (wfc[(i*yDim + j)].x)/(sum);
            wfc[(i*yDim + j)].y = (wfc[(i*yDim + j)].y)/(sum);
        }
    }
    
    std::cout << "modified wfc" << '\n';
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    
    std::cout << "xDim is: " << xDim << '\t' << "yDim is: " << yDim << '\n';
    std::cout << "plan_2d is: " << plan_2d << '\n';
    result = cufftPlan2d(&plan_2d, xDim, yDim, CUFFT_Z2Z);
    std::cout << "found result" << '\n';
    if(result != CUFFT_SUCCESS){
        printf("Result:=%d\n",result);
        printf("Error: Could not execute cufftPlan2d(%s ,%d, %d).\n", "plan_2d",
                (unsigned int)xDim, (unsigned int)yDim);
        return -1;
    }

    result = cufftPlan1d(&plan_1d, xDim, CUFFT_Z2Z, yDim);
    if(result != CUFFT_SUCCESS){
        printf("Result:=%d\n",result);
        printf("Error: Could not execute cufftPlan3d(%s ,%d ,%d ).\n", 
               "plan_1d", (unsigned int)xDim, (unsigned int)yDim);
        return -1;
    }
    
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    std::cout << GV[0].x << '\t' << GK[0].x << '\t' 
              << xPy[0] << '\t' << yPx[0] << '\n';

    std::cout << "storing variables..." << '\n';

    // Storing variables that have been initialized
    // Re-establishing variables from parsed Grid class
    // Initializes uninitialized variables to 0 values
    par.store("omega", omega);
    par.store("gdt", gdt);
    par.store("dt", dt);
    par.store("omegaX", omegaX);
    par.store("omegaY", omegaY);
    par.store("omegaZ", omegaZ);
    par.store("dx", dx);
    par.store("dy", dy);
    par.store("xMax", xMax);
    par.store("yMax", yMax);
    par.store("winding", l);
    par.store("x", x);
    par.store("y", y);
    par.store("xp", xp);
    par.store("yp", yp);
    wave.store("Energy", Energy);
    wave.store("r", r);
    opr.store("V", V);
    opr.store("V_opt", V_opt);
    wave.store("Phi", Phi);
    wave.store("Phi_gpu", Phi_gpu);
    opr.store("K", K);
    opr.store("xPy", xPy);
    opr.store("yPx", yPx);
    opr.store("Energy_gpu", Energy_gpu);
    par.store("atoms", N);
    par.store("xDim", xDim);
    par.store("yDim", yDim);
    par.store("threads", threads);
    wave.store("wfc", wfc);
    opr.store("V_gpu", V_gpu);
    opr.store("EV_opt", EV_opt);
    wave.store("wfc_backup", wfc_backup);
    opr.store("GK", GK);
    opr.store("GV", GV);
    opr.store("EV", EV);
    opr.store("EK", EK);
    opr.store("ExPy", ExPy);
    opr.store("EyPx", EyPx);
    opr.store("EappliedField", EappliedField);
    wave.store("wfc_gpu", wfc_gpu);
    opr.store("K_gpu", K_gpu);
    opr.store("xPy_gpu", xPy_gpu);
    opr.store("yPx_gpu", yPx_gpu);
    wave.store("par_sum", par_sum);

    cupar.store("result", result);
    cupar.store("plan_1d", plan_1d);
    cupar.store("plan_2d", plan_2d);

    cupar.store("grid", grid);

    std::cout << "variables stored" << '\n';

    return 0;
}

int main(int argc, char **argv){

    Grid par = parseArgs(argc,argv);
    Wave wave;
    Op opr;
    Cuda cupar;

    int device = par.ival("device");
    cudaSetDevice(device);

    std::string buffer;
    time_t start,fin;
    time(&start);
    printf("Start: %s\n", ctime(&start));

    //************************************************************//
    /*
    * Initialise the Params data structure to track params and variables
    */
    //************************************************************//

    initialise(opr, cupar, par, wave);

    std::cout << "initialized" << '\n';

    // Re-establishing variables from parsed Grid class
    std::string data_dir = par.sval("data_dir");
    double dx = par.dval("dx");
    double dy = par.dval("dy");
    double *x = par.dsval("x");
    double *y = par.dsval("y");
    double *V_opt = opr.dsval("V_opt");
    double *xPy = opr.dsval("xPy");
    double *yPx = opr.dsval("yPx");
    double *xPy_gpu = opr.dsval("xPy_gpu");
    double *yPx_gpu = opr.dsval("yPx_gpu");
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    bool read_wfc = par.bval("read_wfc");
    int gsteps = par.ival("gsteps");
    int esteps = par.ival("esteps");
    cufftDoubleComplex *wfc = wave.cufftDoubleComplexval("wfc");
    cufftDoubleComplex *V_gpu = opr.cufftDoubleComplexval("V_gpu");
    cufftDoubleComplex *GK = opr.cufftDoubleComplexval("GK");
    cufftDoubleComplex *GV = opr.cufftDoubleComplexval("GV");
    cufftDoubleComplex *EV = opr.cufftDoubleComplexval("EV");
    cufftDoubleComplex *EK = opr.cufftDoubleComplexval("EK");
    cufftDoubleComplex *ExPy = opr.cufftDoubleComplexval("ExPy");
    cufftDoubleComplex *EyPx = opr.cufftDoubleComplexval("EyPx");
    cufftDoubleComplex *wfc_gpu = wave.cufftDoubleComplexval("wfc_gpu");
    cufftDoubleComplex *K_gpu = opr.cufftDoubleComplexval("K_gpu");
    cufftDoubleComplex *par_sum = wave.cufftDoubleComplexval("par_sum");
    cudaError_t err = cupar.cudaError_tval("err");

    std::cout << "variables re-established" << '\n';
    std::cout << read_wfc << '\n';

    //************************************************************//
    /*
    * Groundstate finder section
    */
    //************************************************************//
    FileIO::writeOutParam(buffer, par, data_dir + "Params.dat");
    if(read_wfc){
        printf("Loading wavefunction...");
        wfc=FileIO::readIn("wfc_load","wfci_load",xDim, yDim);
        printf("Wavefunction loaded.\n");
    }

    std::cout << "gsteps: " << gsteps << '\n';
    
    if(gsteps > 0){
        err=cudaMemcpy(K_gpu, GK, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy K_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(V_gpu, GV, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy V_gpu to device" << '\n';
            exit(1);
        }
        FileIO::writeOut(buffer, data_dir + "GK1",GK,xDim*yDim,0);
        FileIO::writeOut(buffer, data_dir + "GV1",GV,xDim*yDim,0);
        err=cudaMemcpy(xPy_gpu, xPy, sizeof(double)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy xPy_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(yPx_gpu, yPx, sizeof(double)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy yPx_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(wfc_gpu, wfc, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy wfc_gpu to device" << '\n';
            exit(1);
        } 
        opr.store("yPx", yPx);
        opr.store("xPy", xPy);
        opr.store("GK", GK);
        opr.store("GV", GV);
        wave.store("wfc", wfc);
        opr.store("K_gpu", K_gpu);
        opr.store("V_gpu", V_gpu);
        wave.store("wfc_gpu", wfc_gpu);
        opr.store("xPy_gpu", xPy_gpu);
        opr.store("yPx_gpu", yPx_gpu);
        
        evolve(wave, opr, par_sum,
               gsteps, cupar, 0, 0, par, buffer);
        wfc = wave.cufftDoubleComplexval("wfc");
        wfc_gpu = wave.cufftDoubleComplexval("wfc_gpu");
        cudaMemcpy(wfc, wfc_gpu, sizeof(cufftDoubleComplex)*xDim*yDim, 
                   cudaMemcpyDeviceToHost);
    }

    std::cout << GV[0].x << '\t' << GK[0].x << '\t' 
              << xPy[0] << '\t' << yPx[0] << '\n';

    //free(GV); free(GK); free(xPy); free(yPx);

    // Re-initializing wfc after evolution
    //wfc = wave.cufftDoubleComplexval("wfc");
    //wfc_gpu = wave.cufftDoubleComplexval("wfc_gpu");

    std::cout << "evolution started..." << '\n';
    std::cout << "esteps: " << esteps << '\n';

    //************************************************************//
    /*
    * Evolution
    */
    //************************************************************//
    if(esteps > 0){
        err=cudaMemcpy(xPy_gpu, ExPy, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy xPy_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(yPx_gpu, EyPx, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy yPx_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(K_gpu, EK, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy K_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(V_gpu, EV, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy V_gpu to device" << '\n';
            exit(1);
        }
        err=cudaMemcpy(wfc_gpu, wfc, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
        if(err!=cudaSuccess){
            std::cout << "ERROR: Could not copy wfc_gpu to device" << '\n';
            exit(1);
        }

        opr.store("yPx", yPx);
        opr.store("xPy", xPy);
        opr.store("EK", EK);
        opr.store("EV", EV);
        wave.store("wfc", wfc);
        opr.store("K_gpu", K_gpu);
        opr.store("V_gpu", V_gpu);
        wave.store("wfc_gpu", wfc_gpu);
        opr.store("xPy_gpu", xPy_gpu);
        opr.store("yPx_gpu", yPx_gpu);

        FileIO::writeOutDouble(buffer, data_dir + "V_opt",V_opt,xDim*yDim,0);
        evolve(wave, opr, par_sum,
               esteps, cupar, 1, 0, par, buffer);
    
        wfc = wave.cufftDoubleComplexval("wfc");
        wfc_gpu = wave.cufftDoubleComplexval("wfc_gpu");
    }

    std::cout << "done evolving" << '\n';
    free(EV); free(EK); free(ExPy); free(EyPx);
    free(x);free(y);
    cudaFree(wfc_gpu); cudaFree(K_gpu); cudaFree(V_gpu); cudaFree(yPx_gpu); 
    cudaFree(xPy_gpu); cudaFree(par_sum);

    time(&fin);
    printf("Finish: %s\n", ctime(&fin));
    printf("Total time: %ld seconds\n ",(long)fin-start);
    return 0;
}
