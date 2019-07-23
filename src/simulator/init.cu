#include "init.h"
#include "dynamic.h"
#include "split_op.h"

void check_memory(Grid &par){
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    int zDim = par.ival("zDim");
    int wfc_num = par.ival("wfc_num");

    bool energy_calc = par.bval("energy_calc");

    int gSize = xDim*yDim*zDim;
    size_t free = 0;
    size_t total = 0;

    cudaHandleError( cudaMemGetInfo(&free, &total) );

    // Note that this check is specifically for the case where we need to keep
    // 8 double2* values on the GPU. This is not the case for dynamic fields
    // and the test should be updated accordingly as these are used more.
    size_t req_memory = 16*8*(size_t)gSize*(size_t)wfc_num;
    if (energy_calc){
        req_memory += 4*16*(size_t)gSize*(size_t)wfc_num;
    }
    if (free < req_memory){
        std::cout << "Not enough GPU memory for gridsize!\n";
        std::cout << "Free memory is: " << free << '\n';
        std::cout << "Required memory is: " << req_memory << '\n';
        if (energy_calc){
            std::cout << "Required memory for energy calc is: "
                      << 4*16*(size_t)gSize << '\n';
        }
        std::cout << "xDim is: " << xDim << '\n';
        std::cout << "yDim is: " << yDim << '\n';
        std::cout << "zDim is: " << zDim << '\n';
        std::cout << "gSize is: " << gSize << '\n';
        exit(1);
    }
}

int init(Grid &par){

    check_memory(par);
    set_fns(par);

    // Re-establishing variables from parsed Grid class
    // Initializes uninitialized variables to 0 values
    std::string data_dir = par.sval("data_dir");
    int dimnum = par.ival("dimnum");
    int N = par.ival("atoms");
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    int zDim = par.ival("zDim");
    int wfc_num = par.ival("wfc_num");
    int step_offset = par.ival("step_offset");
    bool write_file = par.bval("write_file");
    bool cyl_coord = par.bval("cyl_coord");
    bool corotating = par.bval("corotating");
    dim3 threads;
    unsigned int gSize = xDim;
    if (dimnum > 1){
        gSize *= yDim;
    }
    if (dimnum > 2){
        gSize *= zDim;
    }
    double gdt = par.dval("gdt");
    double dt = par.dval("dt");
    double omegaX = par.dval("omegaX");
    double omegaY = par.dval("omegaY");
    double omegaZ = par.dval("omegaZ");
    double gammaY = par.dval("gammaY"); //Aspect ratio of trapping geometry.
    double winding = par.dval("winding");
    double box_size = par.dval("box_size");
    double *Energy;
    double *r;
    std::vector<double *> V_opt(wfc_num);
    double *Energy_gpu;
    std::vector<cufftDoubleComplex *> wfc_array(wfc_num);
    if (par.bval("read_wfc") == true){
        for (int i = 0; i < wfc_array.size(); ++i){
            wfc_array = par.d2svecval("wfc_array");
        }
    }
    std::vector<cufftDoubleComplex *> EV_opt(wfc_num);
    cufftDoubleComplex *wfc_backup;
    cufftDoubleComplex *EappliedField;

    std::cout << "gSize is: " << gSize << '\n';
    cufftHandle plan_1d;
    cufftHandle plan_2d;
    cufftHandle plan_3d;
    cufftHandle plan_other2d;
    cufftHandle plan_dim2;
    cufftHandle plan_dim3;

    std::string buffer;
    double Rxy; //Condensate scaling factor.
    double a0x, a0y, a0z; //Harmonic oscillator length in x and y directions

    generate_grid(par);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    double mass = 1.4431607e-25; //Rb 87 mass, kg
    par.store("mass",mass);
    double a_s = 4.76e-9;
    par.store("a_s",a_s);

    double sum = 0.0;

    a0x = sqrt(HBAR/(2*mass*omegaX));
    a0y = sqrt(HBAR/(2*mass*omegaY));
    a0z = sqrt(HBAR/(2*mass*omegaZ));
    par.store("a0x",a0x);
    par.store("a0y",a0y);
    par.store("a0z",a0z);

    // Let's go ahead and define the gDensConst here
    // N*4*HBAR*HBAR*PI*(4.67e-9/mass)*sqrt(mass*(omegaZ)/(2*PI*HBAR)
    double gDenConst = N*4*HBAR*HBAR*PI*(a_s/mass);
    if (dimnum == 2){
        gDenConst*= sqrt(mass*(omegaZ)/(2*PI*HBAR));
    }
    par.store("gDenConst", gDenConst);

    Rxy = pow(15,0.2)*pow(N*a_s*sqrt(mass*omegaZ/HBAR),0.2);
    par.store("Rxy",Rxy);

    //std::cout << "Rxy is: " << Rxy << '\n';
    double xMax, yMax, zMax;
    if (box_size > 0){
        xMax = box_size;
        yMax = box_size;
        zMax = box_size;
    }
    else{
        xMax = 6*Rxy*a0x;
        yMax = 6*Rxy*a0y;
        zMax = 6*Rxy*a0z;
    }
    par.store("xMax",xMax);
    par.store("yMax",yMax);
    par.store("zMax",zMax);

    double pxMax, pyMax, pzMax;
    pxMax = (PI/xMax)*(xDim>>1);
    pyMax = (PI/yMax)*(yDim>>1);
    pzMax = (PI/zMax)*(zDim>>1);
    par.store("pyMax",pyMax);
    par.store("pxMax",pxMax);
    par.store("pzMax",pzMax);

    double dx = xMax/(xDim>>1);
    double dy = yMax/(yDim>>1);
    double dz = zMax/(zDim>>1);
    if (dimnum < 3){
        dz = 1;
    }
    if (dimnum < 2){
        dy = 1;
    }
    par.store("dx",dx);
    par.store("dy",dy);
    par.store("dz",dz);

    double dpx, dpy, dpz;
    dpx = PI/(xMax);
    dpy = PI/(yMax);
    dpz = PI/(zMax);
    //std::cout << "yMax is: " << yMax << '\t' << "xMax is: " << xMax << '\n';
    //std::cout << "dpx and dpy are:" << '\n';
    //std::cout << dpx << '\t' << dpy << '\n';
    par.store("dpx",dpx);
    par.store("dpy",dpy);
    par.store("dpz",dpz);


    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    /* Initialise wavefunction, momentum, position, angular momentum,
       imaginary and real-time evolution operators . */
    Energy = (double*) malloc(sizeof(double) * gSize);
    r = (double *) malloc(sizeof(double) * gSize);
    for (int i = 0; i < wfc_array.size(); ++i){
        V_opt[i] = (double *) malloc(sizeof(double) * gSize);
        EV_opt[i] = (cufftDoubleComplex *)
                 malloc(sizeof(cufftDoubleComplex) * gSize);
    }
    EappliedField = (cufftDoubleComplex *) 
                    malloc(sizeof(cufftDoubleComplex) * gSize);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

/*
    #ifdef __linux
    int cores = omp_get_num_procs();
    par.store("Cores_Total",cores);

    // Assuming dev system specifics (Xeon with HT -> cores detected / 2)
    par.store("Cores_Max",cores/2);
    omp_set_num_threads(cores/2);

    //#pragma omp parallel for private(j)
    #endif
*/

    par.store("gSize", xDim*yDim*zDim);
    if (par.bval("use_param_file")){
        parse_param_file(par);
    }
    std::cout << "generating fields...\n";
    generate_fields(par);
    std::cout << "generated fields...\n";
    std::vector<double *> K = par.dsvecval("K");
    std::vector<double *> Ax = par.dsvecval("Ax");
    std::vector<double *> Ay = par.dsvecval("Ay");
    std::vector<double *> Az = par.dsvecval("Az");
    std::vector<double *> V = par.dsvecval("V");

    std::vector<double *> pAx = par.dsvecval("pAx");
    std::vector<double *> pAy = par.dsvecval("pAy");
    std::vector<double *> pAz = par.dsvecval("pAz");

    double *x = par.dsval("x");
    double *y = par.dsval("y");
    double *z = par.dsval("z");

    std::vector<double2 *> GpAx = par.d2svecval("GpAx");
    std::vector<double2 *> GpAy = par.d2svecval("GpAy");
    std::vector<double2 *> GpAz = par.d2svecval("GpAz");
    std::vector<double2 *> EpAx = par.d2svecval("EpAx");
    std::vector<double2 *> EpAy = par.d2svecval("EpAy");
    std::vector<double2 *> EpAz = par.d2svecval("EpAz");

    std::vector<double2 *> GV = par.d2svecval("GV");
    std::vector<double2 *> EV = par.d2svecval("EV");
    std::vector<double2 *> GK = par.d2svecval("GK");
    std::vector<double2 *> EK = par.d2svecval("EK");

    wfc_array = par.d2svecval("wfc_array");


    for(int i=0; i < wfc_array.size(); i++ ){
        for (int j = 0; j < gSize; ++j){
            sum+=sqrt(wfc_array[i][j].x*wfc_array[i][j].x
                      +wfc_array[i][j].y*wfc_array[i][j].y);
        }
    }

    if (write_file){
        FileIO::init(par);
        std::vector<double *> Bz(wfc_num);
        std::vector<double *> Bx(wfc_num);
        std::vector<double *> By(wfc_num);
        if (dimnum == 2){
            for (int i = 0; i < wfc_array.size(); ++i){
                Bz[i] = curl2d(par, Ax[i], Ay[i]);
            }
        }
        if (dimnum == 3){
            std::cout << "Calculating the 3d curl..." << '\n';
            for (int i = 0; i < wfc_array.size(); ++i){
                Bx[i] = curl3d_x(par, Ax[i], Ay[i], Az[i]);
                By[i] = curl3d_y(par, Ax[i], Ay[i], Az[i]);
                Bz[i] = curl3d_z(par, Ax[i], Ay[i], Az[i]);
            }
            std::cout << "Finished calculating Curl" << '\n';
        }
        std::cout << "writing initial variables to file..." << '\n';
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
        //hdfWriteDouble(xDim, V, 0, "V_0"); //HDF COMING SOON!
        //hdfWriteComplex(xDim, wfc, 0, "wfc_0");
        if (cyl_coord && dimnum > 2){
            std::vector<double *> Br(wfc_num);
            std::vector<double *> Bphi(wfc_num);

            for (int i = 0; i < wfc_array.size(); ++i){
                Br[i] = curl3d_r(par, Bx[i], By[i]);
                Bphi[i] = curl3d_phi(par, Bx[i], By[i]);

                FileIO::writeOutDouble(data_dir + "Br_" + std::to_string(i),
                                       Br[i],gSize,step_offset);
                FileIO::writeOutDouble(data_dir + "Bphi_" + std::to_string(i),
                                       Bphi[i],gSize,step_offset);
                FileIO::writeOutDouble(data_dir + "Bz_" + std::to_string(i),
                                       Bz[i],gSize,step_offset);

                free(Br[i]);
                free(Bx[i]);
                free(By[i]);
                free(Bz[i]);
                free(Bphi[i]);
            }
        }
        else{
            if (dimnum > 1){
                for (int i = 0; i < wfc_array.size(); ++i){
                    FileIO::writeOutDouble(data_dir + "Bz_" + std::to_string(i),
                                           Bz[i],gSize,step_offset);
                    free(Bz[i]);
                }
            }
            if (dimnum > 2){
                for (int i = 0; i < wfc_array.size(); ++i){
                    FileIO::writeOutDouble(data_dir + "Bx_"+std::to_string(i),
                                           Bx[i],gSize,step_offset);
                    FileIO::writeOutDouble(data_dir + "By_"+std::to_string(i),
                                           By[i],gSize,step_offset);
                    free(Bx[i]);
                    free(By[i]);
                }
            }
        }

        FileIO::writeOutV(par, V, 0);
        FileIO::writeOutK(par, K, 0);

        for (int i = 0; i < wfc_array.size(); ++i){
            FileIO::writeOutDouble(data_dir + "V_"+std::to_string(i),
                                   V[i],gSize,step_offset);
            FileIO::writeOutDouble(data_dir + "K_"+std::to_string(i),
                                   K[i],gSize,step_offset);
            FileIO::writeOutDouble(data_dir+"pAy_"+std::to_string(i),
                                   pAy[i],gSize,step_offset);
            FileIO::writeOutDouble(data_dir+"pAx_"+std::to_string(i),
                                   pAx[i],gSize,step_offset);
            FileIO::writeOutDouble(data_dir + "Ax_"+std::to_string(i),
                                   Ax[i],gSize,step_offset);
            FileIO::writeOutDouble(data_dir + "Ay_"+std::to_string(i),
                                   Ay[i],gSize,step_offset);
            FileIO::writeOutDouble(data_dir + "Az_"+std::to_string(i),
                                   Az[i],gSize,step_offset);
            FileIO::writeOutDouble(data_dir + "x",x,xDim,step_offset);
            FileIO::writeOutDouble(data_dir + "y",y,yDim,step_offset);
            FileIO::writeOutDouble(data_dir + "z",z,zDim,step_offset);
            FileIO::writeOut(data_dir + "EpAz_"+std::to_string(i),
                             EpAz[i],gSize,step_offset);
            FileIO::writeOut(data_dir + "EpAy_"+std::to_string(i),
                             EpAy[i],gSize,step_offset);
            FileIO::writeOut(data_dir + "EpAx_"+std::to_string(i),
                             EpAx[i],gSize,step_offset);
            FileIO::writeOut(data_dir + "GK_"+std::to_string(i),
                             GK[i],gSize,step_offset);
            FileIO::writeOut(data_dir + "GV_"+std::to_string(i),
                             GV[i],gSize,step_offset);
            FileIO::writeOut(data_dir + "GpAx_"+std::to_string(i),
                             GpAx[i],gSize,step_offset);
            FileIO::writeOut(data_dir + "GpAy_"+std::to_string(i),
                             GpAy[i],gSize,step_offset);
            FileIO::writeOut(data_dir + "GpAz_"+std::to_string(i),
                             GpAz[i],gSize,step_offset);
        }
    }

    if (par.bval("read_wfc") == false){
        sum=sqrt(sum*dx*dy*dz);
        for (int i = 0; i < wfc_array.size(); ++i){
            for (int j = 0; j < gSize; j++){
                wfc_array[i][j].x = (wfc_array[i][j].x)/(sum);
                wfc_array[i][j].y = (wfc_array[i][j].y)/(sum);
            }
        }
    }

    cufftHandleError( cufftPlan2d(&plan_2d, xDim, yDim, CUFFT_Z2Z) );

    generate_plan_other3d(&plan_1d, par, 0);
    if (dimnum == 2){
        generate_plan_other2d(&plan_other2d, par);
    }
    if (dimnum == 3){
        generate_plan_other3d(&plan_dim3, par, 2);
        generate_plan_other3d(&plan_dim2, par, 1);
    }
    cufftHandleError( cufftPlan3d(&plan_3d, xDim, yDim, zDim, CUFFT_Z2Z) );

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    //std::cout << GV[0].x << '\t' << GK[0].x << '\t'
    //          << pAy[0] << '\t' << pAx[0] << '\n';

    //std::cout << "storing variables..." << '\n';

    // Storing variables that have been initialized
    // Re-establishing variables from parsed Grid class
    // Initializes uninitialized variables to 0 values
    par.store("Energy", Energy);
    par.store("r", r);
    par.store("Energy_gpu", Energy_gpu);
    par.store("wfc_array", wfc_array);
    par.store("EV_opt", EV_opt);
    par.store("V_opt", V_opt);
    par.store("EappliedField", EappliedField);

    par.store("plan_1d", plan_1d);
    par.store("plan_2d", plan_2d);
    par.store("plan_other2d", plan_other2d);
    par.store("plan_3d", plan_3d);
    par.store("plan_dim2", plan_dim2);
    par.store("plan_dim3", plan_dim3);

    // Parameters for time-depd variables.
    par.store("K_time", false);
    par.store("V_time", false);
    par.store("Ax_time", false);
    par.store("Ay_time", false);
    par.store("Az_time", false);

    std::cout << "variables stored" << '\n';

    return 0;
}

void set_variables(Grid &par, bool ev_type){
    // Re-establishing variables from parsed Grid class
    // Note that 3d variables are set to nullptr's unless needed
    //      This might need to be fixed later
    double dx = par.dval("dx");
    double dy = par.dval("dy");
    int wfc_num = par.ival("wfc_num");
    std::vector<double *> V_opt = par.dsvecval("V_opt");
    std::vector<double *> pAy = par.dsvecval("pAy");
    std::vector<double *> pAx = par.dsvecval("pAx");
    std::vector<double2 *> pAy_gpu(wfc_num);
    std::vector<double2 *> pAx_gpu(wfc_num);
    std::vector<double2 *> pAz_gpu(wfc_num);
    std::vector<double2 *> V_gpu(wfc_num);
    std::vector<double2 *> K_gpu(wfc_num);
    std::vector<double2 *> wfc_array = par.d2svecval("wfc_array");
    std::vector<double2 *> wfc_gpu_array =
         par.d2svecval("wfc_gpu_array");
    int dimnum = par.ival("dimnum");
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    int zDim = par.ival("zDim");
    int gsize = xDim;

    // Special variables for the 3d case
    if (dimnum > 1){
        gsize *= yDim;
    }
    if (dimnum > 2){
        gsize *= zDim;
    }
    if(!par.bval("V_time")){
        for (int i = 0; i < wfc_array.size(); ++i){
            cudaHandleError(cudaMalloc((void**) &V_gpu[i],
                           sizeof(double2)*gsize*wfc_num));
        }
    }
    if(!par.bval("K_time")){
        for (int i = 0; i < wfc_array.size(); ++i){
            cudaHandleError(cudaMalloc((void**) &K_gpu[i],
                            sizeof(double2)*gsize*wfc_num));
        }
    }
    if(!par.bval("Ax_time")){
        for (int i = 0; i < wfc_array.size(); ++i){
            cudaHandleError(cudaMalloc((void**) &pAx_gpu[i],
                            sizeof(double2)*gsize*wfc_num));
        }
    }
    if(!par.bval("Ay_time") && dimnum > 1){
        for (int i = 0; i < wfc_array.size(); ++i){
            cudaHandleError(cudaMalloc((void**) &pAy_gpu[i],
                            sizeof(double2)*gsize*wfc_num));
        }
    }
    if(!par.bval("Az_time") && dimnum > 2){
        for (int i = 0; i < wfc_array.size(); ++i){
            cudaHandleError(cudaMalloc((void**) &pAz_gpu[i],
                            sizeof(double2)*gsize*wfc_num));
        }
    }

    if (ev_type == 0){
        std::vector<double2 *> GK = par.d2svecval("GK");
        std::vector<double2 *> GV = par.d2svecval("GV");
        std::vector<double2 *> GpAx = par.d2svecval("GpAx");
        std::vector<double2 *> GpAy(wfc_num);
        std::vector<double2 *> GpAz(wfc_num);

        if(!par.bval("K_time")){
            for (int i = 0; i < wfc_array.size(); ++i){
                cudaHandleError(cudaMemcpy(K_gpu[i], GK[i],
                                sizeof(cufftDoubleComplex)*gsize,
                                cudaMemcpyHostToDevice));
            }
        }
        if(!par.bval("V_time")){
            for (int i = 0; i < wfc_array.size(); ++i){
                cudaHandleError(cudaMemcpy(V_gpu[i], GV[i],
                                sizeof(cufftDoubleComplex)*gsize,
                                cudaMemcpyHostToDevice));
            }
        }
        if(!par.bval("Ax_time")){
            for (int i = 0; i < wfc_array.size(); ++i){
                cudaHandleError(cudaMemcpy(pAx_gpu[i], GpAx[i],
                                sizeof(cufftDoubleComplex)*gsize,
                                cudaMemcpyHostToDevice));
            }
        }
        for (int i = 0; i < wfc_array.size(); ++i){
            cudaHandleError(cudaMemcpy(wfc_gpu_array[i], wfc_array[i],
                            sizeof(cufftDoubleComplex)*gsize,
                            cudaMemcpyHostToDevice));
        }
        par.store("K_gpu", K_gpu);
        par.store("V_gpu", V_gpu);
        par.store("wfc_gpu_array", wfc_gpu_array);
        par.store("pAx_gpu", pAx_gpu);

        // Special cases for 3d
        if (dimnum > 1 && !par.bval("Ay_time")){
            GpAy = par.d2svecval("GpAy");
            for (int i = 0; i < wfc_array.size(); ++i){
                cudaHandleError(cudaMemcpy(pAy_gpu[i], GpAy[i],
                                sizeof(cufftDoubleComplex)*gsize,
                                cudaMemcpyHostToDevice));
            }
            par.store("pAy_gpu", pAy_gpu);

        }
        if (dimnum > 2 && !par.bval("Az_time")){
            GpAz = par.d2svecval("GpAz");
            for (int i = 0; i < wfc_array.size(); ++i){
                cudaHandleError(cudaMemcpy(pAz_gpu[i], GpAz[i],
                                sizeof(cufftDoubleComplex)*gsize,
                                cudaMemcpyHostToDevice));
            }
            par.store("pAz_gpu", pAz_gpu);

        }
        for (int i = 0; i < wfc_array.size(); ++i){
            free(GV[i]); free(GK[i]); free(GpAy[i]);
            free(GpAx[i]); free(GpAz[i]);
        }
    }
    else if (ev_type == 1){

        std::vector<double2 *> EV = par.d2svecval("EV");
        std::vector<double2 *> EK = par.d2svecval("EK");
        std::vector<double2 *> EpAx = par.d2svecval("EpAx");
        std::vector<double2 *> EpAy(wfc_num);
        std::vector<double2 *> EpAz(wfc_num);
        if (!par.bval("K_time")){
            for (int i = 0; i < wfc_array.size(); ++i){
                cudaHandleError(cudaMemcpy(K_gpu[i], EK[i],
                                sizeof(cufftDoubleComplex)*gsize,
                                cudaMemcpyHostToDevice));
                par.store("K_gpu", K_gpu);
            }
        }
        if(!par.bval("Ax_time")){
            for (int i = 0; i < wfc_array.size(); ++i){
                cudaHandleError(cudaMemcpy(pAx_gpu[i], EpAx[i],
                               sizeof(cufftDoubleComplex)*gsize,
                               cudaMemcpyHostToDevice));
                par.store("pAx_gpu", pAx_gpu);
            }
        }

        if (!par.bval("V_time")){
            for (int i = 0; i < wfc_array.size(); ++i){
                cudaHandleError(cudaMemcpy(V_gpu[i], EV[i],
                                sizeof(cufftDoubleComplex)*gsize,
                                cudaMemcpyHostToDevice));
                par.store("V_gpu", V_gpu);
            }
        }
        for (int i = 0; i < wfc_array.size(); ++i){
            cudaHandleError(cudaMemcpy(wfc_gpu_array[i], wfc_array[i],
                            sizeof(cufftDoubleComplex)*gsize,
                            cudaMemcpyHostToDevice));
        }

        par.store("wfc_gpu_array", wfc_gpu_array);

        // Special variables / instructions for 2/3d case
        if (dimnum > 1 && !par.bval("Ay_time")){
            EpAy = par.d2svecval("EpAy");
            for (int i = 0; i < wfc_array.size(); ++i){
                cudaHandleError(cudaMemcpy(pAy_gpu[i], EpAy[i],
                                sizeof(cufftDoubleComplex)*gsize,
                                cudaMemcpyHostToDevice));
                par.store("pAy_gpu", pAy_gpu);
            }
        }

        if (dimnum > 2 && !par.bval("Az_time")){
            EpAz = par.d2svecval("EpAz");
            for (int i = 0; i < wfc_array.size(); ++i){
                cudaHandleError(cudaMemcpy(pAz_gpu[i], EpAz[i],
                                sizeof(cufftDoubleComplex)*gsize,
                                cudaMemcpyHostToDevice));
                par.store("pAz_gpu", pAz_gpu);
            }
        }

        for (int i = 0; i < wfc_array.size(); ++i){
            free(EV[i]);
            free(EK[i]);
            free(EpAy[i]);
            free(EpAx[i]);
            free(EpAz[i]);
        }
    }

}
