#include "../include/evolution.h"

void evolve_2d(Wave &wave, Op &opr,
               cufftDoubleComplex *gpuParSum, int numSteps, Cuda &cupar,
               unsigned int gstate, Grid &par,
               std::string buffer){

    // Re-establishing variables from parsed Grid class
    std::string data_dir = par.sval("data_dir");
    double omega = par.dval("omega");
    double angle_sweep = par.dval("angle_sweep");
    double gdt = par.dval("gdt");
    double dt = par.dval("dt");
    double omegaX = par.dval("omegaX");
    double omegaY = par.dval("omegaY");
    double omegaZ = par.dval("omegaZ");
    double mass = par.dval("mass");
    double dx = par.dval("dx");
    double dy = par.dval("dy");
    double interaction = par.dval("interaction");
    double laser_power = par.dval("laser_power");
    double gDenConst = par.dval("gDenConst");
    double DX = par.dval("DX");
    double mask_2d = par.dval("mask_2d");
    double *x = par.dsval("x");
    double *y = par.dsval("y");
    double *V = opr.dsval("V");
    double *V_opt = opr.dsval("V_opt");
    double *Phi = wave.dsval("Phi");
    double *gpu1dpAx = opr.dsval("pAx_gpu");
    double *gpu1dpAy = opr.dsval("pAy_gpu");
    double *Phi_gpu = wave.dsval("Phi_gpu");
    int kick_it = par.ival("kick_it");
    bool write_it = par.bval("write_it");
    bool graph = par.bval("graph");
    int N = par.ival("atoms");
    int printSteps = par.ival("printSteps");
    bool nonlin = par.bval("gpe");
    bool lz = par.bval("corotating");
    bool ramp = par.bval("ramp");
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    int gridSize = xDim * yDim;
    int kill_idx = par.ival("kill_idx");
    int charge = par.ival("charge");
    double x0_shift = par.dval("x0_shift");
    double y0_shift = par.dval("y0_shift");
    cufftDoubleComplex *EV = opr.cufftDoubleComplexval("EV");
    cufftDoubleComplex *wfc = wave.cufftDoubleComplexval("wfc");
    cufftDoubleComplex *EV_opt = opr.cufftDoubleComplexval("EV_opt");
    cufftDoubleComplex *gpuWfc = wave.cufftDoubleComplexval("wfc_gpu");
    cufftDoubleComplex *K_gpu =
        opr.cufftDoubleComplexval("K_gpu");
    cufftDoubleComplex *V_gpu =
        opr.cufftDoubleComplexval("V_gpu");

    std::cout << x[0] << '\t' << EV[0].x << '\t' << wfc[0].x << '\t'
              << EV_opt[0].x << '\t' << '\n';

    // getting data from Cuda class
    cufftResult result = cupar.cufftResultval("result");
    cufftHandle plan_1d = cupar.cufftHandleval("plan_1d");
    cufftHandle plan_2d = cupar.cufftHandleval("plan_2d");
    cufftHandle plan_other2d = cupar.cufftHandleval("plan_other2d");

    dim3 threads = cupar.dim3val("threads");
    dim3 grid = cupar.dim3val("grid");

    // Because no two operations are created equally.
    // Multiplication is faster than divisions.
    double renorm_factor_2d=1.0/pow(gridSize,0.5);
    double renorm_factor_1d=1.0/pow(xDim,0.5);

    // outputting a bunch of variables just to check thigs out...
    std::cout << omega << '\t' << angle_sweep << '\t' << gdt << '\t'
              << dt << '\t' << omegaX << '\t' << omegaY << '\t'
              << mass << '\t' << dx << '\t' << dy << '\t' << interaction << '\t'
              << laser_power << '\t' << N << '\t' << xDim << '\t'
              << yDim << '\n';


    clock_t begin, end;
    double time_spent;
    double Dt;
    if(gstate==0){
        Dt = gdt;
        printf("Timestep for groundstate solver set as: %E\n",Dt);
    }
    else{
        Dt = dt;
        printf("Timestep for evolution set as: %E\n",Dt);
    }
    begin = clock();
    double omega_0=omega*omegaX;

    // ** ############################################################## ** //
    // **         HERE BE DRAGONS OF THE MOST DANGEROUS KIND!            ** //
    // ** ############################################################## ** //

    // Double buffering and will attempt to thread free and calloc operations to
    // hide time penalty. Or may not bother.
    int num_vortices[2] = {0,0};

    // binary matrix of size xDim*yDim,
    // 1 for vortex at specified index, 0 otherwise
    int* vortexLocation;
    //int* olMaxLocation = (int*) calloc(xDim*yDim,sizeof(int));

    std::shared_ptr<Vtx::Vortex> central_vortex; //vortex closest to the central position
    /*
    central_vortex.coords.x = -1;
    central_vortex.coords.y = -1;
    central_vortex.coordsD.x = -1.;
    central_vortex.coordsD.y = -1.;
    central_vortex.wind = 0;
    */

    // Angle of vortex lattice. Add to optical lattice for alignment.
    double vort_angle;

    // array of vortex coordinates from vortexLocation 1's
    //struct Vtx::Vortex *vortCoords = NULL;


    std::shared_ptr<Vtx::VtxList> vortCoords = std::make_shared<Vtx::VtxList>(7);
    //std::vector<std::shared_ptr<Vtx::Vortex> vortCoords;

    //Previous array of vortex coordinates from vortexLocation 1's
    //struct Vtx::Vortex *vortCoordsP = NULL;
    //std::vector<struct Vtx::Vortex> vortCoordsP;
    std::shared_ptr<Vtx::VtxList> vortCoordsP = std::make_shared<Vtx::VtxList>(7);


    LatticeGraph::Lattice lattice; //Vortex lattice graph.
    double* adjMat;

    double vortOLSigma=0.0;
    double sepAvg = 0.0;

    int num_kick = 0;
    // Assuming triangular lattice at rotation of omega_0
    double t_kick = (2*PI/omega_0)/(6*Dt); 

    //std::cout << "numSteps is: " << numSteps << '\n';
    // Iterating through all of the steps in either g or esteps.
    for(int i=0; i < numSteps; ++i){
        if ( ramp ){
            //Adjusts omega for the appropriate trap frequency.
            omega_0=omegaX*((omega-0.39)*((double)i/(double)(numSteps)) + 0.39);
            // assuming critical rot. freq. of 0.39\omega_\perp to seed a vortex
        }

        // Print-out at pre-determined rate.
        // Vortex & wfc analysis performed here also.
        if(i % printSteps == 0) {
            // If the unit_test flag is on, we need a special case
            printf("Step: %d    Omega: %lf\n", i, omega_0 / omegaX);
            cudaMemcpy(wfc, gpuWfc, sizeof(cufftDoubleComplex) * xDim * yDim,
                       cudaMemcpyDeviceToHost);

            // Printing out time of iteration
            end = clock();
            time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
            printf("Time spent: %lf\n", time_spent);
            std::string fileName = "";
            printf("ramp=%d        gstate=%d    rg=%d        \n",
                   ramp, gstate, ramp | (gstate << 1));
            switch (ramp | (gstate << 1)) {
                case 0: //Groundstate solver, constant Omega value.
                    fileName = "wfc_0_const";
                    break;
                case 1: //Groundstate solver, ramped Omega value.
                    fileName = "wfc_0_ramp";
                    break;
                case 2: //Real-time evolution, constant Omega value.
                    fileName = "wfc_ev";
                    vortexLocation = (int *) calloc(xDim * yDim, sizeof(int));
                    num_vortices[0] = Tracker::findVortex(vortexLocation, wfc,
                                                          mask_2d, xDim, x, i);
                    std::cout << "Found " << num_vortices[0] << " vortices" << std::endl;
                    // If initial step, locate vortices, least-squares to find
                    // exact centre, calculate lattice angle, generate optical
                    // lattice.
                    if (i == 0) {
                         if(num_vortices[0] > 0){
                         //Reserve enough space for the vortices
                         //reserve(num_vortices[0]);
                         vortCoords = std::make_shared<Vtx::VtxList>
                                          (num_vortices[0]);
                         vortCoordsP = std::make_shared<Vtx::VtxList>
                                           (num_vortices[0]);

                        //Locate the vortex positions to the nearest grid, then                         //perform a least-squares fit to determine the location                         //to sub-grid reolution.
                         Tracker::vortPos(vortexLocation,
                                 vortCoords->getVortices(), xDim, wfc);
                         Tracker::lsFit(vortCoords->getVortices(),wfc,xDim);

                        //Find the centre-most vortex in the lattice
                         central_vortex = Tracker::vortCentre(vortCoords->
                                 getVortices(), xDim);
                        //Determine the Angle formed by the lattice relative to                         //the x-axis
                         vort_angle = Tracker::vortAngle(vortCoords->
                                 getVortices(), central_vortex);

                        //Store the vortex angle in the parameter file
                         par.store("Vort_angle", vort_angle);
                             
                        //Determine average lattice spacing.
                         sepAvg = Tracker::vortSepAvg(vortCoords->
                                 getVortices(), central_vortex);

                         par.store("Central_vort_x",
                                  (double) central_vortex->getCoords().x);
                         par.store("Central_vort_y",
                                  (double) central_vortex->getCoords().y);
                         par.store("Central_vort_winding",
                                  (double) central_vortex->getWinding());
                         par.store("Num_vort", (double) vortCoords->
                                 getVortices().size());

                        //Setup the optical lattice to match the spacing and
                        // angle+angle_sweep of the vortex lattice.
                        // Amplitude matched by setting laser_power
                        // parameter switch.
                         optLatSetup(central_vortex, V, 
                                    vortCoords->getVortices(),
                                    vort_angle + PI * angle_sweep / 180.0,
                                    laser_power * HBAR * sqrt(omegaX * omegaY),
                                    V_opt, x, y, par, opr);


			}
                        // If kick_it param is 2, perform a single kick of the 
                        // optical lattice for the first timestep only.
                        // This is performed by loading the
                        // EV_opt exp(V + V_opt) array into GPU memory
                        // for the potential.
                        if (kick_it == 2) {
                            printf("Kicked it 1\n");
                            cudaMemcpy(V_gpu, EV_opt,
                                       sizeof(cufftDoubleComplex) * xDim * yDim,
                                       cudaMemcpyHostToDevice);
                        }
                        // Write out the newly specified potential
                        // and exp potential to files
                        FileIO::writeOutDouble(buffer, data_dir + "V_opt_1",
                                               V_opt, xDim * yDim, 0);
                        FileIO::writeOut(buffer, data_dir + "EV_opt_1", EV_opt,
                                         xDim * yDim, 0);

                        //Store necessary parameters to Params.dat file.
                        FileIO::writeOutParam(buffer, par,
                                              data_dir + "Params.dat");
                    }
                    //If i!=0 and the number of vortices changes
/*
                    else if (num_vortices[0] > num_vortices[1]) {
                        printf("Number of vortices increased from %d to %d\n",
                               num_vortices[1], num_vortices[0]);
                        Tracker::vortPos(vortexLocation, vortCoords.data(), 
                                         xDim, wfc);
                        Tracker::lsFit(vortCoords.data(), wfc, num_vortices[0],
                                       xDim);
                    }
*/
                    // if num_vortices[1] < num_vortices[0] ... Fewer vortices
                    else {
                         if (num_vortices[0] > 0){
                        	Tracker::vortPos(vortexLocation, 
                                    vortCoords->getVortices(), xDim, wfc);
                        	Tracker::lsFit(vortCoords->getVortices(), 
                                               wfc, xDim);
                        	Tracker::vortArrange(vortCoords->getVortices(),
                                                    vortCoordsP->getVortices());
                    		//FileIO::writeOutInt(buffer, data_dir + "vLoc_",
                            //                   vortexLocation, xDim * yDim, i);
                         }
                    }

                    // The following will eventually be modified and moved into
                    // a new library that works closely wy0_shiftUE. Used to 
                    // also defined for vortex elimination using graph positions                    // and UID numbers.
                    if (graph && num_vortices[0] > 0) {
                        for (int ii = 0; ii < vortCoords->getVortices().size();                              ++ii) {
                            std::shared_ptr<LatticeGraph::Node>
                                n(new LatticeGraph::Node(
                                    *vortCoords->getVortices().at(ii).get()));
                            lattice.addVortex(std::move(n));
                        }
                        unsigned int *uids = (unsigned int *) malloc(
                                sizeof(unsigned int) *
                                lattice.getVortices().size());
                        for (size_t a=0; a < lattice.getVortices().size(); ++a){
                            uids[a] = lattice.getVortexIdx(a)->getUid();
                        }
                        if(i==0) {
                            //Lambda for vortex annihilation/creation.
                            auto killIt=[&](int idx, int winding, 
                                            double delta_x, double delta_y) {
                                //if (abs(delta_x) > 0 || abs(delta_y) > 0){
                                    // Killing initial vortex and then 
                                    // imprinting new one
                                //Remove pre-existing vortex
                                int imprint = -lattice.getVortexUid(idx)->getData().getWinding();
                                std::cout << "Annihilating Vortex " << idx << " at position (" \
                                        << lattice.getVortexUid(idx)->getData().getCoordsD().x \
                                        << "," 
                                        << lattice.getVortexUid(idx)->getData().getCoordsD().y \
                                        << ")." << std::endl;

                                if(winding>0){ //Shift vortex position; remove and add at new x,y
                                    WFC::phaseWinding(Phi, imprint, x, y, dx, dy,	
                                        lattice.getVortexUid(idx)->getData().getCoordsD().x,
                                        lattice.getVortexUid(idx)->getData().getCoordsD().y,
                                        xDim);
                                    WFC::phaseWindingSum(Phi, -imprint, x, y, dx, dy,
                                        lattice.getVortexUid(idx)->getData().getCoordsD().x + delta_x,
                                        lattice.getVortexUid(idx)->getData().getCoordsD().y + delta_y,
                                        xDim);

                                }
                                else if(winding<0){//Flip vortex; imprint charge q+1 at x,y (let 1 annihilate with nearby core)
                                    WFC::phaseWindingSum(Phi, -2*imprint, x, y, dx, dy,
                                        lattice.getVortexUid(idx)->getData().getCoordsD().x + delta_x,
                                        lattice.getVortexUid(idx)->getData().getCoordsD().y + delta_y,
                                        xDim);
                                }
                                else {
                                }

                               cudaMemcpy(Phi_gpu, Phi, sizeof(double) * xDim * yDim, cudaMemcpyHostToDevice);
                               cMultPhi <<<grid, threads>>> ( gpuWfc, Phi_gpu, gpuWfc );

                               std::string wfc_phase_imprint = "wfc_phase_imp";
                               FileIO::writeOut(buffer, data_dir + wfc_phase_imprint, wfc, xDim*yDim, i);
                               FileIO::writeOutDouble(buffer, "PHI_IMP", Phi, xDim*yDim, i);
                            };
                            if (kill_idx > 0){
                                killIt(kill_idx, charge, x0_shift, y0_shift);
                            }
                        }
                        lattice.createEdges(1.5 * 2e-5 / dx);

                        //Assumes that vortices
                        //only form edges when the distance is upto 1.5*2e-5.
                        //Replace with delaunay triangulation determined edges 
                        //for better computational scaling (and sanity)

                        //O(n^2) -> terrible implementation. It works for now.
                        //Generates the adjacency matrix from the graph, and
                        //outputs to a Mathematica compatible format.
                        adjMat = (double *)calloc(lattice.getVortices().size() *
                                                  lattice.getVortices().size(),
                                                   sizeof(double));
                        lattice.genAdjMat(adjMat);
                        FileIO::writeOutAdjMat(buffer, data_dir + "graph",
                                               adjMat, uids,
                                               lattice.getVortices().size(), i);

                        //Free and clear all memory blocks
                        free(adjMat);
                        free(uids);
                        lattice.getVortices().clear();
                        lattice.getEdges().clear();
                        //exit(0);
                    }

                    //Write out the vortex locations
                    FileIO::writeOutVortex(buffer, data_dir + "vort_arr",
                                           vortCoords->getVortices(), i);
                    printf("Located %d vortices\n", 
                           vortCoords->getVortices().size());

                    //Free memory block for now.
                    free(vortexLocation);

                    //Current values become previous values.
                    num_vortices[1] = num_vortices[0];
                    vortCoords->getVortices().swap(vortCoordsP->getVortices());
		            vortCoords->getVortices().clear();
			        std::cout << "Vortex tracking complete for timestep " << i << std::endl;

                    break;

                case 3:
                    fileName = "wfc_ev_ramp";
                    break;
                default:
                    break;
            }

            //std::cout << "writing" << '\n';
            if (write_it) {
                FileIO::writeOut(buffer, data_dir + fileName,
                                 wfc, xDim * yDim, i);
            }
            //std::cout << "written" << '\n';
            if (par.bval("energy_calc")){
                printf("Energy[t@%d]=%E\n",i,energy_angmom(V_gpu, 
                       K_gpu, gpuWfc, gstate, par, cupar));
            }
        }


        // ** ########################################################## ** //
        // **                     More F'n' Dragons!                     ** //
        // ** ########################################################## ** //

        // If not already kicked at this time step more than 6 times... kick it!
        if(i%((int)t_kick+1) == 0 && num_kick<=6 && gstate==1 && kick_it == 1 ){
            cudaMemcpy(V_gpu, EV_opt, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
            ++num_kick;
        }
        // ** ########################################################## ** //

        // U_r(dt/2)*wfc
        if(nonlin == 1){
            //std::cout << Dt << '\t' << mass << '\t' << omegaZ << '\t'
            //          << gstate << '\t' << N*interaction << '\n';
            cMultDensity<<<grid,threads>>>(V_gpu,gpuWfc,gpuWfc,0.5*Dt,
                                           mass,gstate,interaction*gDenConst);
        }
        else {
            cMult<<<grid,threads>>>(V_gpu,gpuWfc,gpuWfc);
        }

        // U_p(dt)*fft2(wfc)
        result = cufftExecZ2Z(plan_2d,gpuWfc,gpuWfc,CUFFT_FORWARD);

        // Normalise
        scalarMult<<<grid,threads>>>(gpuWfc,renorm_factor_2d,gpuWfc);
        cMult<<<grid,threads>>>(K_gpu,gpuWfc,gpuWfc);
        result = cufftExecZ2Z(plan_2d,gpuWfc,gpuWfc,CUFFT_INVERSE);

        // Normalise
        scalarMult<<<grid,threads>>>(gpuWfc,renorm_factor_2d,gpuWfc);

        // U_r(dt/2)*wfc
        if(nonlin == 1){
            cMultDensity<<<grid,threads>>>(V_gpu,gpuWfc,gpuWfc,Dt*0.5,
                                           mass,gstate,interaction*gDenConst);
        }
        else {
            cMult<<<grid,threads>>>(V_gpu,gpuWfc,gpuWfc);
        }

        // If first timestep and kick_it >= 1, kick.
        // Also kick if not kicked enough
        if( (i % (int)(t_kick+1) == 0 && num_kick<=6 && gstate==1) ||
            (kick_it >= 1 && i==0) ){
            cudaMemcpy(V_gpu, EV, sizeof(cufftDoubleComplex)*xDim*yDim,
                       cudaMemcpyHostToDevice);
            printf("Got here: Cuda memcpy EV into GPU\n");
        }
        // Angular momentum pAy-pAx (if engaged)  //
        if(lz == 1){
            // Multiplying by ramping factor if necessary
            // Note: using scalarPow to do the scaling inside of the exp
            if (ramp ){
                scalarPow<<<grid,threads>>>((cufftDoubleComplex*) gpu1dpAy,
                                            omega_0/(omega * omegaY),
                                            (cufftDoubleComplex*) gpu1dpAy);
                scalarPow<<<grid,threads>>>((cufftDoubleComplex*) gpu1dpAx,
                                            omega_0/(omega * omegaX),
                                            (cufftDoubleComplex*) gpu1dpAx);
            }
            switch(i%2 | (gstate<<1)){
                case 0: //Groundstate solver, even step

                    // 1d forward / mult by Ay
                    result = cufftExecZ2Z(plan_1d,gpuWfc,gpuWfc,CUFFT_FORWARD);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAy, gpuWfc);
                    result = cufftExecZ2Z(plan_1d,gpuWfc,gpuWfc,CUFFT_INVERSE);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d, gpuWfc);


                    // 1D FFT to wfc_pAx
                    result = cufftExecZ2Z(plan_other2d,gpuWfc,gpuWfc,
                                          CUFFT_FORWARD);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAx, gpuWfc);

                    result = cufftExecZ2Z(plan_other2d,gpuWfc,gpuWfc,
                                          CUFFT_INVERSE);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d, gpuWfc);
                    break;

                case 1:    //Groundstate solver, odd step
                    // 1D FFT to wfc_pAx
                    result = cufftExecZ2Z(plan_other2d,gpuWfc,gpuWfc,
                                          CUFFT_FORWARD);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAx, gpuWfc);

                    result = cufftExecZ2Z(plan_other2d,gpuWfc,gpuWfc,
                                          CUFFT_INVERSE);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d, gpuWfc);

                    // wfc_pAy
                    result = cufftExecZ2Z(plan_1d,gpuWfc,gpuWfc,CUFFT_FORWARD);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAy, gpuWfc);
                    result = cufftExecZ2Z(plan_1d,gpuWfc,gpuWfc,CUFFT_INVERSE);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d, gpuWfc);
                    break;

                case 2: //Real time evolution, even step
                    //std::cout << "RT solver even." << '\n';

                    // wfc_pAy
                    result = cufftExecZ2Z(plan_1d,gpuWfc,gpuWfc,CUFFT_FORWARD);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAy, gpuWfc);
                    result = cufftExecZ2Z(plan_1d,gpuWfc,gpuWfc,CUFFT_INVERSE);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);

                    // 1D to wfc_pAx
                    result = cufftExecZ2Z(plan_other2d,gpuWfc,gpuWfc,
                                          CUFFT_FORWARD);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAx, gpuWfc);

                    // wfc_pAy
                    result = cufftExecZ2Z(plan_other2d,gpuWfc,gpuWfc,
                                          CUFFT_INVERSE);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);

                    break;

                case 3:    //Real time evolution, odd step
                    //std::cout << "RT solver odd." << '\n';

                    // 1D inverse to wfc_pAx
                    result = cufftExecZ2Z(plan_other2d,gpuWfc,gpuWfc,
                                          CUFFT_FORWARD);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAx, gpuWfc);

                    // wfc_pAy
                    result = cufftExecZ2Z(plan_other2d,gpuWfc,gpuWfc,
                                          CUFFT_INVERSE);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);

                    // wfc_pAy
                    result = cufftExecZ2Z(plan_1d,gpuWfc,gpuWfc,CUFFT_FORWARD);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAy, gpuWfc);
                    result = cufftExecZ2Z(plan_1d,gpuWfc,gpuWfc,CUFFT_INVERSE);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    break;

            }
        }

        if(gstate==0){
            parSum(gpuWfc, gpuParSum, par, cupar);
        }
    }

    // std::cout << "finished evolution" << '\n';
    // Storing wavefunctions for later
    //std::cout << gpuWfc[0].x << '\t' << gpuWfc[0].y << '\n';
    wave.store("wfc", wfc);
    wave.store("wfc_gpu", gpuWfc);

/*
    par.store("omega", omega);
    par.store("angle_sweep", angle_sweep);
    par.store("gdt", gdt);
    par.store("dt", dt);
    par.store("omegaX", omegaX);
    par.store("omegaY", omegaY);
    par.store("omegaZ", omegaZ);
    par.store("mass", mass);
    par.store("dx", dx);
    par.store("dy", dy);
    par.store("interaction", interaction);
    par.store("laser_power", laser_power);
    par.store("x", x);
    par.store("y", y);
    opr.store("V", V);
    opr.store("V_opt", V_opt);
    wave.store("Phi", Phi);
    opr.store("pAx_gpu", gpu1dpAx);
    opr.store("pAy_gpu", gpu1dpAy);
    wave.store("Phi_gpu", Phi_gpu);
    opr.store("EV", EV);
    //opr.store("V_gpu", V_gpu);
    //opr.store("K_gpu", K_gpu);
    opr.store("EV_opt", EV_opt);

    // getting data from Cuda class
    cupar.store("result", result);
    cupar.store("plan_1d", plan_1d);
    cupar.store("plan_2d", plan_2d);
    cupar.store("grid", grid);
*/

}

/*----------------------------------------------------------------------------//
* 3D
* Notes: In this case, we need to think about how to do the vortex tracking
*        Kicking will also be hard to do... Though not impossible, I suppose.
*-----------------------------------------------------------------------------*/

void evolve_3d(Wave &wave, Op &opr,
               cufftDoubleComplex *gpuParSum, int numSteps, Cuda &cupar,
               unsigned int gstate, Grid &par,
               std::string buffer){

    // Re-establishing variables from parsed Grid class
    std::string data_dir = par.sval("data_dir");
    double omega = par.dval("omega");
    double angle_sweep = par.dval("angle_sweep");
    double gdt = par.dval("gdt");
    double dt = par.dval("dt");
    double omegaX = par.dval("omegaX");
    double omegaY = par.dval("omegaY");
    double omegaZ = par.dval("omegaZ");
    double mass = par.dval("mass");
    double dx = par.dval("dx");
    double dy = par.dval("dy");
    double dz = par.dval("dz");
    double interaction = par.dval("interaction");
    double laser_power = par.dval("laser_power");
    double gDenConst = par.dval("gDenConst");
    double *x = par.dsval("x");
    double *y = par.dsval("y");
    double *z = par.dsval("z");
    double *V = opr.dsval("V");
    double *V_opt = opr.dsval("V_opt");
    double *Phi = wave.dsval("Phi");
    double *gpu1dpAx = opr.dsval("pAx_gpu");
    double *gpu1dpAy = opr.dsval("pAy_gpu");
    double *gpu1dpAz = opr.dsval("pAz_gpu");
    double *Phi_gpu = wave.dsval("Phi_gpu");
    bool write_it = par.bval("write_it");
    bool graph = par.bval("graph");
    int N = par.ival("atoms");
    int printSteps = par.ival("printSteps");
    bool nonlin = par.bval("gpe");
    bool lz = par.bval("corotating");
    std::cout << "COROTATING IS: " << lz << '\n';
    bool ramp = par.bval("ramp");
    int ramp_type = par.ival("ramp_type");
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    int zDim = par.ival("zDim");
    int gridSize = xDim * yDim * zDim;
    cufftDoubleComplex *EV = opr.cufftDoubleComplexval("EV");
    cufftDoubleComplex *wfc = wave.cufftDoubleComplexval("wfc");
    cufftDoubleComplex *EV_opt = opr.cufftDoubleComplexval("EV_opt");
    cufftDoubleComplex *gpuWfc = wave.cufftDoubleComplexval("wfc_gpu");
    cufftDoubleComplex *K_gpu =
        opr.cufftDoubleComplexval("K_gpu");
    cufftDoubleComplex *V_gpu =
        opr.cufftDoubleComplexval("V_gpu");

    std::cout << x[0] << '\t' << EV[0].x << '\t' << wfc[0].x << '\t'
              << EV_opt[0].x << '\t' << '\n';

    // getting data from Cuda class
    cufftResult result = cupar.cufftResultval("result");
    cufftHandle plan_1d = cupar.cufftHandleval("plan_1d");
    cufftHandle plan_3d = cupar.cufftHandleval("plan_3d");
    cufftHandle plan_dim2 = cupar.cufftHandleval("plan_dim2");
    cufftHandle plan_dim3 = cupar.cufftHandleval("plan_dim3");
    dim3 threads = cupar.dim3val("threads");
    dim3 grid = cupar.dim3val("grid");

    // Because no two operations are created equally.
    // Multiplication is faster than divisions.
    double renorm_factor_3d=1.0/pow(gridSize,0.5);
    double renorm_factor_1d=1.0/pow(xDim,0.5);

    // outputting a bunch of variables just to check thigs out...
    std::cout << omega << '\t' << angle_sweep << '\t' << gdt << '\t'
              << dt << '\t' << omegaX << '\t' << omegaY << '\t'
              << mass << '\t' << dx << '\t' << dy << '\t' << interaction << '\t'
              << laser_power << '\t' << N << '\t' << xDim << '\t'
              << yDim << '\n';


    clock_t begin, end;
    double time_spent;
    double Dt;
    if(gstate==0){
        Dt = gdt;
        printf("Timestep for groundstate solver set as: %E\n",Dt);
    }
    else{
        Dt = dt;
        printf("Timestep for evolution set as: %E\n",Dt);
    }
    begin = clock();
    double omega_0=omega*omegaX;

    // ** ############################################################## ** //
    // **         HERE BE DRAGONS OF THE MOST DANGEROUS KIND!            ** //
    // ** ############################################################## ** //

    //std::cout << "numSteps is: " << numSteps << '\n';
    // Iterating through all of the steps in either g or esteps.
    for(int i=0; i < numSteps; ++i){
        if (ramp){

            //Adjusts omega for the appropriate trap frequency.
            if (ramp_type == 1){
                if (i == 0){
                    omega_0 = (double)omega;
                }
                else{
                    omega_0 = (double)i / (double)(i+1);
                }
            }
            else{
                if (i == 0){
                    omega_0=(double)omega/(double)(numSteps);
                }
                else{
                    omega_0 = (double)(i+1) / (double)i;
                }
            }
        }

        // Print-out at pre-determined rate.
        // Vortex & wfc analysis performed here also.
        if(i % printSteps == 0) {
            // If the unit_test flag is on, we need a special case
            printf("Step: %d    Omega: %lf\n", i, omega_0);
            cudaMemcpy(wfc, gpuWfc, sizeof(cufftDoubleComplex)*xDim*yDim*zDim, 
                       cudaMemcpyDeviceToHost);

            // Printing out time of iteration
            end = clock();
            time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
            printf("Time spent: %lf\n", time_spent);
            std::string fileName = "";
            printf("ramp=%d        gstate=%d    rg=%d        \n",
                   ramp, gstate, ramp | (gstate << 1));
            switch (ramp | (gstate << 1)) {
                case 0: //Groundstate solver, constant Omega value.
                {
                    fileName = "wfc_0_const";
                    break;
                }
                case 1: //Groundstate solver, ramped Omega value.
                {
                    fileName = "wfc_0_ramp";
                    break;
                }
                case 2: //Real-time evolution, constant Omega value.
                {
                    // Note: In the case of 3d, we need to think about
                    //       vortex tracking in a new way.
                    //       It may be as simple as splitting the problem into
                    //       2D elements and working from there, but let's
                    //       look into it when we need it in the future.
                    std::cout << "commencing 3d vortex tracking" << '\n';

                    // Creating the necessary double* values
                    double* edges = (double *)malloc(sizeof(double)*gridSize);

                    // calling the kernel to find the edges
                    find_edges(par, cupar, wave, wfc, edges);

                    // Now we need to output everything
                    if ( write_it){
                        FileIO::writeOutDouble(buffer, data_dir + "Edges",
                                               edges, gridSize, i);
                    }

                    fileName = "wfc_ev";
                    break;
                }
                case 3:
                {
                    fileName = "wfc_ev_ramp";
                    break;
                }
                default:
                {
                    break;
                }
            }

            //std::cout << "writing" << '\n';
            if (write_it) {
                FileIO::writeOut(buffer, data_dir + fileName,
                                 wfc, xDim*yDim*zDim, i);
            }
            //std::cout << "written" << '\n';
            if (par.bval("energy_calc")){
                printf("Energy[t@%d]=%E\n",i,energy_angmom(V_gpu,
                       K_gpu, gpuWfc, gstate, par, cupar));
            }
        }

        // No longer writing out

        // ** ########################################################## ** //
        // **                     More F'n' Dragons!                     ** //
        // ** ########################################################## ** //

        // U_r(dt/2)*wfc
        if(nonlin == 1){
            //std::cout << Dt << '\t' << mass << '\t' << omegaZ << '\t'
            //          << gstate << '\t' << N*interaction << '\n';
            cMultDensity<<<grid,threads>>>(V_gpu,gpuWfc,gpuWfc,0.5*Dt,
                                           mass,gstate,interaction*gDenConst);
        }
        else {
            cMult<<<grid,threads>>>(V_gpu,gpuWfc,gpuWfc);
        }

        // U_p(dt)*fft2(wfc)
        result = cufftExecZ2Z(plan_3d,gpuWfc,gpuWfc,CUFFT_FORWARD);

        // Normalise
        scalarMult<<<grid,threads>>>(gpuWfc,renorm_factor_3d,gpuWfc);
        cMult<<<grid,threads>>>(K_gpu,gpuWfc,gpuWfc);
        result = cufftExecZ2Z(plan_3d,gpuWfc,gpuWfc,CUFFT_INVERSE);

        // Normalise
        scalarMult<<<grid,threads>>>(gpuWfc,renorm_factor_3d,gpuWfc);

        // U_r(dt/2)*wfc
        if(nonlin == 1){
            cMultDensity<<<grid,threads>>>(V_gpu,gpuWfc,gpuWfc,Dt*0.5,
                                           mass,gstate,interaction*gDenConst);
        }
        else {
            cMult<<<grid,threads>>>(V_gpu,gpuWfc,gpuWfc);
        }

        // Angular momentum pAy-pAx (if engaged)  //
        if(lz == true){
            // Multiplying by ramping factor if necessary
            // Note: using scalarPow to do the scaling inside of the exp
            if (ramp){
                scalarPow<<<grid,threads>>>((cufftDoubleComplex*) gpu1dpAy, 
                                            omega_0,
                                            (cufftDoubleComplex*) gpu1dpAy);
                scalarPow<<<grid,threads>>>((cufftDoubleComplex*) gpu1dpAx, 
                                            omega_0,
                                            (cufftDoubleComplex*) gpu1dpAx);
                scalarPow<<<grid,threads>>>((cufftDoubleComplex*) gpu1dpAz, 
                                            omega_0,
                                            (cufftDoubleComplex*) gpu1dpAz);
            }
            int size = xDim*zDim;
            switch(i%2 | (gstate<<1)){
                case 0: //Groundstate solver, even step

                    // 1d forward / mult by Az
                    result = cufftExecZ2Z(plan_1d,gpuWfc,gpuWfc,CUFFT_FORWARD);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAz, gpuWfc);
                    result = cufftExecZ2Z(plan_1d,gpuWfc,gpuWfc,CUFFT_INVERSE);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d, gpuWfc);

                    // loop to multiply by Ay
                    for (int i = 0; i < yDim; i++){
                        //size = xDim * zDim;
                        result = cufftExecZ2Z(plan_dim2,
                                 &gpuWfc[i*size],
                                 &gpuWfc[i*size],CUFFT_FORWARD);
                    }

                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAy, gpuWfc);

                    for (int i = 0; i < yDim; i++){
                        //size = xDim * zDim;
                        result = cufftExecZ2Z(plan_dim2,
                                 &gpuWfc[i*size],
                                 &gpuWfc[i*size],CUFFT_INVERSE);
                    }
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);

                    // 1D FFT to Ax
                    result = cufftExecZ2Z(plan_dim3,gpuWfc,gpuWfc,
                                          CUFFT_FORWARD);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAx, gpuWfc);

                    result = cufftExecZ2Z(plan_dim3,gpuWfc,gpuWfc,
                                          CUFFT_INVERSE);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d, gpuWfc);

                    break; 

                case 1: //Groundstate solver, odd step

                    // 1D FFT to Ax
                    result = cufftExecZ2Z(plan_dim3,gpuWfc,gpuWfc,
                                          CUFFT_FORWARD);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAx, gpuWfc);
  
                    result = cufftExecZ2Z(plan_dim3,gpuWfc,gpuWfc,
                                          CUFFT_INVERSE);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d, gpuWfc);

                    // loop to multiply by Ay
                    for (int i = 0; i < yDim; i++){
                        //size = xDim * zDim;
                        result = cufftExecZ2Z(plan_dim2,
                                 &gpuWfc[i*size],
                                 &gpuWfc[i*size],CUFFT_FORWARD);
                    }

                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAy, gpuWfc);

                    for (int i = 0; i < yDim; i++){
                        //size = xDim * zDim;
                        result = cufftExecZ2Z(plan_dim2,
                                 &gpuWfc[i*size],
                                 &gpuWfc[i*size],CUFFT_INVERSE);
                    }
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);

                    // 1d forward / mult by Az
                    result = cufftExecZ2Z(plan_1d,gpuWfc,gpuWfc,CUFFT_FORWARD);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAz, gpuWfc);
                    result = cufftExecZ2Z(plan_1d,gpuWfc,gpuWfc,CUFFT_INVERSE);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d, gpuWfc);

                    break;
                
                case 2: //Real time evolution, even step
                    // 1d forward / mult by Az
                    result = cufftExecZ2Z(plan_1d,gpuWfc,gpuWfc,CUFFT_FORWARD);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAz, gpuWfc);
                    result = cufftExecZ2Z(plan_1d,gpuWfc,gpuWfc,CUFFT_INVERSE);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d, gpuWfc);

                    // loop to multiply by Ay
                    for (int i = 0; i < yDim; i++){
                        result = cufftExecZ2Z(plan_dim2,
                                 &gpuWfc[i*size],
                                 &gpuWfc[i*size],CUFFT_FORWARD);
                    }

                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAy, gpuWfc);

                    for (int i = 0; i < yDim; i++){
                        result = cufftExecZ2Z(plan_dim2,
                                 &gpuWfc[i*size],
                                 &gpuWfc[i*size],CUFFT_INVERSE);
                    }
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);


                    // 1D FFT to Ax
                    result = cufftExecZ2Z(plan_dim3,gpuWfc,gpuWfc,
                                          CUFFT_FORWARD);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAx, gpuWfc);

                    result = cufftExecZ2Z(plan_dim3,gpuWfc,gpuWfc,
                                          CUFFT_INVERSE);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d, gpuWfc);

                    break;

                case 3: //Real time evolution, odd step

                    // 1D FFT to Ax
                    result = cufftExecZ2Z(plan_dim3,gpuWfc,gpuWfc,
                                          CUFFT_FORWARD);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAx, gpuWfc);

                    result = cufftExecZ2Z(plan_dim3,gpuWfc,gpuWfc,
                                          CUFFT_INVERSE);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d, gpuWfc);

                    // loop to multiply by Ay
                    for (int i = 0; i < yDim; i++){
                        result = cufftExecZ2Z(plan_dim2,
                                 &gpuWfc[i*size],
                                 &gpuWfc[i*size],CUFFT_FORWARD);
                    }

                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAy, gpuWfc);

                    for (int i = 0; i < yDim; i++){
                        result = cufftExecZ2Z(plan_dim2,
                                 &gpuWfc[i*size],
                                 &gpuWfc[i*size],CUFFT_INVERSE);
                    }
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);

                    // 1d forward / mult by Az
                    result = cufftExecZ2Z(plan_1d,gpuWfc,gpuWfc,CUFFT_FORWARD);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d,gpuWfc);
                    cMult<<<grid,threads>>>(gpuWfc,
                        (cufftDoubleComplex*) gpu1dpAz, gpuWfc);
                    result = cufftExecZ2Z(plan_1d,gpuWfc,gpuWfc,CUFFT_INVERSE);
                    scalarMult<<<grid,threads>>>(gpuWfc,
                                                 renorm_factor_1d, gpuWfc);

                    break;
                
            }
        }

        if(gstate==0){
            parSum(gpuWfc, gpuParSum, par, cupar);
        }
    }

    // std::cout << "finished evolution" << '\n';
    // Storing wavefunctions for later
    //std::cout << gpuWfc[0].x << '\t' << gpuWfc[0].y << '\n';
    wave.store("wfc", wfc);
    wave.store("wfc_gpu", gpuWfc);
/*

    par.store("omega", omega);
    par.store("angle_sweep", angle_sweep);
    par.store("gdt", gdt);
    par.store("dt", dt);
    par.store("omegaX", omegaX);
    par.store("omegaY", omegaY);
    par.store("omegaZ", omegaZ);
    par.store("mass", mass);
    par.store("dx", dx);
    par.store("dy", dy);
    par.store("interaction", interaction);
    par.store("laser_power", laser_power);
    par.store("x", x);
    par.store("y", y);
    opr.store("V", V);
    opr.store("V_opt", V_opt);
    wave.store("Phi", Phi);
    opr.store("pAx_gpu", gpu1dpAx);
    opr.store("pAy_gpu", gpu1dpAy);
    wave.store("Phi_gpu", Phi_gpu);
    opr.store("EV", EV);
    //opr.store("V_gpu", V_gpu);
    //opr.store("K_gpu", K_gpu);
    opr.store("EV_opt", EV_opt);

    // getting data from Cuda class
    cupar.store("result", result);
    cupar.store("plan_1d", plan_1d);
    cupar.store("plan_2d", plan_2d);
    cupar.store("grid", grid);

*/
}
