#include <iostream>
#include <string>
#include "./include/tracker.h"
#include "./include/constants.h"
#include "./include/vort.h"
#include "./include/fileIO.h"

/*struct double2{
    double x;
    double y;
};*/

    void writeOutVortex(std::string buffer, std::string file,
                            std::vector<std::shared_ptr<Vtx::Vortex>> &data, int step){
        FILE *f;
        sprintf ((char *)buffer.c_str(), "%s_%d", file.c_str(), step);

        f = fopen (buffer.c_str(),"w");
        int i;

        fprintf (f, "#UID,X,Xd,Y,Yd,WINDING,isOn\n");
        for (i = 0; i < data.size(); i++)
            //fprintf (f, "%d,%d,%e,%d,%e,%d\n",data[i]->getUID(),data[i]->getCoords().x,data[i]->getCoordsD().x,data[i]->getCoords().y,data[i]->getCoordsD().y,data[i]->getWinding());
            fprintf (f, "%d,%e,%d,%e,%d\n",data[i]->getCoords().x,data[i]->getCoordsD().x,data[i]->getCoords().y,data[i]->getCoordsD().y,data[i]->getWinding());
        fclose (f);
    }


double2 complexMult(double2 in1, double2 in2){
    double2 result;
    result.x = (in1.x*in2.x - in1.y*in2.y);
    result.y = (in1.x*in2.y + in1.y*in2.x);
    return result;
}

double2 complexScale(double2 comp, double scale){
    double2 result;
    result.x = comp.x*scale;
    result.y = comp.y*scale;
    return result;
}

double complexMag(double2 in){
    return sqrt(in.x*in.x + in.y*in.y);
}

double complexMag2(double2 in){
    return in.x*in.x + in.y*in.y;
}

double2 conj(double2 c){
    double2 result = c;
    result.y = -result.y;
    return result;
}

double2 complexDiv(double2 num, double2 den){
    double2 c = conj(den);
    return complexScale(complexMult(num,c),(1.0/complexMag2(den)));
}

void readIn(double2* wfc, const char* fileR, const char* fileI, int xDim, int yDim){
    FILE *f;
    f = fopen(fileR,"r");
    int i = 0;
    double line;
    while(fscanf(f,"%lE",&line) > 0){
        wfc[i].x = line;
        ++i;
    }
    fclose(f);
    f = fopen(fileI,"r");
    i = 0;
    while(fscanf(f,"%lE",&line) > 0){
        wfc[i].y = line;
        ++i;
    }
    fclose(f);
    //return arr;
}



void readInD(double *x, const char* file, int xDim){
    FILE *f;
    f = fopen(file,"r");
    int i = 0;
    double line;
    while(fscanf(f,"%lE",&line) > 0){
        x[i] = line;
        ++i;
    }
    fclose(f);
}



int main(int argc, char *argv[]){
    int xDim=2048, yDim=2048;
    int num_vortices[2] = {0,0};
    double vort_angle, sepAvg;
    std::string data_dir = "./";
    char buffer[100];
    double2 *wfc = (double2*) malloc(sizeof(double2)*xDim*yDim);
    double *x = (double*) malloc(sizeof(double)*xDim);

	readInD(x,"x_0",xDim);

    int* vortexLocation;
    std::shared_ptr<Vtx::Vortex> central_vortex; //vortex closest to the central position
    std::shared_ptr<Vtx::VtxList> vortCoords = std::make_shared<Vtx::VtxList>(7);
    std::shared_ptr<Vtx::VtxList> vortCoordsP = std::make_shared<Vtx::VtxList>(7);
    std::string r = "wfc_ev_", i="wfc_evi_", tmpr, tmpi;

    std::string start = argv[1];
    std::string end = argv[2];
    double mask_2d=atof(argv[3]);
    int incr = atoi(argv[4]);
    std::size_t st_st;
    std::size_t st_en;

    for(unsigned long ii=atol(start.c_str()); ii<=atol(end.c_str()); ii+=incr){
		std::cout << "ii=" << ii << std::endl;
        tmpr = r + std::to_string(ii);
        tmpi = i + std::to_string(ii);
        readIn(wfc, tmpr.c_str(), tmpi.c_str(), xDim, yDim);

        vortexLocation = (int *) calloc(xDim * yDim, sizeof(int));
        num_vortices[0] = Tracker::findVortex(vortexLocation, wfc, mask_2d, xDim, x, ii);
        if(ii==std::stoi(start, &st_st)){
            if(num_vortices[0] > 0){
                vortCoords = std::make_shared<Vtx::VtxList>(num_vortices[0]);
                vortCoordsP = std::make_shared<Vtx::VtxList>(num_vortices[0]);
                Tracker::vortPos( vortexLocation, vortCoords->getVortices(), xDim, wfc);
                Tracker::lsFit( vortCoords->getVortices(), wfc, xDim);
                central_vortex = Tracker::vortCentre(vortCoords->getVortices(), xDim);
                vort_angle = Tracker::vortAngle(vortCoords->getVortices(), central_vortex);
                sepAvg = Tracker::vortSepAvg(vortCoords->getVortices(), central_vortex);
            }
        }
        else {
            if (num_vortices[0] > 0){
                Tracker::vortPos(vortexLocation, vortCoords->getVortices(), xDim, wfc);
                Tracker::lsFit(vortCoords->getVortices(), wfc, xDim);
                Tracker::vortArrange(vortCoords->getVortices(), vortCoordsP->getVortices());
            }
        }
        writeOutVortex(buffer, "vort_arr", vortCoords->getVortices(), ii);
        printf("Located %d vortices\n", vortCoords->getVortices().size());
        free(vortexLocation);
		num_vortices[1] = num_vortices[0];
        vortCoords->getVortices().swap(vortCoordsP->getVortices());
        vortCoords->getVortices().clear();
    }
}
