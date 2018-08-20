#!/bin/bash -l

#SBATCH --job-name=GPUE_SBATCH
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00 
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee-oriordan2@oist.jp
#SBATCH --output=JOB_%j.log
#SBATCH --error=JOB_%j.err

#Script arguments
BASEDIR=$1 #Directory to base all simulation dirs within. Must contain wfc_load and wfci_load
GPUE=$2 #Pass in binary as argument
DIRNAME=$3 #Name of directory to store data
SHIFT_FLIP=$4 #Indicate if shifting (1) or flipping (-1) new vortex [changes sign of imprint]
XOFFSET=$5 #Offset along X for imprinted vortex from previous position in units of DX
YOFFSET=$6 #Offset along Yfor imprinted vortex from previous position in units of DY
ATOMS=$7 #Atom count to control interaction strength in condensate
RES=$8 #2D grid resolution for x and y
OMEGA=$9 #Rotation frequency of trap relative to trapping frquency

echo	"BASEDIR=${BASEDIR}" \
	"GPUE=${GPUE}" \
	"DIRNAME=${DIRNAME}" \
	"SHIFT_FLIP=${SHIFT_FLIP}" \
	"XOFFSET=${XOFFSET}" \
	"YOFFSET=${YOFFSET}" \
	"ATOMS=${ATOMS}" \
	"RES=${RES}" \
	"OMEGA=${OMEGA}"

#Parameter values
REALTIME_STEPS=5.0001e6
REALTIME_DT=1e-5
PRINT_INTERVAL=1000
RESX=${RES} #Resolution of simulation along X and Y; specifies number of discrete intervals
RESY=${RES}
HF=false
Omega_z=100 #Harmonic confinement along z dim; in 2D must be >> Omega_x and Omega_y [scales interaction strength]
Omega_x=6.283185307179586 #2Pi
Omega_y=6.283185307179586 #2Pi
VORTEX_MASK=3.5e-5 # 1.00e-4 #Radius of range to examine condensate for vortices from central point

GSTEPS=0
G_DT=1e-4
READIN="-r"

#Create directory structure, and run the code
cd ${BASEDIR}
module load cuda
mkdir -p ${DIRNAME} 

LOADED=false
CENTRAL_IMPRINT=0

if ! ${HF};then
if [[ "${ATOMS}" == "1e6"  ]];then
    cp ./wfc_load_1024 ${DIRNAME}/wfc_load &&
    cp ./wfci_load_1024 ${DIRNAME}/wfci_load &&
    LOADED=true 
elif [[ "${ATOMS}" == "1e7" ]];then
    cp ./wfc_load_2048_4Rxy ${DIRNAME}/wfc_load &&
    cp ./wfci_load_2048_4Rxy ${DIRNAME}/wfci_load &&
    LOADED=true
fi
else
    cp ./wfc_load_1024_xy62z1000w2_4Rxy ${DIRNAME}/wfc_load &&
    cp ./wfci_load_1024_xy62z1000w2_4Rxy ${DIRNAME}/wfci_load &&
    LOADED=true
fi

if ! ${LOADED};then
    echo "NOT LOADED"
    GSTEPS=1.0001e6
    READIN=""
    PRINT_INTERVAL=100000
    REALTIME_STEPS=0
    CENTRAL_IMPRINT=4
fi

cp ${GPUE} ./${DIRNAME}/
echo "${GPUE} -d ${DIRNAME} \
	-g ${GSTEPS} \
	-e ${REALTIME_STEPS} \
	-t ${REALTIME_DT} \
	-T ${G_DT} \
	-n ${ATOMS} \
	-p ${PRINT_INTERVAL} \
	-W \
	-w ${OMEGA} \
	-x ${RESX} \
	-y ${RESY} \
	-Z ${Omega_z} \
	-s \
	-l \
    -L ${CENTRAL_IMPRINT} \
	-U ${XOFFSET} \
	-V ${YOFFSET} \
	-X ${Omega_x} \
	-Y ${Omega_y} \
	-K 1 \
	-q ${SHIFT_FLIP} \
	${READIN} \
	-a \
	-m ${VORTEX_MASK} > >(tee -a ${DIRNAME}.log) 2> >(tee -a ${DIRNAME}.err >&2)"

${GPUE} -d ${DIRNAME} \
	-g ${GSTEPS} \
	-e ${REALTIME_STEPS} \
	-t ${REALTIME_DT} \
	-T ${G_DT} \
	-n ${ATOMS} \
	-p ${PRINT_INTERVAL} \
	-W \
	-w ${OMEGA} \
	-x ${RESX} \
	-y ${RESY} \
	-Z ${Omega_z} \
	-s \
	-l \
    -L ${CENTRAL_IMPRINT} \
	-U ${XOFFSET} \
	-V ${YOFFSET} \
	-X ${Omega_x} \
	-Y ${Omega_y} \
	-K 1 \
	-q ${SHIFT_FLIP} \
	${READIN} \
	-a \
	-m ${VORTEX_MASK} > >(tee -a ${DIRNAME}.log) 2> >(tee -a ${DIRNAME}.err >&2)
