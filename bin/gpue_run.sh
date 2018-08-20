#!/bin/bash -l

#Script arguments
BASEDIR=/work/scratch/mlxd/gpue_dir/DATA #Directory to base all simulation dirs within. Must contain wfc_load and wfci_load
GPUE=/work/scratch/mlxd/gpue_dir/gpue #gpue #Pass in binary as argument

YOFFSET=0
A=(1e7);
S=(-1 );
X=$(seq -20 10 20 );
RES=(1024 2048) #For increased interaction strength (1e7 atoms) the vortex core becomes tiny. Need higher resoltuion to resolve the structure;
OMEGA=(0.2 0.2)

for ATOMS in "${A[@]}"; do
	for SHIFT_FLIP in "${S[@]}"; do
		for XOFFSET in ${X}; do
			sleep 0.5
			XDIV10=$(perl -e "print ${XOFFSET}/10")
			DIRNAME="ATOMS${ATOMS}_SF${SHIFT_FLIP}_XOFFSET${XDIV10}_freq10perp"
			cd ${BASEDIR}
			mkdir -p ${DIRNAME}
			R=""
			if [[ "${ATOMS}" == "1e7" ]];then
				R=${RES[1]}
				O=${OMEGA[1]}
			else
				R=${RES[0]}
				O=${OMEGA[0]}
			fi
			cp /work/scratch/mlxd/gpue_dir/DATA/MLXD_slurm/gpue_sbatch.slurm ${DIRNAME}
			echo "sbatch /work/scratch/mlxd/gpue_dir/DATA/MLXD_slurm/gpue_sbatch.slurm \
				${BASEDIR} ${GPUE} ${DIRNAME} ${SHIFT_FLIP} ${XDIV10} ${YOFFSET} ${ATOMS} ${R} ${O}" \
				&> ${BASEDIR}/${DIRNAME}/runtime.log
			sbatch /work/scratch/mlxd/gpue_dir/DATA/MLXD_slurm/gpue_sbatch.slurm \
				${BASEDIR} ${GPUE} ${DIRNAME} ${SHIFT_FLIP} ${XDIV10} ${YOFFSET} ${ATOMS} ${R} ${O}
		done
	done
done



#ffmpeg -framerate 10 -i wfc_evr_%d000_abspsi2.png -s:v 1280x720 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ATOMS1e6_SF1_XOFFSET0.mp4
