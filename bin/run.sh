#!/bin/bash
i=0
EMAIL=$1
count=0
NAME=$2
PARAMS=$3
declare -a JOBS=(-1 -1 -1 -1 -1 -1 -1 -1)
function run_gpue_test {
	echo $1
}

function run_gpue {
	if [ -n  "$NAME" ];then
		NAME=$(echo $NAME)_
	fi
	A=$(date '+%y/%m/%d/%H_%M_%S')
	if [ -d ./$A ]; then
		echo "Exists"
		A=$A-$i
		i=$((i+1))
	fi
	echo "$NAME$A"
	mkdir -p $NAME$A
	cp ./gpue ./$NAME$A; cp -r ./src ./$NAME$A; cp -r ./include ./$NAME$A; cp ./Makefile ./$NAME$A; cp -r ./py ./$NAME$A; cp -r ./bin ./$NAME$A; cp ./wfc_load ./$NAME$A; cp ./wfci_load ./$NAME$A;
	cd ./$NAME$A
	pwd >> result.log
	echo $1 >>result.log
	mail -s "#Started GPU Job# $A" ${EMAIL} < result.log
	./gpue $1 2>&1> result.log
	mkdir -p ./images
	#python ./py/vis.py >> result.log
	cp *.png ./images
	cd ./images
	ls | grep wfc_evr | grep _abs | grep png | sort -k3 -t _ -n > list1.txt;mencoder mf://@list1.txt -mf w=1280:h=1024:fps=24:type=png -oac copy -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:mv0:trell:v4mv:cbp:last_pred=3:predia=2:dia=2:vmax_b_frames=2:vb_strategy=1:precmp=2:cmp=2:subcmp=2:preme=2:qns=2:vbitrate=10000000 -o wfc_${PWD##*/}.avi
	rm -rf ./*.png
	rm wfc*
	mail -s "#Completed GPU Job# $A" $EMAIL < $(echo $(cat result.log; cat ./Params.dat))
	cd ../../../../..
	sleep 1
}

while read line ; do
	if [[ $(echo $line | head -c 1) == "#" ]];then
		continue;
	elif [[ $(echo $line | head -c 1) == "" ]];then 
		continue;
	else 
		sleep 1;
		run_gpue "$line" &
	fi
	
	#echo "Running $line"
	JOBS[$count]=$!
	let count+=1
	if [ $count -gt 7 ]; then
		wait
		count=0
	fi
done < $PARAMS
