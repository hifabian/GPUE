#!/bin/bash
i=0
count=0
declare -a JOBS=(-1 -1 -1 -1 -1 -1 -1 -1)
function run_gpue_tes {
    echo $1 >> test_file.txt
}
function run_gpue {
    sleep 1
    A=$(date '+%y/%m/%d/%H_%M_%S')
    if [ -d ./$A ]; then
        echo "Exists"
        A=$A-$i
        i=$((i+1))
    fi
    echo $A
    mkdir -p $A
    cp ./gpue ./$A; cp -r ./src ./$A; cp -r ./include ./$A; cp ./Makefile ./$A; cp -r ./py ./$A; cp -r ./bin ./$A; cp ./wfc_load ./$A; cp ./wfci_load ./$A;
    cd ./$A
    pwd >> result.log
    echo $1 >>result.log
    mail -s "#Started GPU Job# $A" lee.oriordan@oist.jp < result.log
    ./gpue $1 2>&1> result.log
    mkdir -p ./images
    python ./py/vis.py >> result.log
    cp *.png ./images
    cd ./images
    ls | grep wfc_evr | grep _abs | grep png | sort -k3 -t _ -n > list1.txt;mencoder mf://@list1.txt -mf w=1280:h=1024:fps=24:type=png -oac copy -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:mv0:trell:v4mv:cbp:last_pred=3:predia=2:dia=2:vmax_b_frames=2:vb_strategy=1:precmp=2:cmp=2:subcmp=2:preme=2:qns=2:vbitrate=10000000 -o wfc_${PWD##*/}.avi
    ls | grep wfc_evr | grep _diff | grep png | sort -k3 -t _ -n > list1.txt;mencoder mf://@list2.txt -mf w=1280:h=1024:fps=24:type=png -oac copy -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:mv0:trell:v4mv:cbp:last_pred=3:predia=2:dia=2:vmax_b_frames=2:vb_strategy=1:precmp=2:cmp=2:subcmp=2:preme=2:qns=2:vbitrate=10000000 -o wfc_${PWD##*/}_diff.avi
    rm -rf ./*.png
    python ./py/hist3d.py
    rm wfc*
    mail -s "#Completed GPU Job# $A" lee.oriordan@oist.jp < result.log
    cd ../../../../..
}

while read line ; do
    run_gpue "$line" &
    #echo "Running $line"
    JOBS[$count]=$!
    let count+=1
    sleep 1
    if [ $count -gt 7 ]; then
        wait
        count=0
    fi
done < ./bin/run_params.conf
