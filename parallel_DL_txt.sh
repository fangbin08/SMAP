#!/bin/bash
#-----------------------BEGIN NOTICE -- DO NOT EDIT-----------------------
# University of Virginia, Hydrosense Group
#-------------------------END NOTICE -- DO NOT EDIT-----------------------

#BOP
# !ROUTINE: parallel_DL_txt.sh
#
# !REVISION HISTORY:
# 07 Nov 2019: Hyunglok Kim; initial specification for all systems (mac, linux, window)
#                          ; random-wait option was added incase EarthData performs log analysis

# !ARGUMENTS:(REFER THE DEISCRIPTION BELOW)
ifpfn="/Users/binfang/Downloads/Processing/bash_codes/download/0521940633-download.txt"
ofp="/Users/binfang/Downloads/Processing/bash_codes/download"                
N="3"
ED_ID="XXXXX"
ED_Pass="XXXXX"
speed_lim="3m"

# !DESCRIPTION:   
#   This bash code makes possible to download multiple urls
#   Input file should be a txt file which includes a url for each line
#   Resume downloading the same file if some files have failed in the last time queue
#
#   The arguments are: 
#   \begin{description}
#    \item[ifpfn]       Input *.txt file's location
#    \item[ofp]         Output folder path
#    \item[N]           The number of urls you want to download at one time queue
#    \item[ED_ID]       Your EarthData ID
#    \item[ED_Pass]     Your EarthData Password
#    \item[speed\_lim]  Download speed limit (ex. 5m > 5mb/s; 50k > 50kb/s)
#   \end{description}
#
# !HOW TO USE THIS CODE
#   1) Put this script file to your $working_directory
#   2) Use 'cd' command to move your current directory to your #working_directory
#   3) Set ifpfn, ofp, N, ED_ID, ED_Pass, and speed_lim accordingly
#   4) Type 'chmod 777 parallel_DL_txt.sh' in the command line
#   5) Type './parallel_DL_txt.sh' in the command line to excute
#
# !COMMENTS:
#    N > 20 is not recommended; large N may cause lag when you download data (depends on your network conditons)
#    Use several $pkill wget commands if you need to stop downloading immediately
#EOP

cd $ofp
nof=$(wc -l < $ifpfn)
i="1"
while [ $i -lt $nof ];do

    for j in $(eval echo "{$[1+N*(i-1)]..$[N*i]}");do    
        if [ $j -le $nof ];then
            echo Downloading file: $j / $nof
            wget --user=$ED_ID --password=$ED_Pass --limit-rate=$speed_lim -q -c -o logfile --random-wait $(sed -n $j'p' < $ifpfn) &
        fi
    done
    i=$[$i+1]
    wait
    
done
echo Download finished...
