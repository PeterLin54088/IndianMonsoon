# >>> Meta >>>
source_path="/work/b08209033/DATA/IndianMonsoon/ERA5"
destination_path="/work/b08209033/DATA/IndianMonsoon/TempestExtremes"
plevel=925 # hPa
svo_limit=0.00003 # s-1
# <<< Meta <<<

input_files=""
input_files+="${source_path}/slp.nc;"
input_files+="${source_path}/vor${plevel}.nc"
output_files=""
output_files+="${destination_path}/TropicalCycloneCount_${plevel}.csv"

DetectNodes \
--in_data ${input_files} \
--out ${output_files} \
--searchbymin msl \
--mergedist 5 \
--closedcontourcmd "msl,100,5,0;svo,-${svo_limit},5,0" \
--thresholdcmd "msl,<=,100400,0;svo,>=,${svo_limit},0" \
--outputcmd "msl,min,0;svo,max,0" \
--minlon 45 \
--maxlon 100 \
--minlat 0 \
--maxlat 25;