plevel=925
svo_limit=0.00003

DetectNodes \
--in_data "/work/b08209033/DATA/ERA5/slp/slp_alltime.nc;/work/b08209033/DATA/ERA5/div_vor/test.nc" \
--out "TropicalCycloneCount_${plevel}.csv" \
--searchbymin msl \
--mergedist 5 \
--closedcontourcmd "msl,100,5,0;svo,-${svo_limit},5,0" \
--thresholdcmd "msl,<=,100400,0;svo,>=,${svo_limit},0" \
--outputcmd "msl,min,0;svo,max,0" \
--minlon 45 \
--maxlon 100 \
--minlat 0 \
--maxlat 30;