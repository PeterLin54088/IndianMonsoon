DetectNodes \
--in_data "/work/b08209033/preprocess_ERA5/preprocess_MSLP.nc;/work/b08209033/preprocess_ERA5/preprocess_Vorticity_925.nc" \
--out "TropicalCycloneCount.csv" \
--searchbymin msl \
--mergedist 5 \
--closedcontourcmd "msl,100,5,0;svo,-0.00003,5,0" \
--thresholdcmd "msl,<=,100400,0;svo,>=,0.00003,0" \
--outputcmd "msl,min,0;svo,max,0" \
--minlon 45 \
--maxlon 100 \
--minlat 0 \
--maxlat 30;