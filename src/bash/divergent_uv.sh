# >>> Meta >>>
destination_path="/work/b08209033/DATA/IndianMonsoon/ERA5/raw_grid"
# <<< Meta <<<


# >>> CDO Non-rotational Wind >>>
cdo \
-remapbil,ERA5_grid \
-dv2uv \
-expr,"svo=svo*0.0;sd=sd;" \
-uv2dv \
-remapbil,F180 \
-merge ${destination_path}/u.nc ${destination_path}/v.nc \
${destination_path}/pure_divergent_wind.nc
# <<< CDO Non-rotational Wind <<<


# >>> CDO Zonal Mean >>>
# cdo zonavg ${destination_path}/divergent_uv.nc ${destination_path}/zonavg_divergent_uv.nc
# <<< CDO Zonal Mean <<<


# >>> CDO Select LonLat >>>
# cdo sellonlatbox,45,100,0,25 ${destination_path}/divergent_uv.nc ${destination_path}/indian_divergent_uv.nc
# <<< CDO Select LonLat <<<