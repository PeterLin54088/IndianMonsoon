# >>> Meta >>>
local_ERA5_fpath="/work/b08209033/DATA/ERA5"
# <<< Meta <<<

# >>> CDO Divergent Wind >>>
cd ${local_ERA5_fpath}

cdo \
-remapbil,r576x360 \
-dv2uv \
-expr,"svo=svo*0.0;sd=sd;" \
-uv2dv \
-remapbil,n120 \
-merge u/u_alllevel_alltime.nc v/v_alllevel_alltime.nc \
Divergent_UV.nc
# <<< CDO Divergent Wind <<<

# -zonavg \
# -sellonlatbox,40,100,0,30 \