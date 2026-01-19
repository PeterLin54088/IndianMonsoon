# >>> Meta >>>
destination_path="/work/b08209033/DATA/IndianMonsoon/ERA5/raw_grid"
# <<< Meta <<<


# >>> CDO Non-rotational Wind >>>
cdo \
-remapbil,ERA5_grid \
-sp2gp \
-uv2dv \
-remapbil,F180 \
-merge ${destination_path}/u.nc ${destination_path}/v.nc \
${destination_path}/div_vor.nc
# <<< CDO Non-rotational Wind <<<