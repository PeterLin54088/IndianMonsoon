temporary_path="/work/b08209033/DATA/tmp"
destination_path="/work/b08209033/DATA/IndianMonsoon/ERA5/raw_grid"

cdo -b 32 \
--no_history \
-setattribute,history="$(date): Reinitialized by HSUAN-CHENG LIN" \
-copy \
${temporary_path}/equivalent_potential_temperature.nc \
${destination_path}/ept.nc \

rm -f ${temporary_path}/equivalent_potential_temperature.nc