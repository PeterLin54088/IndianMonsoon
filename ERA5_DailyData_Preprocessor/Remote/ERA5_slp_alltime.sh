# >>> Meta >>>
varname="slp"
remote_ERA5_fpath="/work/DATA/Reanalysis/ERA5"
local_ERA5_fpath="/work/b08209033/DATA/ERA5"
# <<< Meta <<<

# >>> Extract Filename Patterns >>>
cd ${remote_ERA5_fpath}/${varname}
years=();
for complete_filename in ${varname}_*[0-9]*.nc; do
    filename="${complete_filename%.*}"
    year="${filename#*_}"
    if [[ ! " ${years[*]} " =~ [[:space:]]${year}[[:space:]] ]]; 
    then
        years+=($year)
    fi
done
echo "Start year: ${years[0]}"
# <<< Extract Filename Patterns <<<

# >>> CDO Mergetime >>>
cd ${local_ERA5_fpath}
cdo \
--no_history \
-setattribute,history="$(date): Reinitialized by HSUAN-CHENG LIN" \
-setreftime,"1900-01-01,09:00:00" \
-settaxis,"${years[0]}-01-01,09:00:00,24hour" \
-del29feb \
-mergetime \
${remote_ERA5_fpath}/${varname}/${varname}_*[0-9]*.nc \
${local_ERA5_fpath}/${varname}/${varname}_alltime.nc
# <<< CDO Mergetime <<<