# >>> Meta >>>
varname="slp"
source_path="/work/DATA/Reanalysis/ERA5"
destination_path="/work/b08209033/DATA/IndianMonsoon/ERA5/raw_grid"
# <<< Meta <<<


# >>> Extract Filename Patterns >>>
cd ${source_path}/${varname}
years=();
for complete_filename in ${varname}_*[0-9]*.nc; do
    filename="${complete_filename%.*}"
    year="${filename#*_}"
    if [[ ! " ${years[*]} " =~ [[:space:]]${year}[[:space:]] ]]; 
    then
        years+=($year)
    fi
done
years=($(printf '%s\n' "${years[@]}" | sort -n))
echo "Years: ${years[*]}"
# <<< Extract Filename Patterns <<<


# >>> CDO Mergetime >>>
cdo \
--no_history \
-setattribute,history="$(date): Reinitialized by HSUAN-CHENG LIN" \
-setreftime,"1900-01-01,09:00:00" \
-settaxis,"${years[0]}-01-01,09:00:00,24hour" \
-del29feb \
-mergetime \
${source_path}/${varname}/${varname}_*[0-9]*.nc \
${destination_path}/${varname}.nc
# <<< CDO Mergetime <<<