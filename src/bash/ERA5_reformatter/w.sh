# >>> Meta >>>
varname="w"
source_path="/work/DATA/Reanalysis/ERA5"
temporary_path="/work/b08209033/DATA/tmp"
destination_path="/work/b08209033/DATA/IndianMonsoon/ERA5"
# <<< Meta <<<


# >>> Extract Filename Patterns >>>
cd ${source_path}/${varname}
years=(); levels=();
for complete_filename in ${varname}*[0-9]*_*[0-9]*.nc; do
    filename="${complete_filename%.*}"
    tmp="${filename/${varname}}"
    level="${tmp%_*}"
    year="${tmp#*_}"
    if [[ ! " ${years[*]} " =~ [[:space:]]${year}[[:space:]] ]] 
    then
        years+=($year)
    fi
    if [[ ! " ${levels[*]} " =~ [[:space:]]${level}[[:space:]] ]];
    then
        levels+=($level)
    fi
done
years=($(printf '%s\n' "${years[@]}" | sort -n))
levels=($(printf '%s\n' "${levels[@]}" | sort -nr))
echo "Years: ${years[*]}"
echo "Pressure levels: ${levels[*]}"
# <<< Extract Filename Patterns <<<


# >>> CDO Merge Time >>>
for plevel in ${levels[*]}
do
    # pressure configuration
    TMPFILE=$(mktemp /tmp/cdo-PeterLin-XXXXXXXXXXXXXXXXXXXX)
    trap "rm -f $TMPFILE" EXIT
    echo "zaxistype = pressure" > $TMPFILE
    echo "size = 1" >> $TMPFILE
    echo "levels = $((plevel * 100))" >> $TMPFILE
    cat $TMPFILE
    #
    cdo \
    -setzaxis,$TMPFILE \
    -setreftime,"1900-01-01,09:00:00" \
    -settaxis,"${years[0]}-01-01,09:00:00,24hour" \
    -del29feb \
    -mergetime \
    ${source_path}/${varname}/${varname}${plevel}_*[0-9]*.nc \
    ${temporary_path}/${varname}_${plevel}.nc
done
# <<< CDO Merge Time <<<


# >>> CDO Merge Level >>>
cd ${temporary_path}
declare -a fnames; fnames=(${varname}_*[0-9]*.nc)
IFS=$'\n'; sorted_fnames=($(sort -n -r -t _ -k 2 <<<"${fnames[*]}"))
echo "Merging ${sorted_fnames[*]}"
cdo \
--no_history \
-setattribute,history="$(date): Reinitialized by HSUAN-CHENG LIN" \
-merge \
${sorted_fnames[*]} \
${destination_path}/${varname}.nc
rm -rf ${temporary_path}/${varname}_*[0-9]*.nc
# <<< CDO Merge Level <<<