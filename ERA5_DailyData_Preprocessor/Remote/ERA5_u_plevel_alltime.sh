# >>> Meta >>>
varname="u"
remote_ERA5_fpath="/work/DATA/Reanalysis/ERA5"
local_ERA5_fpath="/work/b08209033/DATA/ERA5"
# <<< Meta <<<

# >>> Extract Filename Patterns >>>
cd ${remote_ERA5_fpath}/${varname}
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
echo "Start year: ${years[0]}"
# <<< Extract Filename Patterns <<<

# >>> CDO Mergetime >>>
while true; do
    read -p "Specify pressure level (hPa): " tmp
    if [[ ! " ${levels[*]} " =~ [[:space:]]${tmp}[[:space:]] ]]
    then
        echo "Not an available pressure level!"
    else
        plevel=$tmp
        break
    fi
done

TMPFILE=$(mktemp /tmp/cdo-PeterLin-XXXXXXXXXXXXXXXXXXXX)
trap "rm -f $TMPFILE" EXIT
echo "zaxistype = pressure" > $TMPFILE
echo "size = 1" >> $TMPFILE
echo "levels = $((plevel * 100))" >> $TMPFILE
cat $TMPFILE

cd ${local_ERA5_fpath}
cdo \
--no_history \
-setzaxis,$TMPFILE \
-setattribute,history="$(date): Reinitialized by HSUAN-CHENG LIN" \
-setreftime,"1900-01-01,09:00:00" \
-settaxis,"${years[0]}-01-01,09:00:00,24hour" \
-del29feb \
-mergetime \
${remote_ERA5_fpath}/${varname}/${varname}${plevel}_*[0-9]*.nc \
${local_ERA5_fpath}/${varname}/${varname}_${plevel}_alltime.nc
# <<< CDO Mergetime <<<