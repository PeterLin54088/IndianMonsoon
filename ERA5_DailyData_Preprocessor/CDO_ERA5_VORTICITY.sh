# cdo merge preprocess_U.nc preprocess_V.nc preprocess_UV.nc

# for regular grid
# -remapbil,r384x192
# for gaussian grid
# -remapbil,n90

# cdo expr,"svo=svo*0.0;sd=sd;" -uv2dv -remapbil,n90 preprocess_UV.nc preprocess_zero_DV.nc

# Meta
varname_input="div_vor"
varname_output="Vorticity"
pre_ERA5_fpath="/work/DATA/Reanalysis/ERA5"
post_ERA5_fpath="/work/b08209033/preprocess_ERA5"

# Extract patterns
: '
The pattern should be ${varname}_${pressure}_{year}.nc
but the pressure & year is unknown
extract them from possible filenames
'
cd ${pre_ERA5_fpath}/${varname_input}
years=()
levels=()
for fullname in ${varname_input}*[0-9]*_*[0-9]*.nc; do
    fname="${fullname%.*}"
    tmp="${fname#*_*_}"
    plevel="${tmp%_*}"
    year="${fname##*_}"
    if [[ ! " ${years[*]} " =~ [[:space:]]${year}[[:space:]] ]]; 
    then
        years+=($year)
    fi

    if [[ ! " ${levels[*]} " =~ [[:space:]]${plevel}[[:space:]] ]]; 
    then
        levels+=($plevel)
    fi
done
unset fullname
unset fname
unset tmp
unset plevel
unset year
echo "Start year: ${years[0]}"

# CDO
: '
1. Merge time
2. Remove Feb 29
3. Reset reference time and time interval units
4. Overwrite global attribution text
5. Insert pressure level
'
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
echo "zaxistype = pressure" > $TMPFILE
echo "size = 1" >> $TMPFILE
echo "levels = $((plevel * 100))" >> $TMPFILE
cat $TMPFILE
cd ${post_ERA5_fpath}
cdo \
--no_history \
-setzaxis,$TMPFILE \
-setattribute,history="$(date): Reinitialized by HSUAN-CHENG LIN" \
-setreftime,"1900-01-01,09:00:00" \
-settaxis,"${years[0]}-01-01,09:00:00,24hour" \
-del29feb \
-selvar,svo \
-mergetime \
${pre_ERA5_fpath}/${varname_input}/${varname_input}_${plevel}_*[0-9]*.nc \
${post_ERA5_fpath}/${varname_output}_${plevel}.nc
rm $TMPFILE