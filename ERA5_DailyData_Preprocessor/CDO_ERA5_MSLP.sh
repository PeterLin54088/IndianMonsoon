# Meta
varname_input="slp"
varname_output="MSLP"
pre_ERA5_fpath="/work/DATA/Reanalysis/ERA5"
post_ERA5_fpath="/work/b08209033/preprocess_ERA5"

# Extract patterns
: '
The pattern should be ${varname}_{year}.nc
but the year is unknown
extract them from possible filenames
'
cd ${pre_ERA5_fpath}/${varname_input}
years=()
for fullname in ${varname_input}_*[0-9]*.nc; do
    fname="${fullname%.*}"
    tmp="${fname/${varname_input}}"
    year="${fname#*_}"
    if [[ ! " ${years[*]} " =~ [[:space:]]${year}[[:space:]] ]]; 
    then
        years+=($year)
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
'
cd ${post_ERA5_fpath}
cdo \
--no_history \
-setattribute,history="$(date): Reinitialized by HSUAN-CHENG LIN" \
-setreftime,"1900-01-01,09:00:00" \
-settaxis,"${years[0]}-01-01,09:00:00,24hour" \
-del29feb \
-mergetime \
${pre_ERA5_fpath}/${varname_input}/${varname_input}_*[0-9]*.nc \
${post_ERA5_fpath}/${varname_output}.nc