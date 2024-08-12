# >>> Meta >>>
temporary_path="/work/b08209033/DATA/tmp"
destination_path="/work/b08209033/DATA/IndianMonsoon/ERA5"
available_pressure_level=(1000 925 850 700 500 250)
# <<< Meta <<<


# >>> CDO Split Level & Re-expr >>>
for plevel in ${available_pressure_level[*]}
do
    #
    expression=""
    expression+="theta=t*(100000/$((plevel*100)))^(0.285);"
    #
    cdo \
    -expr,${expression} \
    -sellevel,$((plevel*100)) \
    ${destination_path}/t.nc \
    ${temporary_path}/theta_${plevel}.nc
done
# <<< CDO Split Level & Re-expr <<<


# >>> CDO Merge Level >>>
cd ${temporary_path}
declare -a fnames; fnames=(theta_*[0-9]*.nc)
IFS=$'\n'; sorted_fnames=($(sort -n -r -t _ -k 2 <<<"${fnames[*]}"))
echo "Merging ${sorted_fnames[*]}"
cdo \
--no_history \
-setattribute,history="$(date): Reinitialized by HSUAN-CHENG LIN" \
-merge \
${sorted_fnames[*]} \
${destination_path}/theta.nc
rm -rf ${temporary_path}/theta_*[0-9]*.nc
# <<< CDO Merge Level <<<


# >>> CDO Select LonLat >>>
cdo sellonlatbox,45,100,0,25 ${destination_path}/theta.nc ${destination_path}/indian_theta.nc
# <<< CDO Select LonLat <<<