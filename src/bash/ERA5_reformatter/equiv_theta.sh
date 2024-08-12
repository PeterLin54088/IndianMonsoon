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
    expression+="equiv_theta=(t+2500*q)*(100000/$((plevel*100)))^(0.285);"
    merge_files=""
    merge_files+="${temporary_path}/t_${plevel}.nc "
    merge_files+="${temporary_path}/q_${plevel}.nc "
    #
    cdo sellevel,$((plevel*100)) ${destination_path}/t.nc ${temporary_path}/t_${plevel}.nc
    cdo sellevel,$((plevel*100)) ${destination_path}/q.nc ${temporary_path}/q_${plevel}.nc
    #
    cdo \
    -expr,${expression} \
    -merge ${merge_files} \
    ${temporary_path}/equiv_theta_${plevel}.nc
    #
    rm -rf ${temporary_path}/t_${plevel}.nc
    rm -rf ${temporary_path}/q_${plevel}.nc
done
# <<< CDO Split Level & Re-expr <<<


# >>> CDO Merge Level >>>
cd ${temporary_path}
declare -a fnames; fnames=(equiv_theta_*[0-9]*.nc)
IFS=$'\n'; sorted_fnames=($(sort -n -r -t _ -k 3 <<<"${fnames[*]}"))
echo "Merging ${sorted_fnames[*]}"
cdo \
--no_history \
-setattribute,history="$(date): Reinitialized by HSUAN-CHENG LIN" \
-merge \
${sorted_fnames[*]} \
${destination_path}/equiv_theta.nc
rm -rf ${temporary_path}/equiv_theta_*[0-9]*.nc
# <<< CDO Merge Level <<<


# >>> CDO Select LonLat >>>
cdo sellonlatbox,45,100,0,25 ${destination_path}/equiv_theta.nc ${destination_path}/indian_equiv_theta.nc
# <<< CDO Select LonLat <<<