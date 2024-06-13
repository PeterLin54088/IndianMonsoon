# >>> Meta >>>
varname="t"
local_ERA5_fpath="/work/b08209033/DATA/ERA5"
# <<< Meta <<<

# >>> CDO Merge Level >>>
cd ${local_ERA5_fpath}/${varname}
declare -a fnames; fnames=(${varname}_*[0-9]*_alltime.nc)
IFS=$'\n'; sorted_fnames=($(sort -n -t _ -k 2 <<<"${fnames[*]}"))

cdo merge ${sorted_fnames[@]} ${varname}_alllevel_alltime.nc
# <<< CDO Merge Level <<<