;---------------------------------------------
; wkSpaceTime.ncl
;---------------------------------------------
;
; These files are loaded by default in NCL V6.2.0 and newer
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
;
; This file still has to be loaded manually
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/diagnostics_cam.ncl"

;==================================================================
;                     USER INTERFACE 
;                   MAIN DRIVER SCRIPT
;==================================================================
  f          = addfile ("/work/b08209033/DATA/IndianMonsoon/ERA5/sparse_grid/sparse_div.nc", "r")
  caseName   = "ERA5"
  diro       = "/home/b08209033/IndianMonsoon/img/NCL_wkSpaceTime/div/"
  latBound   = 15
  spd        = 1
  nDayWin    = 96
  nDaySkip   = -65
  
  plevel   = 100000
  varName  = "divergence"
  varName := varName+" (at "+plevel/100+" hPa)"
  var = f->sd(:,{plevel},{latBound:-latBound},:)

  printVarSummary(var)

  opt      = True
  opt@debug       = True   
  opt@pltType     = "png"        ; send graphics to PNG file
  opt@pltTitle    = caseName+" "+varName+" LOG[Power: "+latBound+"S-"+latBound+"N]"
  opt@Fig_3a      = fspan(0.6, 2.0, 15)
  opt@Fig_3b      = fspan(0.6, 2.0, 15)
  

  wkSpaceTime (var,  diro, caseName, varName             \
              ,latBound, spd , nDayWin, nDaySkip, opt  )