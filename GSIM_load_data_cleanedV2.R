#GSIM annual streamflow and precip correlation

#The script was adapted from this publication:
#Stein, L., Pianosi, F., & Woods, R. (2020). Event-based classification for global study of river flood generating processes. Hydrological Processes, 34(7), 1514-1529.
#code was slightly altered to focus on mean flow instead of annual maximum


library(readr)
library(parallel)
library(zoo)
library(chron)
library(data.table)

library(readr)
library(parallel)
library(zoo)
library(chron)
library(data.table)
library(terra)
library(tidyverse)

#file_input_path = "E:/phd/phd_20210618/Data/04_Calculations/mech_compare/"
#file_input_path_base = paste0(file_input_path, "v1b", "_")

file_output_path = "C:/Users/stein/Nextcloud/Projects/ISIMIP_modelcomparison/"



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. GSIM homgeneity test info -----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#load information about homogeneity of flow timeseries from GSIM database
yearly_homogeneity <- read_csv("D:/GSIM/GSIM_updated/GSIM_indices/HOMOGENEITY/yearly_homogeneity.csv")
yearly_homogeneity_max = yearly_homogeneity[c(1:13, 34:37)]

#according to Gudmundsson et al, accept if tests produce NS and not more than one p1. Else reject as unsuitable.
#create a mask (T/F) to exclude inhomogenous values
homogeneity_mask = apply(yearly_homogeneity_max[,14:17], 1, FUN = function(x){
  outval = F
  if (length(grep("NS$",x)) == 4){
    outval = T
  } else if(length(grep("p1",x) == 1)){
    outval = T
  } else {
    outval = F
  }
  outval
})


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get magnitudes of annual mean flow -----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#load information of annual mean flow timeseries from GSIM database
indices_y_f = list.files("D:/GSIM/GSIM_updated/GSIM_indices/TIMESERIES/yearly", full.names = T)
#https://www.r-bloggers.com/how-to-go-parallel-in-r-basics-tips/
valuesperyear = 350


library(vroom)


annual_flow_list = lapply(indices_y_f, function(x){
  #read in catchment index file
  temp = vroom(x, "\t", escape_double = FALSE, trim_ws = TRUE, skip = 21, na = "NA", show_col_types = FALSE, col_select = c(1,2,26), col_types = "ccc")
  #remove commas
  temp = apply(temp, c(1,2), FUN = function(x) gsub(",", "", x))
  
  temp_catnum = vroom_lines(x, n_max = 20)
  #extract catchment identifier from file metadata
  catnum = grep(pattern = "[A-Z]{2}_[0-9]{7}", unlist(strsplit(temp_catnum, ":")), value = T)
  catnum = gsub(" ", "", catnum)
  #remove all values where less than a threshold number of days contributed to the value
  #mask = as.numeric(temp[,3])>= valuesperyear
  
  out = list(catnum, temp)
  return(out)
})


length(annual_flow_list)

save(annual_flow_list, file = paste0(file_output_path, "annual_flow_list.rdata"))
load(file = paste0(file_output_path, "annual_flow_list.rdata"))



cat_num_vec = unlist(lapply(annual_flow_list, function(x){x[[1]]}))

#exclude suspect stations
cat_suspect <- read_csv("D:/GSIM/GSIM_metadata/GSIM_catalog/GSIM_suspect_coordinates_stations.csv", col_select = 1)

cat_not_suspect =!(cat_num_vec %in% c(cat_suspect)$gsim.no)
#all values that were found "suspect" by Gudmundsson et al, were already excluded from calculating the index. 

cat_keep = tibble(mask = homogeneity_mask, cat_not_suspect = cat_not_suspect)%>%
  mutate(keep_checks = ifelse(mask, cat_not_suspect, F))%>%
  pull(keep_checks)
save(cat_keep, file = paste0(file_output_path, "cat_keep.rdata"))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Mean annual precipitation calculation  -----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Read in shapefiles

shp_fshort = list.files("D:/GSIM/GSIM_metadata/GSIM_metadata/GSIM_catchments/", pattern = ".shp")
shp_fall = list.files("D:/GSIM/GSIM_metadata/GSIM_metadata/GSIM_catchments/", pattern = ".shp", full.names = T)

length(cat_num_vec[cat_keep] %in% toupper(sub(".shp", "", shp_fshort)))

length(toupper(sub(".shp", "", shp_fshort)) %in% cat_num_vec[cat_keep])
length(shp_fall[toupper(sub(".shp", "", shp_fshort)) %in% cat_num_vec[cat_keep]])
#indices match
tail(cbind(shp_fshort[toupper(sub(".shp", "", shp_fshort)) %in% cat_num_vec[cat_keep]], cat_num_vec[cat_keep]))


#run for all now
#but if wanted to restrain use: [toupper(sub(".shp", "", shp_fshort)) %in% cat_num_vec[cat_keep]]

GSIM_shps = vect(lapply(shp_fall, vect) ) 

writeVector(GSIM_shps, file = paste0(file_output_path, "GSIM_shps.shp"), overwrite = T)
#simplify shapes for better handling
GSIM_shps_simple = simplifyGeom(GSIM_shps, tolerance=0.01, preserveTopology=TRUE, makeValid=TRUE)
writeVector(GSIM_shps_simple, file = paste0(file_output_path, "GSIM_shps_simple.shp"), overwrite = T)

shp_cat_num_vec = toupper(GSIM_shps_simple$FILENAME)
save(shp_cat_num_vec, file = paste0(file_output_path, "shp_cat_num_vec.rdata"))




#Read in precip raster gswp3
GSIM_shps_simple = vect(paste0(file_output_path, "GSIM_shps_simple.shp"))

list_rast_f = list.files("D:/ISMIP_functionalrelation/gswp3/",pattern = ".nc4", full.names = T)
input_w_r = terra::rast(list_rast_f[[1]])
GSIM_shps_simple_t = terra::project(GSIM_shps_simple, crs(input_w_r))


weight_extract_r_gswp3 = terra::extract(input_w_r, GSIM_shps_simple_t,cells = T, weights = T,  exact = T)



GSIM_P_mean_gswp3_list = lapply(list_rast_f, function(rast_file){
  temp_rast = terra::rast(rast_file)
  temp_r_df = as.data.frame(temp_rast, cells = T)
  temp_extract = left_join(weight_extract_r_gswp3, temp_r_df, by = "cell")
  x <- by(temp_extract[,c(5, 4)], temp_extract[,1], function(x) weighted.mean(x[,1], x[,2]))
  out = do.call("rbind", as.list(x))
  return(out)
})

GSIM_P_mean_gswp3_yearly = do.call(cbind, GSIM_P_mean_gswp3_list)
save(GSIM_P_mean_gswp3_yearly, file = paste0(file_output_path, "GSIM_P_mean_gswp3_yearly.rdata"))

load(file = paste0(file_output_path, "GSIM_P_mean_gswp3_yearly.rdata"))
list_rast_f = list.files("D:/ISMIP_functionalrelation/gswp3/",pattern = ".nc4", full.names = T)
P_gswp_year = as.numeric(sub(".nc4", "", sub("D:/ISMIP_functionalrelation/gswp3/average_GSWP3_", "", list_rast_f)))


annual_P_list_gswp3 = lapply(1:nrow(GSIM_P_mean_gswp3_yearly), function(cat_P){ 
  tibble(annual_P_gswp3 = GSIM_P_mean_gswp3_yearly[cat_P,], date_year = P_gswp_year)
})
save(annual_P_list_gswp3, file = paste0(file_output_path, "annual_P_list_gswp3.rdata"))



#Read in precip raster HadGEM2
GSIM_shps_simple = vect(paste0(file_output_path, "GSIM_shps_simple.shp"))

list_rast_f = list.files("D:/ISMIP_functionalrelation/HadGEM2/",pattern = ".nc4", full.names = T)
input_w_r = terra::rast(list_rast_f[[1]])
GSIM_shps_simple_t = terra::project(GSIM_shps_simple, crs(input_w_r))


weight_extract_r_HadGEM2 = terra::extract(input_w_r, GSIM_shps_simple_t,cells = T, weights = T,  exact = T)



GSIM_P_mean_HadGEM2_list = lapply(list_rast_f, function(rast_file){
  temp_rast = terra::rast(rast_file)
  temp_r_df = as.data.frame(temp_rast, cells = T)
  temp_extract = left_join(weight_extract_r_HadGEM2, temp_r_df, by = "cell")
  x <- by(temp_extract[,c(5, 4)], temp_extract[,1], function(x) weighted.mean(x[,1], x[,2]))
  out = do.call("rbind", as.list(x))
  return(out)
})


GSIM_P_mean_HadGEM2_yearly = do.call(cbind, GSIM_P_mean_HadGEM2_list)
save(GSIM_P_mean_HadGEM2_yearly, file = paste0(file_output_path, "GSIM_P_mean_HadGEM2_yearly.rdata"))

load(file = paste0(file_output_path, "GSIM_P_mean_HadGEM2_yearly.rdata"))
list_rast_f = list.files("D:/ISMIP_functionalrelation/HadGEM2/",pattern = ".nc4", full.names = T)
P_HadGEM2_year = as.numeric(sub(".nc4", "", sub("D:/ISMIP_functionalrelation/HadGEM2/average_HadGEM2_", "", list_rast_f)))


annual_P_list_HadGEM2 = lapply(1:nrow(GSIM_P_mean_HadGEM2_yearly), function(cat_P){ 
  tibble(annual_P_HadGEM2 = GSIM_P_mean_HadGEM2_yearly[cat_P,], date_year = P_HadGEM2_year)
})

save(annual_P_list_HadGEM2, file = paste0(file_output_path, "annual_P_list_HadGEM2.rdata"))




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Merge with precipitation gswp3 -----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#annual_flow_list_keep = annual_flow_list[cat_keep]
#optimised for server

file_output_path = "/home/stein/R_server/ISIMIPmodelcomparison/"

load(file = paste0(file_output_path, "annual_flow_list.rdata"))
load(file = paste0(file_output_path, "cat_keep.rdata"))

load(file = paste0(file_output_path, "annual_P_list_gswp3.rdata"))
load(file = paste0(file_output_path, "annual_P_list_HadGEM2.rdata"))

load(file = paste0(file_output_path, "shp_cat_num_vec.rdata"))


cat_num_vec = unlist(lapply(annual_flow_list, function(x){x[[1]]}))


library(parallel)
numCores = 30

mean_annual_list = mclapply(c(1:length(shp_cat_num_vec)), function(i){ #length(annual_flow_list_keep)
  #print(i)
  temp_cat = shp_cat_num_vec[i]
  temp_cat_flow = which(cat_num_vec == temp_cat)
  
  temp_flow = annual_flow_list[[temp_cat_flow]]
  
  
  df = tibble(DATE = temp_flow[[2]][,1], MEAN = temp_flow[[2]][,2], n.available = temp_flow[[2]][,3])
  df_cleaned = df %>% 
    mutate(DATE = as.Date(DATE))%>%
    mutate(MEAN = as.numeric(MEAN))%>%
    mutate(n.available = as.numeric(n.available))%>%
    mutate(date_year = as.numeric(substr(DATE, 1,4)))%>%
    mutate(leap_year = leap.year(date_year))%>%
    filter(n.available>=350)%>% #remove years with not enough days contributing
    filter(!(abs(MEAN - median(MEAN)) > 2*sd(MEAN))) #remove outliers. Known issue for several Brazilian catchments
  #ignore warning. refers to existing NA
  
  temp_Pgswp3 = annual_P_list_gswp3[[i]]
  temp_PHadGEM2 = annual_P_list_HadGEM2[[i]]
  
  out = left_join(df_cleaned, temp_Pgswp3, by = "date_year")%>%
    left_join(temp_PHadGEM2, by = "date_year")%>%
    mutate(gauge_id = temp_flow[[1]])%>%
    mutate(num_years = sum(!is.na(annual_P_gswp3)))%>%
    filter(!is.na(annual_P_gswp3))%>%
    #filter(!is.na(MEAN))%>% not needed due to left join
    mutate(mean_dailyQ = mean(MEAN, na.rm = T))%>%
    mutate(mean_annual_Pgswp3 = mean(annual_P_gswp3, na.rm = T))%>%
    mutate(mean_annual_PHadGEM2 = mean(annual_P_HadGEM2, na.rm = T))%>%
    mutate(QF_homog_suspect = cat_keep[temp_cat_flow])%>%
    dplyr::select(gauge_id, mean_annual_Pgswp3,mean_annual_PHadGEM2, mean_dailyQ, num_years, QF_homog_suspect)%>%
    slice_head(n = 1)
  return(out)
}, mc.cores = numCores)


save(mean_annual_list, file = paste0(file_output_path, "mean_annual_list.rdata"))



load(file = paste0(file_output_path, "mean_annual_list.rdata"))
mean_annual_df = do.call(rbind, mean_annual_list)
#overwrite of MSWEP data






#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Merge with metadata  -----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



GSIM_metadata <- read_csv("D:/GSIM/GSIM_metadata/GSIM_metadata/GSIM_catalog/GSIM_metadata.csv")
#load catchment metadata about catchment characterstics to filter catchments with low quality catchment shape
GSIM_cat_char <- read.csv("D:/GSIM/GSIM_metadata/GSIM_metadata/GSIM_catalog/GSIM_catchment_characteristics.csv")
head(GSIM_metadata)
head(GSIM_cat_char)

# update Australian quality flags -----------------------------------------
library(readxl)
library(rgdal)
library(raster)

#Calculate catchment area
AU_catshp = readOGR(dsn = "D:/GSIM/Wasko_cat_boundaries/GIS/HRS_Boundaries_fromSRTM_v0.1_20140326", layer = "HRS_Boundaries_fromSRTM")
AU_catshp$area_sqkm <- area(AU_catshp) / 1000000
AU_catdf = data.frame(AU_catshp)
colnames(AU_catdf) = c("reference.no", "areasqkm")
AU_mergedf = merge(GSIM_metadata, AU_catdf, by = "reference.no")

GSIM_cat_char <- read.csv("D:/GSIM/GSIM_metadata/GSIM_metadata/GSIM_catalog/GSIM_catchment_characteristics.csv")
AUind = GSIM_cat_char$gsim.no %in% AU_mergedf$gsim.no

#calculate difference in values
cat_diff = abs(GSIM_cat_char[AUind, "area.est"]-AU_mergedf$areasqkm)/AU_mergedf$areasqkm
AU_QF = unlist(lapply(cat_diff, function(x){
  if((x)<0.05){
    out="High"
  } else if((x)<0.1){
    out="Medium"
  } else if((x)<0.5){
    out="Low"
  } else if ((x)>=0.5){
    out="Caution"
  }
}))
GSIM_cat_char[AUind, "quality"] = AU_QF
GSIM_cat_char[AUind, "area.meta"] = AU_mergedf$areasqkm

# metadata  -----------------------------------------

#known issue
#Change catchment area of station "IN_0000162" to 5050km2 for both "GSIM_metadata.csv" ("area" attribute) and "GSIM_catchment_characteristics.csv" ("area.meta" attribute) files
GSIM_cat_char[which(GSIM_cat_char$gsim.no == "IN_0000162"), "area.meta"] = 5050



mean_annual_df_QF = tibble(merge(GSIM_cat_char[,c("gsim.no", "area.meta", "long.new", "lat.new", "quality")], mean_annual_df, by.x = "gsim.no", by.y = "gauge_id"))%>%
  #filter(quality == "High" | quality == "Medium") %>% #remove all catchments with "low" or "caution" quality flag for catchment
  #filter(!is.na(mean_dailyQ)) %>%
  mutate(mean_annualQ_mm = mean_dailyQ*1000*60*60*24*365/(area.meta*1000000))

write.csv(mean_annual_df_QF, file = paste0(file_output_path, "GSIM_P_Q_data_gswp3.csv"))



