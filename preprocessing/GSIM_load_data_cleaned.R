#GSIM annual streamflow and precip correlation

#The script and preprocessed data originates from this publication:
#Stein, L., Pianosi, F., & Woods, R. (2020). Event-based classification for global study of river flood generating processes. Hydrological Processes, 34(7), 1514-1529.
#code was slightly altered to focus on mean flow instead of annual maximum


library(readr)
library(parallel)
library(zoo)
library(chron)
library(data.table)


file_input_path = "E:/phd/phd_20210618/Data/04_Calculations/mech_compare/"
file_input_path_base = paste0(file_input_path, "v1b", "_")

file_output_path = "C:/Users/stein/Nextcloud/Projects/ISIMIP_modelcomparison/"



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. GSIM homgeneity test info -----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#load information about homogeneity of flow timeseries from GSIM database
yearly_homogeneity <- read_csv("E:/phd/phd_20210618/Data/GSIM/GSIM_updated/GSIM_indices/HOMOGENEITY/yearly_homogeneity.csv")
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
indices_y_f = list.files("C:/Users/stein/Nextcloud/Data/GSIM/GSIM_updated/GSIM_indices/TIMESERIES/yearly", full.names = T)
#https://www.r-bloggers.com/how-to-go-parallel-in-r-basics-tips/
valuesperyear = 350


library(vroom)


s_time = Sys.time()
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
e_time = Sys.time()
e_time-s_time

length(annual_flow_list)

save(annual_flow_list, file = paste0(file_output_path, "annual_flow_list.rdata"))
load(file = paste0(file_output_path, "annual_flow_list.rdata"))



cat_num_vec = unlist(lapply(annual_flow_list, function(x){x[[1]]}))

#exclude suspect stations
cat_suspect <- read_csv("C:/Users/stein/Nextcloud/Data/GSIM_metadata/GSIM_catalog/GSIM_suspect_coordinates_stations.csv", col_select = 1)

cat_keep =!(cat_num_vec %in% c(cat_suspect)$gsim.no)
#all values that were found "suspect" by Gudmundsson et al, were already excluded from calculating the index. 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Merge with precipitation  -----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#vector of catchment shapefile names (without location)
shp_fshort = list.files("C:/Users/stein/Nextcloud/Data/GSIM/GSIM_updated/GSIM_shp/", pattern = ".shp")
shp_fshort = list.files("E:/phd/phd_20210618/Data/GSIM/GSIM_updated/GSIM_metadata/GSIM_catchments", pattern = ".shp")
#MSWEP was extract in order of and for all shapefiles

MSWEP_cat = sub(".shp", "", shp_fshort)
test = cat_num_vec %in% toupper(MSWEP_cat)
cat_num_vec[!test] %in% c(cat_suspect)$gsim.no
#suspect catchments already removed from MSWEP data

#are catchments in the same order for MSWEP and GSIM? yes
length(MSWEP_cat)
length(cat_num_vec[cat_keep])
sum(toupper(MSWEP_cat) == cat_num_vec[cat_keep])

#for each cat calculate mean annual P 

load(file =paste0(file_input_path, "timeseries_eq_dates.Rdata"))
#ind_df
MSWEP_date = ind_df[,1]
MSWEP_years = years(MSWEP_date)
MSWEP_years_num = as.numeric(levels(unique(MSWEP_years)))

load(file = paste0(file_input_path, "timeseries_eq_MSWEP.Rdata"))
#MSWEP_eq_mat

annualP = apply(MSWEP_eq_mat, 2, FUN= function(x){
  aggregate(x, by = list(as.character(MSWEP_years)), sum)[,2]
})

annualP_list = lapply(c(1:ncol(annualP)), function(i){
  Pcol = annualP[,i]
  out = data.frame(date_year = MSWEP_years_num , mean_annual_P = Pcol)
  return(out)
})

annual_flow_list_keep = annual_flow_list[cat_keep]

mean_annual_list = lapply(c(1:length(annualP_list)), function(i){
  temp_flow = annual_flow_list_keep[[i]]
  
  df = tibble(DATE = temp_flow[[2]][,1], MEAN = temp_flow[[2]][,2], n.available = temp_flow[[2]][,3])
  df_cleaned = df %>% 
    mutate(DATE = as.Date(DATE))%>%
    mutate(MEAN = as.numeric(MEAN))%>%
    mutate(n.available = as.numeric(n.available))%>%
    mutate(date_year = as.numeric(substr(DATE, 1,4)))%>%
    mutate(leap_year = leap.year(date_year))%>%
    filter(n.available>=350)%>% #remove years with not enough days contributing
    filter(!(abs(MEAN - median(MEAN)) > 2*sd(MEAN))) #remove outliers. Known issue for several Brazilian catchments
    
  
  temp_P = annualP_list[[i]]
  
  mergedf = merge(df_cleaned, temp_P, by = "date_year")
  out = data.frame(gauge_id = temp_flow[[1]], mean_annualP = mean(mergedf$mean_annual_P), mean_dailyQ = mean(mergedf$MEAN), num_years = nrow(mergedf), na_num = sum(is.na(mergedf$mean_annual_flow)))
  
})
save(mean_annual_list, file = paste0(file_output_path, "mean_annual_list.rdata"))
load(file = paste0(file_output_path, "mean_annual_list.rdata"))

mean_annual_df = do.call(rbind, mean_annual_list)

#mean_annual_df_temp = tibble(mean_annual_df)%>%
#  filter(num_years>10)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Merge with metadata  -----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



GSIM_metadata <- read_csv("E:/phd/phd_20210618/Data/GSIM/GSIM_updated/GSIM_metadata/GSIM_catalog/GSIM_metadata.csv")
#load catchment metadata about catchment characterstics to filter catchments with low quality catchment shape
GSIM_cat_char <- read.csv("E:/phd/phd_20210618/Data/GSIM/GSIM_updated/GSIM_metadata/GSIM_catalog/GSIM_catchment_characteristics.csv")
head(GSIM_metadata)
head(GSIM_cat_char)


# update Australian quality flags -----------------------------------------
library(readxl)
library(rgdal)
library(raster)

#Calculate catchment area
AU_catshp = readOGR(dsn = "E:/phd/phd_20210618/Data/streamflow_countries/Australia/Wasko_cat_boundaries/GIS/HRS_Boundaries_fromSRTM_v0.1_20140326", layer = "HRS_Boundaries_fromSRTM")
AU_catshp$area_sqkm <- area(AU_catshp) / 1000000
AU_catdf = data.frame(AU_catshp)
colnames(AU_catdf) = c("reference.no", "areasqkm")
AU_mergedf = merge(GSIM_metadata, AU_catdf, by = "reference.no")

GSIM_cat_char <- read.csv("E:/phd/phd_20210618/Data/GSIM/GSIM_updated/GSIM_metadata/GSIM_catalog/GSIM_catchment_characteristics.csv")
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
  filter(quality == "High" | quality == "Medium") %>% #remove all catchments with "low" or "caution" quality flag for catchment
  filter(!is.na(mean_dailyQ)) %>%
  mutate(mean_annualQ_mm = mean_dailyQ*1000*60*60*24*365/(area.meta*1000000))

write.csv(mean_annual_df_QF, file = paste0(file_output_path, "GSIM_P_Q_data.csv"))







