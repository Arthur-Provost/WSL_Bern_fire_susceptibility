

###-------------------------------------------------------------------------###
###                         FOREST FIRE REPORT SCRIPT                       ###
###-------------------------------------------------------------------------###

# This script is associated with the methodological report.
# Author: Arthur PROVOST (arthur.provost@wsl.ch or arthur.provost50@gmail.com)


#* ----
# I) PREPARATION  #####################################################

## 1.1) Preparing workspace ===============================
Sys.setLanguage("en")  # version in English

library(AUC)
library(caret)
library(cartography)
library(data.table)
library(datawizard)
library(ecospat)
library(lme4)
library(magick)
library(mgcv)
library(PresenceAbsence)
library(raster)
library(RColorBrewer)
library(rdacca.hp)
library(sf)
library(stringr)
library(terra)
library(tidyterra)
library(tidyverse)
library(xgboost)

source("functions.r")  # loading functions in workspace




## 1.2) Loading data ===============================
df_train <- read.csv("input/datasets/fires_training.csv", sep=",")  # training fire records for cross-validation
df_test <- read.csv("input/datasets/fires_testing.csv", sep=",")  # testing fire records

vect_water <- vect("input/vectors/water.shp")  # water polygons, lakes
vect_contour <- vect("input/vectors/contour.shp")  # contour of Bern canton

rast_ext <- rast("input/rasters/predictors/aspect_N.tif")  # take aspect N as reference for Bern extent
rast_hillshade <- rast("input/rasters/hillshade.tif")  # hillshade as base layer for predicted maps plots


# Selected variables
rast_aspect_N <- rast("input/rasters/predictors/aspect_N.tif")  ; names(rast_aspect_N) <- "aspect_N"
rast_distBuildLog <- rast("input/rasters/predictors/log_dist_build.tif")  ; names(rast_distBuildLog) <- "log_dist_build"
rast_distRoadsLog <- rast("input/rasters/predictors/log_dist_roads.tif")  ; names(rast_distRoadsLog) <- "log_dist_roads"
rast_evi <- rast("input/rasters/predictors/evi.tif")  ; names(rast_evi) <- "evi"
rast_prec <- rast("input/rasters/predictors/precipitations.tif")  ; names(rast_prec) <- "precipitations_sum"
rast_slope <- rast("input/rasters/predictors/slope.tif")  ; names(rast_slope) <- "slope"
rast_swb <- rast("input/rasters/predictors/swb.tif")  ; names(rast_swb) <- "swb"
rast_dbh_mean <- rast("input/rasters/predictors/tree_bhd_mean.tif")  ; names(rast_dbh_mean) <- "dbh_mean"
rast_dbh_sd <- rast("input/rasters/predictors/tree_bhd_sd.tif")  ; names(rast_dbh_sd) <- "dbh_sd"
rast_stem_density <- rast("input/rasters/predictors/stem_density.tif")  ; names(rast_stem_density) <- "stem_density"





## 1.3) Preparing predictors lists ===============================
# This is how variables have to be put as model function inputs

# Complete nested lists
prd_lst_ALL_GD_HI_VD <- list(ALL=list(rast_aspect_N, rast_dbh_mean, rast_dbh_sd, rast_distBuildLog, rast_distRoadsLog, 
                                      rast_evi, rast_prec, rast_slope, rast_stem_density, rast_swb),
                             GD=list(rast_aspect_N, rast_evi, rast_prec, rast_slope, rast_swb),
                             HI=list(rast_distBuildLog, rast_distRoadsLog),
                             VD=list(rast_dbh_mean, rast_dbh_sd, rast_stem_density))


# Individual nested list
prd_lst_ALL <- list(rast_aspect_N, rast_dbh_mean, rast_dbh_sd, rast_distBuildLog, rast_distRoadsLog, 
                    rast_evi, rast_prec, rast_slope, rast_stem_density, rast_swb)
prd_lst_GD <- list(rast_aspect_N, rast_evi, rast_prec, rast_slope, rast_swb)
prd_lst_HI <- list(rast_distBuildLog, rast_distRoadsLog)
prd_lst_VD <- list(rast_dbh_mean, rast_dbh_sd, rast_stem_density)






#* ----
# II) FOREST FIRE MODELLING  #####################################################
## 2.1) Testing variables for collinearity issues ===============================
# This step consists of variables testing: collinearity and multicollinearity
# More can be tested with the following function, check function options.
predictor_selection(nested_compare_lists = list(prd_select_ALL = prd_lst_ALL),
                    df_fires = rbind(df_train, df_test),
                    cor_plot = TRUE, 
                    VIF = TRUE)

# No issues with collinearity and multicollinearity (VIF) < 4, which is good.






## 2.2) Response curves ===============================
# Declaring label names for plotting
predictors_nm <- data.frame(rast_nm = unlist(lapply(prd_lst_ALL, names)),
                            lab_nm = c("GD-Northness", "GD-EVI", "GD-Precipitations", "GD-Slope", "GD-SWB",
                                       "HI-Distance buildings", "HI-Distance roads", "VD-DBH mean", "VD-DBH sd", "VD-Stem density"))

response_curves(df_fire_choice = df_train,
                list_predictor = prd_lst_ALL,
                lab_predictors = predictors_nm,
                model_type = c("glm","gam"),
                poly.glm.k = c(2,4),
                s.gam = c(3,8),
                plot.points = TRUE)

sdogusdoöigjh
sdfghjsdföghjoöi
dfhjdföhjtroöiju




## 2.3) Fire model ===============================
# The following function perfoms the whole modelling framework
# Notice: the option num_rep is set up to 10 but it can take several hours
# nested_list option can be changed e.g. for tests with individual nested lists like prd_lst_GD
# Check the function.R file for complementary information
# Make sure to have an existing output directory
fire_mod_multi_list(nested_list = prd_lst_ALL_GD_HI_VD,
                    backgrd_pts = 10000,
                    df_train = df_train,
                    df_test = df_test,
                    models = c("glm","gam","gbm"),
                    simple_complex = TRUE,
                    num_rep = 10,
                    save_mod_rds = TRUE,
                    output_source_dir = "output/models/1_ensemble/")








## 2.4) Variables importance ===============================
# This section is based on variation partitioning 
# Reference: Lai (2022): Generalizing hierarchical and variation partitioning in multiple regression and canonical analyses using the rdacca.hp R package

# Preparing the dataset with pres/pseudo-abs
df_occ_var_ALL <- prep_df_occ_var(rbind(df_train, df_test), 
                                  prd_lst_ALL, backgrd_pts = 10000, na.rm=TRUE)

rast_fire_risk <- rast("output/models/1_ensemble/ALL/rast_ALL_ENS.tif")  # loading danger map
df_occ_var_ALL <- df_occ_var_ALL |>     # adding predicted danger value per pres/pseudo-abs
  mutate(risk = terra::extract(rast_fire_risk, df_occ_var_ALL[,c("x","y")])) |>
  drop_na()

risk <- as.data.frame(df_occ_var_ALL$risk$ENS)
env <- df_occ_var_ALL[,c(4:(ncol(df_occ_var_ALL)-1))]

env_gd <- env[ ,names(env) %in% c("aspect_N", "evi", "precipitations", "slopee", "swb")]
env_hi <- env[ ,names(env) %in% c("log_dist_roads", "log_dist_build")]
env_vd <- env[ ,names(env) %in% c("dbh_mean", "dbh_sd", "stem_density")]



## Performing hierarchical partitioning
vp_sgl <- as.data.frame(rdacca.hp(risk, env, method="RDA", type="adjR2", var.part=TRUE)$Hier.part)
vp_group <- as.data.frame(rdacca.hp(risk, list(GD=env_gd, HI=env_hi, VD=env_vd), method="RDA", type="adjR2", var.part=TRUE)$Hier.part)



## Plotting results
# Single predictors
vp_sgl <- vp_sgl |> rename(perc = `I.perc(%)`) |>
  mutate(var = rownames(vp_sgl))

ggplot(vp_sgl, aes(x=reorder(var, perc), y=perc)) + 
  geom_bar(stat = "identity", width=.5) +
  coord_flip() +
  xlab("") + ylab('Contribution') +
  theme(axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
        axis.title = element_text(size = 16))



# Predictor classes
vp_group <- vp_group |> rename(perc = `I.perc(%)`) |>
  mutate(var = rownames(vp_group)) |>
  arrange(desc(perc))

ggplot(vp_group, aes(x=reorder(var, perc), y=perc)) + 
  geom_bar(stat = "identity", width=.5) +
  coord_flip() +
  xlab("") + ylab('Contribution') +
  theme(axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 18),
        axis.title = element_text(size = 16))










## 2.5) Fire return time ===============================
rast_fire_risk <-  rast("output/models/1_ensemble/ALL/rast_ALL_ENS.tif")
df_fire <- rbind(df_train, df_test)


## Finding the right thresholds
fire_prob_summary <- as.data.frame(matrix(ncol=4, nrow=300, dimnames=list(NULL,c("thr_low","thr_high","fire_prob_pix","fire_prob_sum"))))
k <- 1
fire_prob_sum <- 0
for(i in seq(0, 0.99, 0.01)){  # scanning narrow thresholds for better precision
  thr_low <- i
  thr_high <- i + 0.01
  
  ## Calculating return time/threshold
  fire_prob <- fire_return_time(raster_FFS = rast_fire_risk,
                                fire_dataset = df_fire,
                                thr_low = thr_low,
                                thr_high = thr_high,
                                nb_yr = 7)
  
  fire_prob_sum <- fire_prob_sum + as.numeric(fire_prob)
  
  fire_prob_summary$thr_low[k] <- thr_low
  fire_prob_summary$thr_high[k] <- thr_high
  fire_prob_summary$fire_prob_pix[k] <- fire_prob
  fire_prob_summary$fire_prob_sum[k] <- fire_prob_sum
  
  k <- k+1
}
fire_prob_summary <- fire_prob_summary |> na.omit()
fire_prob_summary$fire_prob_sum <- formatC(fire_prob_summary$fire_prob_sum, format = "e", digits = 2)  # scientific writing

## The thresholds are:
# from 0 to 0.15: return time of 1/10000
# from danger thr 0.15 to 0.29: return time of 1/1000,
# from 0.29 to 0.66: return time of 1/100
# above 0.66,return time > 1/100




### Classifying raster
levels_ffs <- c(0, 0.15, 0.29, 0.66, 1)
labels_ffs <- c("below 1e-4", "1e-4 to 1e-3", "1e-3 to 1e-2", "above 1e-2")

rast_returnTm <- rast_fire_risk
rast_returnTm <- classify(rast_returnTm, rcl=levels_ffs)
levels(rast_returnTm) <- labels_ffs


## Plotting
par(mfrow=c(1,1), mar=c(0,0,0,10))
plot(rast_hillshade, col = grey(seq(0, 1, length.out=100)), alpha=0.5, legend=FALSE, axes=FALSE)
plot(rast_returnTm, col=c("#0f23b6", "#2e7d32", "#fdd835", "#da4108"), alpha = 1, add=TRUE)
plot(vect_contour, add=TRUE)
plot(vect_water, col="#c3ebf3", lwd=0.7, add=TRUE)










## 2.6) Site sensibility to environment and human influence ===============================
# RGB plot with GD and HI only
rast_ffs_GD <- rast("output/models/1_ensemble/GD/rast_GD_ENS.tif")
rast_ffs_HI <- rast("output/models/1_ensemble/HI/rast_HI_ENS.tif")
rast_ffs_VD <- rast_ffs_GD  # tmp file
values(rast_ffs_VD) <- 0  # mute VD

rast_veg_mask <- rast_stem_density  # vegetation layer as forest mask
rast_R <- rast_ffs_GD  ; names(rast_R) <- "gd"
rast_G <- rast_ffs_HI  ; names(rast_G) <- "hi"

# Categorize rasters
rast_R <- terra::ifel(rast_R < 0.33, 10, ifel(rast_R < 0.67, 20, 30))
rast_G <- terra::ifel(rast_G < 0.33, 1, ifel(rast_G < 0.67, 2, 3))
rast_RG <- rast_R + rast_G  # combine the 2 rasters

levels_rast_RG <- data.frame(c(11,12,13,21,22,23,31,32,33), c(1:9))
labels_rast_RG <- data.frame(from=c(1:9), to=as.character(levels_rast_RG[,1]))

rast_RG <- classify(rast_RG, rcl=levels_rast_RG)  # categorizing
levels(rast_RG) <- labels_rast_RG


# Plotting results (widening plotting windows to better see)
#png(paste0("output/images/gd_hi_influence.png"), width=20, height=15, unit="cm", res=300, pointsize=15)
rgb_corresp <- magick::image_read("input/images/rgb_plot_col.png")  # loading png of RGB corresponding colors
plot(rast_RG, legend=F, col=c("#EEEEEE","#FFF77F","#FFED00","#7BA9FF","#a9a9a9","#A69B00","#0059FF","#00328F","#473810"),
     xlim=c(xmin(rast_RG), 1.4*xmax(rast_RG)))
rasterImage(rgb_corresp, xleft=0.989*xmax(rast_RG), xright=1.01*xmax(rast_RG),
            ybottom=0.96*ymax(rast_RG), ytop=ymax(rast_RG), xpd=T)
#dev.off()








#* ----
# III) MANAGEMENT OPTIONS #####################################################
## 3.1) Dominant species and habitat classes maps ===============================
### i) Dominant species maps ------------
# This is the reference script but the rasters are already saved in "input/rasters/dominant_sp/"

rast_idLat <- rast("input/rasters/forest_id_lat.tif")
df_idLat_domSp <- read.csv("input/datasets/nais_idLat_domSp.csv")

sp_select <- c("Abies alba", "Fagus sylvatica", "Picea abies", "Pinus sylvestris", "Quercus petraea", "Quercus robur")

# Combining raster and dataset
lvl_domSp <- as.data.frame(levels(rast_idLat)) |> 
  left_join(df_idLat_domSp[,c("ID_lat", "full_name")], by="ID_lat") |>
  rename(dom_sp = full_name) |> 
  na.omit() |> 
  group_by(dom_sp) |>
  dplyr::filter(dom_sp %in% sp_select) |> 
  unique()


for (sp in sp_select){
  val_select <- lvl_domSp$value[which(lvl_domSp$dom_sp == sp)]  # extract corresponding levels
  rast_domSp_i <- segregate(rast_idLat, class = val_select)  # extract all veg types related to the species
  rast_domSp_i <- sum(rast_domSp_i)  # combine all the layers (no pb because no overlap)
  names(rast_domSp_i) <- sp

  #writeRaster(rast_domSp_i, paste0("input/rasters/dominant_sp/",sp,".tif"), overwrite=TRUE)
  
  # Plotting images
  png(paste0("output/images/dominant_sp/",sp,".png"), width=15, height=15, unit="cm", res=300, pointsize=15)
  plot(rast_hillshade, col=grey(seq(0, 1, length.out=100)), alpha=0.2, legend=FALSE, axes=FALSE, main=sp, cex.main=1.5)
  plot(rast_domSp_i, legend=FALSE, add=TRUE)
  dev.off()
}




### ii) Habitat maps - Fagus sylvatica ------------
# Loading data
df_nais_idLat_domSp <- read.csv("input/datasets/nais_idLat_domSp.csv")
df_nais_typoCH_fagus <- read.csv("input/datasets/nais_TYPOCH_Fagus.csv")
rast_nais <- rast("input/rasters/forest_nais.tif")


# Combining datasets
df_fagus <- df_nais_idLat_domSp |>
  dplyr::filter(dom_sp == "Fag_syl") |>
  left_join(df_nais_typoCH_fagus, by="NAIS") |> 
  dplyr::select(dom_sp, NAIS, TYPOCH)



# Keeping levels corresponding to Fagus
rast_fagus <- droplevels(rast_nais,
                           level = setdiff(levels(rast_nais)[[1]]$value,   # level: cat to remove, difference between all NAIS categories and the one of Fagus
                                           which(levels(rast_nais)[[1]]$NAIS %in% df_fagus$NAIS)))
rast_fagus <- droplevels(rast_fagus, level = 97)


# Combine raster levels and fagus data
lvl_fagus <- levels(rast_fagus)[[1]]|> 
  left_join(df_fagus[c("NAIS","TYPOCH")], by="NAIS") |>
  dplyr::select(value, TYPOCH)

levels(rast_fagus) <- lvl_fagus  # change raster levels


# Combining duplicated levels (raster > dataframe > raster)
df_fagus <- as.data.frame(rast_fagus, xy=T)
rast_fagus <- tidyterra::as_spatraster(df_fagus, xycols = 1:2, crs = crs(rast_ext), digits = 6)
rast_fagus <- project(rast_fagus, rast(ext=ext(rast_ext), crs=crs(rast_ext), res=100), method="near")

plot(rast_fagus)



writeRaster(rast_fagus, "output/rasters/typoCH/typoCH_Fag_syl.tif", overwrite=TRUE)







### iii) Habitat maps - Picea abies ------------
# Loading data
df_nais_idLat_domSp <- read.csv("input/datasets/nais_idLat_domSp.csv")
df_nais_typoCH_picea <- read.csv("input/datasets/nais_TYPOCH_Picea.csv")
rast_nais <- rast("input/rasters/forest_nais.tif")


# Combining datasets
df_picea <- df_nais_idLat_domSp |>
  dplyr::filter(dom_sp == "Pic_abi") |>
  left_join(df_nais_typoCH_picea, by="NAIS") |> 
  dplyr::select(dom_sp, NAIS, TYPOCH) |> na.omit()



# Keeping levels corresponding to Picea
rast_picea <- droplevels(rast_nais,
                         level = setdiff(levels(rast_nais)[[1]]$value,   # level: cat to remove, difference between all NAIS categories and the one of Picea
                                         which(levels(rast_nais)[[1]]$NAIS %in% df_picea$NAIS)))


# Combine raster levels and picea data
lvl_picea <- levels(rast_picea)[[1]]|> 
  left_join(df_picea[c("NAIS","TYPOCH")], by="NAIS") |>
  dplyr::select(value, TYPOCH)

levels(rast_picea) <- lvl_picea  # change raster levels


# Combining duplicated levels (raster > dataframe > raster)
df_picea <- as.data.frame(rast_picea, xy=T)
rast_picea <- tidyterra::as_spatraster(df_picea, xycols = 1:2, crs = crs(rast_ext), digits = 6)
rast_picea <- project(rast_picea, rast(ext=ext(rast_ext), crs=crs(rast_ext), res=100), method="near")

plot(rast_picea)



writeRaster(rast_picea, "output/rasters/typoCH/typoCH_Pic_abi.tif", overwrite=TRUE)











## 3.2) GLMM response curves ===============================
# This step is done for 2 species: Fagus sylvatica & Picea abies
# Random effects attributed to TYPOCH forest classes
# We assume here that TYPOCH classes catches environmental conditions

### i) Preparing data for all species ------------
df_fire <- rbind(df_train, df_test)  # combining fire datasets

# Declaring the formula
formula_glmm <- occ ~ dbh_mean + I(dbh_mean^2) + dbh_sd + I(dbh_sd^2) + stem_density + I(stem_density^2) +
  log_dist_roads + I(log_dist_roads^2) + log_dist_build + I(log_dist_build^2) + (1|TYPOCH)

# Preparing dataframe with avg value of each variable
# values are then changed in the glmm response curve plot loop
predictors_noTYPOCH <- list(rast_dbh_mean, rast_dbh_sd, rast_stem_density, 
                            rast_distRoadsLog, rast_distBuildLog)  # list without typoCH
predictors_noTYPOCH_rescale <- lapply(X=predictors_noTYPOCH, function(X) raster_rescale(X, new.min = 0, new.max = 100)) # rescaling predictors to match GLMM model

predictor_values <- values(rast(predictors_noTYPOCH_rescale))
var_stat <- apply(predictor_values, 2, quantile, probs = c(0,.5,1), na.rm = TRUE)  # calculates probabilities per col : minima, maxima, and medians
var_raw <- as.data.frame(var_stat[rep(2,200),])  # take mean of predictor values

var_raw <- cbind(var_raw, rep(NA,200))  # add a TYPOCH class for glmm model (doesn't matter which class)
colnames(var_raw)[ncol(var_raw)] <- "TYPOCH"
var_raw$TYPOCH <- as.factor(var_raw$TYPOCH)

# Prepare variable names for labelling plots
predictors_nm <- data.frame(pred_nm = colnames(var_raw), 
                            lab_nm = c("DBH mean", "DBD sd", "Stem density", "Distance buildings", "Distance roads", "TYPOCH"))





### ii) Fagus sylvatica ------------
## Preparing data
rast_typoCH_fag <- rast("output/rasters/typoCH/typoCH_Fag_syl.tif")
predictors_fag <- list(rast_dbh_mean, rast_dbh_sd, rast_stem_density,
                       rast_distRoadsLog, rast_distBuildLog,
                       rast_typoCH_fag)

df_fire <- rbind(df_train, df_test)  # combining fire datasets
df_var_fag <- prep_df_occ_var(df_fire, predictors_fag, backgrd_pts = 10000, na.rm=TRUE)  # prep dataframe for modelling
df_var_fag[c(4:8)] <- datawizard::rescale(df_var_fag[c(4:8)], to=c(0,100))  ## Rescaling predictors from 0 to 100 (better for GLMM)
wts_fag <- model_weights(df_fire, predictors_fag, backgrd_pts = 10000)  # weights for class imbalance


## GLMM model
glmm_fag <- glmer(formula_glmm, data = df_var_fag, family = "binomial", weights = wts_fag)





## Finding HI values with small influence on Fagus sylvatica fires
for(i in 1:length(predictors_noTYPOCH_rescale)){
  var_i <- var_raw
  var_i[,i] <- as.data.frame(seq(var_stat[1,i], var_stat[3,i], length.out = 200))  # prepare sequence of variable values from min to max
  prd_fire_i <- predict(glmm_fag, newdata = var_i, re.form = NA, allow.new.levels = TRUE, type = "response")  # prediction for Fagus glmm model
  print(paste0("min(", colnames(var_i)[i], ") = ", which(prd_fire_i == min(prd_fire_i))[[1]] * 100 / 200))
}
# CCL: min(log_dist_roads) = 62, min(log_dist_build) = 70.5  (can change a little)




## Response curve
# "Muting" HI
var_raw$log_dist_roads <- 62
var_raw$log_dist_build <- 70.5

# Attributing colors to habitats
lvl_fag <- levels(rast_typoCH_fag)[[1]]$TYPOCH
lvl_fag <- c("Galio-Fagenion", "Cephalanthero-Fagenion", "Lonicero-Fagenion", "Abieti-Fagenion", "Luzulo-Fagenion")  # changing order for better legend
col_lvl <- rev(c("#ffeda0", "#feb24c", "#fc4e2a", "#b10026", "#6e0018"))


par(mfrow=c(1,3), mar=c(10,1,1,1), oma=c(1,1,1,1), cex=2)  # Prepare plotting window
for(i in 1:3){  # 1:3 for the VD var
  
  var_stat_noRescale <- apply(values(predictors_noTYPOCH[[i]]), 2, quantile, probs = c(0,.5,1), na.rm = TRUE)  # for plotting with correct x scale
  var_i_noRescale <- as.data.frame(seq(var_stat_noRescale[1], var_stat_noRescale[3], length.out = 200))  # prepare sequence of variable values from min to max
  
  plot(1,1, ylim=c(0,1) ,xlim=var_stat_noRescale[c(1,3)], type = "n", 
       ylab = "", 
       xlab = predictors_nm$lab_nm[which(predictors_nm$pred_nm == colnames(var_raw)[i])], 
       cex.lab=1, cex.axis=0.7)
  
  for(forest in lvl_fag){
    var_i <- var_raw
    var_i[,i] <- as.data.frame(seq(var_stat[1,i], var_stat[3,i], length.out = 200))  # prepare sequence of variable values from min to max
    var_i$TYPOCH <- forest
    var_i$TYPOCH <- as.factor(var_i$TYPOCH)
    
    prd_fire_i <- predict(glmm_fag, newdata = var_i, type = "response")
    
    lines(x=var_i_noRescale[[1]], y=prd_fire_i,
          col=col_lvl[which(lvl_fag == forest)], lwd=2.5)
  }
}
# Adding legend and title to plot
par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
plot(0, -4, type = 'l', bty = 'n', xaxt = 'n', yaxt = 'n')
legend("bottom", inset = 0,
       legend = lvl_fag, col = col_lvl, lwd = 2.5, bty = "n", cex = 1,
       xpd = TRUE)
mtext("Fagus sylvatica - GLMM", side=3, line=-1.5, cex=2, outer=FALSE)




## Predicting fire danger map based on Fagus vegetation structure
# Fixing HI var to values of smallest influence
rast_distRoadsLog2 <- rast_distRoadsLog
values(rast_distRoadsLog2) <- 62
rast_distBuildLog2 <- rast_distBuildLog
values(rast_distBuildLog2) <- 70.5

# Rescale the VD rasters from 0 to 100 to fit the model
prep_predict_fag <- rast(c(lapply(X=list(rast_dbh_mean, rast_dbh_sd, rast_stem_density),
                                     function(X) raster_rescale(X, new.min = 0, new.max = 100)),
                            rast_distRoadsLog2, rast_distBuildLog2,
                            rast_typoCH_fag))

# Predicting danger map
prd_fag_plot <- predict(prep_predict_fag, glmm_fag, type = "response", na.rm=T)

# Plotting
par(mfrow=c(1,1), oma=c(0,2,1,4))
plot(rast_hillshade, col=grey(seq(0, 1, length.out=100)), legend=FALSE, axes=FALSE, main = "Fagus sylvatica - GLMM \nDanger~VD", cex.main=2)
plot(prd_fag_plot, range = c(0,1), col=colorRampPalette(c(brewer.pal(n=9, name="OrRd")))(100), alpha=0.9, axes=FALSE, plg=list(cex=2), add=TRUE)
plot(vect_contour, add=TRUE)
plot(vect_water, col="#c3ebf3", lwd=0.7, add=TRUE)







### iii) Picea abies ------------
## Preparing data
rast_typoCH_pic <- rast("output/rasters/typoCH/typoCH_Pic_abi.tif")

predictors_pic <- list(rast_dbh_mean, rast_dbh_sd, rast_stem_density,
                       rast_distRoadsLog, rast_distBuildLog,
                       rast_typoCH_pic)

df_fire <- rbind(df_train, df_test)  # combining fire datasets
df_var_pic <- prep_df_occ_var(df_fire, predictors_pic, backgrd_pts = 10000, na.rm=TRUE)  # prep dataframe for modelling
df_var_pic[c(4:8)] <- datawizard::rescale(df_var_pic[c(4:8)], to=c(0,100))  ## rescaling predictors from 0 to 100 (better for GLMM)
wts_pic <- model_weights(df_fire, predictors_pic, backgrd_pts = 10000)  # weights for class imbalance



## GLMM model
glmm_pic <- glmer(formula_glmm, data = df_var_pic, family = "binomial", weights = wts_pic)



## Finding HI values with small influence on fires on Picea abies
for(i in 1:length(predictors_noTYPOCH_rescale)){
  var_i <- var_raw
  var_i[,i] <- as.data.frame(seq(var_stat[1,i], var_stat[3,i], length.out = 200))  # prepare sequence of variable values from min to max
  prd_fire_i <- predict(glmm_pic, newdata = var_i, re.form = NA, allow.new.levels = TRUE, type = "response")
  print(paste0("min(", colnames(var_i)[i], ") = ", which(prd_fire_i == min(prd_fire_i))[[1]] * 100 / 200))
}
# CCL: min(log_dist_roads) = 57, min(log_dist_build) = 60





## Response curve
# "Muting" HI
var_raw$log_dist_roads <- 57
var_raw$log_dist_build <- 60

lvl_pic <- levels(rast_typoCH_pic)[[1]]$TYPOCH
lvl_pic <- c("Abieti-Fagenion", "Abieti-Piceion", "Vaccinio-Piceion", "Sphagno-Piceetum")  # changing order for better legend
col_lvl <- rev(c("#ffeda0", "#feb24c", "#fc4e2a", "#b10026"))


par(mfrow=c(1,3), mar=c(10,1,1,1), oma=c(1,1,1,1), cex=2)  # Prepare plotting window
for(i in 1:3){  # 1:3 for the VD var
  
  var_stat_noRescale <- apply(values(predictors_noTYPOCH[[i]]), 2, quantile, probs = c(0,.5,1), na.rm = TRUE)  # for plotting with correct x scale
  var_i_noRescale <- as.data.frame(seq(var_stat_noRescale[1], var_stat_noRescale[3], length.out = 200))  # prepare sequence of variable values from min to max
  
  plot(1,1, ylim=c(0,1) ,xlim=var_stat_noRescale[c(1,3)], type = "n", 
       ylab = "", 
       xlab = predictors_nm$lab_nm[which(predictors_nm$pred_nm == colnames(var_raw)[i])], 
       cex.lab=1, cex.axis=0.7)
  
  for(forest in lvl_pic){
    var_i <- var_raw
    var_i[,i] <- as.data.frame(seq(var_stat[1,i], var_stat[3,i], length.out = 200))  # prepare sequence of variable values from min to max
    var_i$TYPOCH <- forest
    var_i$TYPOCH <- as.factor(var_i$TYPOCH)
    
    prd_fire_i <- predict(glmm_pic, newdata = var_i, type = "response")
    
    lines(x=var_i_noRescale[[1]], y=prd_fire_i,
          col=col_lvl[which(lvl_pic == forest)], lwd=2.5)
  }
}
# Adding legend and title to plot
par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
plot(0, -4, type = 'l', bty = 'n', xaxt = 'n', yaxt = 'n')
legend("bottom", inset = 0,
       legend = lvl_pic, col = col_lvl, lwd = 2.5, bty = "n", cex = 1,
       xpd = TRUE)
mtext("Picea abies - GLMM", side=3, line=-1.5, cex=2, outer=FALSE)






## Predicting fire danger map based on Picea vegetation structure
# Fixing HI var to values of smallest influence
rast_distRoadsLog2 <- rast_distRoadsLog
values(rast_distRoadsLog2) <- 57
rast_distBuildLog2 <- rast_distBuildLog
values(rast_distBuildLog2) <- 60

# Rescale the VD rasters from 0 to 100 to fit the model
prep_predict_pic <- rast(c(lapply(X=list(rast_dbh_mean, rast_dbh_sd, rast_stem_density),
                                  function(X) raster_rescale(X, new.min = 0, new.max = 100)),
                           rast_distRoadsLog2, rast_distBuildLog2,
                           rast_typoCH_pic))

# Predicting danger map
prd_pic_plot <- predict(prep_predict_pic, glmm_pic, type = "response", na.rm=T)

# Plotting
par(mfrow=c(1,1), oma=c(0,2,1,4))
plot(rast_hillshade, col=grey(seq(0, 1, length.out=100)), legend=FALSE, axes=FALSE, main = "Picea abies - GLMM \nDanger~VD", cex.main=2)
plot(prd_pic_plot, range = c(0,1), col=colorRampPalette(c(brewer.pal(n=9, name="OrRd")))(100), alpha=0.9, axes=FALSE, plg=list(cex=2), add=TRUE)
plot(vect_contour, add=TRUE)
plot(vect_water, col="#c3ebf3", lwd=0.7, add=TRUE)








## 3.3) Shapley values ===============================
### i) XGB parameter tuning ------------

# The goal is to find the best eta and gamma values for XGB model (long process)
# Fully based on cross validation process
# Boyce is the most important metric

# Data preparation
prd_list <- prd_lst_ALL
bck_pts <- 10000

df_occ_var_train_xgb <- prep_df_occ_var(df_train, prd_list, bck_pts, na.rm=TRUE)
df_occ_var_test_xgb <- prep_df_occ_var(df_test, prd_list, bck_pts, na.rm=TRUE)
xgb_rsp_var_train <- as.matrix(df_occ_var_train_xgb[,4:ncol(df_occ_var_train_xgb)])
xgb_rsp_var_test <- as.matrix(df_occ_var_test_xgb[,4:ncol(df_occ_var_test_xgb)])

eta_val <- c(1, 0.1, 0.01, 1e-3, 1e-4)
gamma_val <- c(1, 10, 100)

# Preparing dataframe to put assessment
df_assess <- cbind(expand.grid(eta_val, gamma_val), NA, NA, NA, NA)  # expand grid for all combinations of values
colnames(df_assess) <- c("eta", "gamma", "kappa", "tss", "auc", "boyce")

# Run assessment
for (eta in eta_val){
  for (gamma in gamma_val){
    
    res_xgb <- xgboost(data = xgb_rsp_var_train, 
                       label = df_occ_var_train_xgb$occ, 
                       nround = 5000,  # important to have a high value (longer)
                       nthread = 10,
                       verbose = 0,  # do not print messages
                       scale_pos_weight = (length(which(df_occ_var_train_xgb$occ == 0)) / length(which(df_occ_var_train_xgb$occ == 1))), 
                       objective="binary:logistic",
                       eta = eta,
                       gamma = gamma)
    
    xgb_prd <- predict(res_xgb, newdata = xgb_rsp_var_test)
    
    ## Assessment
    df_prd <- data.frame(ID = 1:nrow(df_occ_var_test_xgb), 
                         Obs = df_occ_var_test_xgb$occ,  # test df occurrences
                         Prd = as.data.frame(xgb_prd))  # model predictions
    
    prd_thres <- optimal.thresholds(df_prd)  # metrics thresholds
    cm_prd <- cmx(df_prd , threshold = prd_thres[4,2])  # binarization with MaxKappa ([4,2])
    eval_kappa <- kappa(cm_prd)  # calculates kappa based on selected threshold
    eval_tss <- tss(cm_prd)   
    eval_auc <- AUC::auc(roc_wrap(df_prd)[[1]])
    eval_boyce <- boyce_wrap(df_prd)$cor
    
    
    
    df_assess$kappa[which(df_assess$eta == eta & df_assess$gamma == gamma)] <- eval_kappa
    df_assess$tss[which(df_assess$eta == eta & df_assess$gamma == gamma)] <- eval_tss
    df_assess$auc[which(df_assess$eta == eta & df_assess$gamma == gamma)] <- eval_auc
    df_assess$boyce[which(df_assess$eta == eta & df_assess$gamma == gamma)] <- eval_boyce
    
    #print(df_assess)
  }
}

df_assess  

# CCL: based on the the Boyce results, the best compromise is eta = 0.001 and gamma = 100






### ii) Saving XGB models ------------
## Save RDS xgb model for the 4 predictors' lists
# then faster to just reopen the models

df_fires <- rbind(df_train, df_test)
for (i in c("ALL", "GD", "HI", "VD")){
  print(i)
  prd_list <- prd_lst_ALL_GD_HI_VD[[i]]
  df_occ_var <- prep_df_occ_var(df_fires, prd_list, 10000, na.rm=TRUE)  # taking train + test datasets
  
  res_xgb <- xgboost(data = as.matrix(df_occ_var[,4:ncol(df_occ_var)]),  # covariates
                     label = df_occ_var$occ,  # var to explain
                     nround = 5000,
                     nthread = 10,  
                     verbose = 0,  # do not print messages
                     scale_pos_weight = (length(which(df_occ_var$occ == 0)) / length(which(df_occ_var$occ == 1))),  # replace wts
                     objective = "binary:logistic",
                     eta = 0.001,  # very slow learning rate to catch interaction
                     gamma = 100)
  
  write_rds(res_xgb, paste0("output/models/2_xgb/res_xgb_", i, ".rds"))
}








### iii) Shapley - ALL, GD, HI, VD ------------
# Based on XGB models, Shapley values are performed on ALL, GD, HI and VD
# Images are saved as .png files in output folder

df_fires <- rbind(df_train, df_test)
for (i in c("ALL", "GD", "HI", "VD")){
  print(i)
  prd_list <- prd_lst_ALL_GD_HI_VD[[i]]
  df_occ_var <- prep_df_occ_var(df_fires, prd_list, 10000, na.rm=TRUE)
  res_xgb <- readRDS(paste0("output/models/2_xgb/res_xgb_", i, ".rds"))  # load saved model
  
  ### Relative importance
  shap_rank <- shap.score.rank(xgb_model = res_xgb,
                               X_train = as.matrix(df_occ_var[,4:ncol(df_occ_var)]),
                               shap_approx = F)
  
  png(paste0("output/images/shapley/models/shapeley_relatImp_",i,".png"), width=15, height=15, unit="cm", res=300, pointsize=7.5)
  print(shap.var.importance(shap_rank, top_n = length(prd_list), title = paste0(i, " model")))
  dev.off()
  
  
  ### Var influence & importance
  shap_prep <- shap.prep(shap = shap_rank,
                         X_train = as.matrix(df_occ_var[,4:ncol(df_occ_var)]),
                         top_n = length(prd_list))
  
  png(paste0("output/images/shapley/models/shapeley_summary_",i,".png"), width=15, height=15, unit="cm", res=300, pointsize=7.5)
  print(plot.shap.summary(data_long = shap_prep, title = paste0(i, " model")))
  dev.off()
  
  
  ### Dependency plots
  png(paste0("output/images/shapley/models/shapeley_DP_",i,".png"), width=15, height=15, unit="cm", res=300, pointsize=7.5)
  xgb.plot.shap(data = as.matrix(df_occ_var[,4:ncol(df_occ_var)]), # input data
                model = res_xgb,
                features = res_xgb$feature_names, # only top 10 var
                n_col = 3, # layout option
                plot_loess = T, # add red line to plot
                cex.lab=2)
  dev.off()
}







### iv) Shapley - Dom sp ------------
# Calculates Shapeley values per dominant species
# Just reuse the existing XGB ALL model to predict on species map

res_xgb_ALL <- readRDS(paste0("output/models/2_xgb/res_xgb_ALL.rds"))  # load full model
sp_select <- c("Abies alba", "Fagus sylvatica", "Picea abies", "Pinus sylvestris", "Quercus petraea", "Quercus robur")
prd_list <- prd_lst_ALL
df_fires <- rbind(df_train, df_test)

for (sp in sp_select){
  print(sp)
  rast_sp_i <- rast(paste0("input/rasters/dominant_sp/", sp,".tif"))  # load map of sp range
  prd_list_msk <- lapply(X=prd_list, Y=rast_sp_i, FUN=function(X,Y) mask(X,Y))  # masking all rasters to species range
  xgb_rsp_var_sp_i <- as.matrix(as.data.frame(rast(prd_list_msk)))  # extracting predictor values
  
  df_occ_var <- prep_df_occ_var(df_fires, prd_list_msk, 100, na.rm=TRUE)  # only for dependency plot ticks
  df_occ_var <- df_occ_var[which(df_occ_var$occ == 1), 4:ncol(df_occ_var)]  # only take values for occ points
  
  if(nrow(xgb_rsp_var_sp_i) > 10000){  # sample if too many pixels
    xgb_rsp_var_sp_i <- xgb_rsp_var_sp_i[sample(nrow(xgb_rsp_var_sp_i),10000), ]
  }
  
  ## Var influence & importance
  shap_rank <- shap.score.rank(xgb_model = res_xgb_ALL, 
                               X_train = xgb_rsp_var_sp_i,
                               shap_approx = F)
  
  png(paste0("output/images/shapley/dom_sp/",sp,"_varImp.png"), width=15, height=15, unit="cm", res=300, pointsize=7.5)
  print(shap.var.importance(shap_rank, top_n = length(prd_list), title = sp))
  dev.off()
  
  ## Summary
  shap_prep <- shap.prep(shap = shap_rank,
                         X_train = xgb_rsp_var_sp_i,
                         top_n = length(prd_list))
  
  png(paste0("output/images/shapley/dom_sp/",sp,"_summary.png"), width=15, height=15, unit="cm", res=300, pointsize=7.5)
  par(mfrow=c(1,1), oma=c(2,2,2,0))
  print(plot.shap.summary(data_long = shap_prep, title = sp))
  dev.off()
  
  
  ## Dependency plots
  png(paste0("output/images/shapley/dom_sp/", sp,"_DP.png"), width=15, height=15, unit="cm", res=300, pointsize=7.5)
  par(mfrow=c(1,1), oma=c(2,2,2,0))
  xgb.plot.shap_M(data = xgb_rsp_var_sp_i, # input data
                  data_occ = df_occ_var,  # for occ tick marks
                  model = res_xgb_ALL,
                  features = res_xgb_ALL$feature_names, # only top 10 var
                  n_col = 3, # layout option
                  plot_loess = T, # add red line to plot
                  cex.lab=2)
  dev.off()
}






### v) Shapley - Dom_sp - Ecoregions ------------
# Let's have a look at variable importance and dependency plot per ecoregion

ecoregions <- vect("input/vectors/ecoregions.shp")
ecoregions$EcoR[2] <- "Swiss Plateau"
plot(ecoregions)

res_xgb_ALL <- readRDS(paste0("output/models/2_xgb/res_xgb_ALL.rds"))  # load full model
sp_select <- c("Abies alba", "Fagus sylvatica", "Picea abies", "Pinus sylvestris", "Quercus petraea", "Quercus robur")
prd_list <- prd_lst_ALL

for (ecoR in ecoregions$EcoR){
  print(ecoR)
  vect_ecoR <- ecoregions[which(ecoregions$EcoR == ecoR)]  # select region
  
  for (sp in sp_select){
    print(sp)
    raster_sp <- rast(paste0("input/rasters/dominant_sp/", sp,".tif"))
    raster_sp_mask <- mask(raster_sp, vect_ecoR)
    
    prd_list_msk <- lapply(X=prd_list, Y=raster_sp_mask, FUN=function(X,Y) mask(X,Y))  # masking all rasters to ecoregion
    xgb_rsp_var_ecoR <- as.matrix(as.data.frame(rast(prd_list_msk)))  # extracting predictor values
    
    if(nrow(xgb_rsp_var_ecoR) > 0){
      if(nrow(xgb_rsp_var_ecoR) > 10000){  # sample if too many pixels
        xgb_rsp_var_ecoR <- xgb_rsp_var_ecoR[sample(nrow(xgb_rsp_var_ecoR),10000), ]
      }
      
      ## Var influence & importance
      shap_rank <- shap.score.rank(xgb_model = res_xgb_ALL, 
                                   X_train = xgb_rsp_var_ecoR,
                                   shap_approx = F)
      
      png(paste0("output/images/shapley/ecoregions/", ecoR, "/", sp, "_varImp.png"), width=15, height=15, unit="cm", res=300, pointsize=7.5)
      print(shap.var.importance(shap_rank, top_n = length(prd_list), title = paste0(sp, " - ",ecoR)))
      dev.off()
      
      
      ## Summary
      shap_prep <- shap.prep(shap = shap_rank,
                             X_train = xgb_rsp_var_ecoR,
                             top_n = length(prd_list))
      
      png(paste0("output/images/shapley/ecoregions/", ecoR, "/", sp, "_summary.png"), width=15, height=15, unit="cm", res=300, pointsize=7.5)
      par(mfrow=c(1,1), oma=c(2,2,2,0))
      print(plot.shap.summary(data_long = shap_prep, title = paste0(sp, " - ",ecoR)))
      dev.off()
      
      
      ## Dependency plots
      png(paste0("output/images/shapley/ecoregions/", ecoR, "/", sp, "_DP.png"), width=15, height=15, unit="cm", res=300, pointsize=7.5)
      par(mfrow=c(1,1), oma=c(2,2,2,0))
      xgb.plot.shap(data = xgb_rsp_var_ecoR, # input data
                    model = res_xgb_ALL,
                    features = res_xgb_ALL$feature_names, # only top 10 var
                    n_col = 3, # layout option
                    plot_loess = T, # add red line to plot
                    cex.lab=2)
      dev.off()
      
    }
  }
}








