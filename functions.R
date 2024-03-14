
#####-------------------------------------------------------------------------###
###                                FUNCTIONS                                  ###
#####-------------------------------------------------------------------------###


#*----
# I) MODELLING FUNCTIONS  #####################################################

## fire_mod_multi_list() ---

# --> Main modelling framework
# Takes as input one or multiple lists of predictors
# Performs multiple models (GLM, GAM, GBM) with k replicates and ensemble them for each list
# Pre-define model parameters in case of simple-complex model (Brun et al., 2020)
# Save multiple outputs in individual folders per predictor list: map, rasters, assessment, RDS files
# Plot metrics

fire_mod_multi_list <- function(nested_list,
                                backgrd_pts,           # number of background points / pseudo-absences
                                df_train,              # raw training dataset with presences
                                df_test,               # raw testing dataset with presences
                                models,                # model choice among GLM, GAM and GBM, can be a vector of several models
                                simple_complex = TRUE,  # whether or not having simple & cplx model per modelling technique (e.g. GLM)
                                num_rep = 2,           # replicate number
                                k.glm = NULL,          # k polynomial order
                                s.gam = NULL,          # s smooth parameter
                                tc.gbm = NULL,         # tc tree complexity
                                save_mod_rds = FALSE,  # saving models as RDS file
                                is.forest_mask = FALSE,  # masking or not the final png maps
                                forest_mask = NULL,    # forest mask as raster format if previous option on TRUE
                                plot.dpi = 300,        # final png resolution
                                plot.metrics = TRUE,   # whether of not plotting metrics
                                output_source_dir
                                ){
  
  list_res <- list()
  nested_list_names <- names(nested_list)
  
  
  ###------ Defining simple/complex models ------###
  # Predefined values
  if(simple_complex == TRUE){
    df_mod <- as.data.frame(matrix(ncol=3, nrow=1))
    if("glm" %in% models){
      k.glm=c(2,4)  # polynomial orders
      df_mod <- rbind(df_mod, as.data.frame(matrix(c("GLM","GLM",k.glm[1],k.glm[2],"smpl","cplx"),ncol=3,nrow=2)))
    }
    if("gam" %in% models){
      s.gam=c(3,8)  # smooth degrees
      df_mod <- rbind(df_mod, as.data.frame(matrix(c("GAM","GAM",s.gam[1],s.gam[2],"smpl","cplx"),ncol=3,nrow=2)))
    }
    if("gbm" %in% models){
      tc.gbm=c(1,10)  # tree complexity
      df_mod <- rbind(df_mod, as.data.frame(matrix(c("GBM","GBM",tc.gbm[1],tc.gbm[2],"smpl","cplx"),ncol=3,nrow=2)))
    }
    colnames(df_mod) <- c("mod","val","name")
    df_mod <- df_mod[-1,]
  }else{df_mod <- NULL}
  
  
  ###------ Running models ------###
  print_delineator("RUNNING MODELS", max_length = 70, delin.type = "#", bl.space = 2)  # display modelling step in a pretty way
  k=1
  for(list_i in nested_list){
    print(paste0("###===============  ", nested_list_names[k], " (",k," out of ", length(nested_list_names),")  ===============###"))
    df_occ_var_train_i <- prep_df_occ_var(df_train, list_i, backgrd_pts, na.rm=TRUE)  # prepare a df with Pres/Abs and the variables values on each point (train df for Pres)
    df_occ_var_test_i <- prep_df_occ_var(df_test, list_i, backgrd_pts, na.rm=TRUE)  # same with testing dataset
    wts <<- model_weights(df_train, list_i, backgrd_pts)   # model weights to counterbalance class-imbalance ; <<- to turn wts a global var (issues otherwise)
    
    # Perform all the models and extract the final outcome
    list_i_res <- fire_mod_single_list(pred_list = list_i,  # one set of predictors
                                      df_occ_var_train = df_occ_var_train_i,
                                      df_occ_var_test = df_occ_var_test_i,
                                      wts = wts,  # weights
                                      models,
                                      num_rep = num_rep,
                                      k.glm = k.glm,
                                      s.gam = s.gam,
                                      tc.gbm = tc.gbm,
                                      df_mod = df_mod  # df related to simple/complex models (if simple_complex = TRUE)
    )
    
    ## Ensembling data
    list_i_res <- mod_ensembling(list_i_res)  # keep the best models, for 6 submodels keep the best 4
    list_res <- append(list_res, list(list_i_res))
    k <- k+1
  }
  names(list_res) <- nested_list_names  # add the name of each list to list_res
  
  
  ###------ Saving models, rasters and risk maps ------###
  print_delineator("SAVING MODELS, RASTERS, RISK MAPS", max_length = 70, delin.type = "#", bl.space = 2)
  for(i in nested_list_names){  # scan across all selected predictor lists (ALL, GD, etc)
    res_list_i <- list_res[i][[1]]
    
    ## Create saving folder if not existing
    if(length(nchar(list.files(paste0(output_source_dir,i,"/")))) < 1){  # test if folder exists
      dir.create(paste0(output_source_dir,i,"/"), showWarnings = FALSE)  # creates it if not
    }
    
    ## Deleting former files
    file.remove(list.files(paste0(output_source_dir,i,"/"), full.names = T))  # delete all file from previous models
    file.remove(list.files(paste0(output_source_dir,"mod_assessment/"), full.names = T))
    
    
    ## Saving results - RDS files
    # Saving metrics
    rds_eval <- lapply(1:length(res_list_i), function(x){res_list_i[[x]][str_which(names(res_list_i[[x]]), "kappa|tss|auc|boyce")]})  # look at each nested list and only keep what's not a raster
    names(rds_eval) <- names(res_list_i)
    saveRDS(rds_eval, paste0(output_source_dir,i,"/",i, "_eval.rds"))
    
    # Saving xy sampling points
    rds_smp <- map(res_list_i, "smp")
    rds_smp <- rds_smp[-str_which(names(rds_smp), "ENS")]
    saveRDS(rds_smp, paste0(output_source_dir,i,"/",i, "_smp.rds"))
    
    # Saving model data
    if(save_mod_rds == TRUE){  # save a second rds file with models
      rds_mod <- map(res_list_i, "models")  # extract only models from results
      rds_mod <- rds_mod[-str_which(names(rds_mod), "ENS")]  # remove ENS element
      saveRDS(rds_mod, paste0(output_source_dir,i,"/",i, "_mod.rds"))
    }
    
    
    ## Saving rasters
    # Only save maps when available (not all dropped for bad behaviour)
    if(is.null(map(res_list_i, "raster_range")$ENS) == FALSE){  
      list_raster_i_mean <- map(res_list_i, "raster_mean")
      list_raster_i_range <- map(res_list_i, "raster_range")
      
      # Save rasters mean
      invisible(lapply(1:length(list_raster_i_mean), 
                       FUN=function(x) writeRaster(list_raster_i_mean[x][[1]], paste0(output_source_dir, i,"/rast_",i,"_",names(list_raster_i_mean)[x],".tif"), overwrite=TRUE)))
      
      # Save rasters range
      invisible(lapply(1:length(list_raster_i_range), 
                       FUN=function(x) writeRaster(list_raster_i_range[x][[1]], paste0(output_source_dir, i,"/rast_range_",i,"_",names(list_raster_i_range)[x],".tif"), overwrite=TRUE)))
      
      
      ###------ Plotting rasters ------### 
      ## Define color palette
      pal_mean <- colorRampPalette(c(brewer.pal(n=9, name="OrRd")))  # white/yellow to red
      pal_range <- colorRampPalette(c(brewer.pal(n=5, name="BuPu")))  # white blue to blue
      
      ## Select plot title according to predictor list at stake
      if(i == "ALL"){plot_title <- "Full model"
      }else{plot_title <- paste0(i, " model")}  # e.g. title: "GD model"
      
      
      ## Mean ensemble
      png(paste0(output_source_dir,i,"/",i,"_tr1_te1_ENS.png"), width=15, height=13, unit="cm", res=plot.dpi, pointsize=7.5)
      par(mfrow=c(1,1), oma=c(0,1,0,4))
      plot(rast_hillshade, col=grey(seq(0, 1, length.out=100)), legend=FALSE, axes=F, main=paste0(plot_title, " - Ensemble"), cex.main=2.5)
      if(is.forest_mask == TRUE){plot(mask(list_raster_i_mean$ENS, forest_mask), range = c(0,1), col=pal_mean(100), alpha=0.9, axes=FALSE, plg=list(cex=2.5), add=TRUE)}
      else{plot(list_raster_i_mean$ENS, range = c(0,1), col=pal_mean(100), alpha=0.9, axes=FALSE, plg=list(cex=2.5), add=TRUE)}  # plot the ENS mean
      plot(vect_contour, add=TRUE)
      plot(vect_water, col="#c3ebf3", lwd=0.7, add=TRUE)
      plot(st_as_sf(df_train[,names(df_train) %in% c("x","y")],coords=c("x","y"),crs="EPSG:2056"), pch=16, add=TRUE)  # plot ignition points training dataset
      dev.off()
      
      ## Range ensemble
      png(paste0(output_source_dir,i,"/",i,"_tr1_te1_ENS_range.png"), width=15, height=13, unit="cm", res=plot.dpi, pointsize=7.5)
      par(mfrow=c(1,1), oma=c(0,1,0,4))
      plot(rast_hillshade, col=grey(seq(0, 1, length.out=100)), legend=FALSE, axes=F, main=paste0(plot_title, " - Range"), cex.main=2.5)
      plot(list_raster_i_range$ENS, range = c(0,1), col=pal_range(100), alpha=0.9, axes=FALSE, plg=list(cex=2.5), add=TRUE)
      plot(vect_contour, add=TRUE)
      plot(vect_water, col="#c3ebf3", lwd=0.7, add=TRUE)
      plot(st_as_sf(df_train[,names(df_train) %in% c("x","y")],coords=c("x","y"),crs="EPSG:2056"), pch=16, add=TRUE)  # plot ignition points training dataset
      dev.off()
      
      
      ## Mean of all submodels
      # Not showing the ENS on this map as it is already plot above (Mean ensemble)
      list_raster_i_mean_no_ENS <- list_raster_i_mean[-str_which(lapply(list_raster_i_mean, names), "ENS")]  # erase ENS form the final raster list
      png(paste0(output_source_dir,i,"/",i,"_combined.png"), width=15, height=13, unit="cm", res=plot.dpi, pointsize=7.5)
      par(mfrow=mfrow_choice(length(list_raster_i_mean_no_ENS)), oma=c(0,0,0,3))  # define plotting window
      for(j in 1:length(list_raster_i_mean_no_ENS)){
        plot(rast_hillshade, col=grey(seq(0, 1, length.out=100)), legend=FALSE, axes=F, main=names(list_raster_i_mean)[j], cex.main=2)
        plot(list_raster_i_mean_no_ENS[j][[1]], range = c(0,1), col=pal_mean(100), alpha=0.7, axes=FALSE, add=TRUE)
        plot(vect_contour, add=TRUE)
        plot(vect_water, col="#c3ebf3", lwd=0.7, add=TRUE)
      }  
      dev.off()
      
      ## Range of all submodels
      # Not showing the ENS range on this map as it does not make sense (this ENS is not the range across replicates but the of the median of all models)
      list_raster_i_range_no_ENS <- list_raster_i_range[-str_which(lapply(list_raster_i_range, names), "ENS")]  # erase ENS form the final raster list
      png(paste0(output_source_dir,i,"/",i,"_combined_range.png"), width=15, height=13, unit="cm", res=plot.dpi, pointsize=7.5)
      par(mfrow=mfrow_choice(length(list_raster_i_range_no_ENS)), oma=c(0,0,0,3))  # define plotting window
      for(j in 1:length(list_raster_i_range_no_ENS)){
        plot(rast_hillshade, col=grey(seq(0, 1, length.out=100)), legend=FALSE, axes=F, main=names(list_raster_i_mean)[j], cex.main=2)  # uses list_raster_i_mean title
        plot(list_raster_i_range_no_ENS[j][[1]], range = c(0,1), col=pal_range(100), alpha=0.9, axes=FALSE, add=TRUE)
        plot(vect_contour, add=TRUE)
        plot(vect_water, col="#c3ebf3", lwd=0.7, add=TRUE)
      }  
      dev.off()
    }
  }
  
  
  ###------ Plotting metrics ------###
  if(plot.metrics==TRUE){
    print_delineator("PLOTTING MODELS METRICS", max_length = 70, delin.type = "#", bl.space = 2)
    
    if(length(nchar(list.files(paste0(output_source_dir,"mod_assessment/")))) < 1){  # test if folder exists
      dir.create(paste0(output_source_dir,"mod_assessment/"), showWarnings = FALSE)   # create folder if not existing
    }
    
    mod_assessment <- compile_assessment(output_source_dir)  # this function stores all results in a dataframe
    title_names <- c("All predictors",
                     "Ground disposition",
                     "Human influence",
                     "Variable disposition")
    
    metrics_name <- c("Kappa", "TSS", "AUC", "Boyce")
    
    for(i in 1:nrow(mod_assessment[[1]])){
      png(paste0(output_source_dir,"mod_assessment/", title_names[i], ".png"), width=20, height=15, unit="cm", res=300, pointsize=7.5)
      par(mfrow = c(2,2), mar = c(4.1, 4, 1.5, 1), oma=c(2,1,3,0), cex = 1.3)
      for(j in 1:4){  # j for each metric
        barplot(as.matrix(mod_assessment[[j]][i,]), beside = TRUE, ylim = c(0,1),
                ylab = metrics_name[j], col="#30A2FF80", las=2, cex.lab=1.3)
      }
      mtext(title_names[i], side=3, line=0.7, cex=2.3, outer=T)   # write model title
      dev.off()
    }
    par(mfrow=c(1,1))
  }
}









## fire_mod_single_list() ---
# Enable to fit multiple models (GLM, GAM, GBM) for 1 list of predictors from fire_mod_multi_list() inputs

fire_mod_single_list <- function(pred_list,  # list of predictors
                                df_occ_var_train,  # prepared training dataframe with extracted predictor values
                                df_occ_var_test,  # prepared testing dataframe with extracted predictor values
                                wts,             # model weights to deal with class imbalance
                                models,          # model type (GLM/GAM/GBM)
                                num_rep = 2,     # number of replicates, default = 2
                                k.glm = NULL,    # GLM polynomial order, can be a vector with multiple values
                                s.gam = NULL,    # GAM smooth degree, can be a vector with multiple values
                                tc.gbm = NULL,   # GBM tree complexity, can be a vector with multiple values
                                df_mod = NULL    # df related to simple/complex models (if simple_complex = TRUE)
                                ){  
  
  list_models_out <- list()
  list_names <- vector()
  
  ###------ GLM ------###
  if("glm" %in% models){  # when current model is GLM
    print_delineator("GLM in process", max_length = 30, delin.type = "-", bl.space = 1)
    for(k in k.glm){  # scan the selected fitting polynomial orders
      print(paste0("GLM k=", k))
      mod_glm_k <- fire_mod_simple(df_occ_var_train, df_occ_var_test, 
                                   list_predictors=pred_list,
                                   wts, model_type="glm", num_rep=num_rep,
                                   poly.glm.k=k)
      
      list_models_out <- append(list_models_out, list(GLM = mod_glm_k))  # add GLM results to model result list
      
      # Adding model name to name list
      if(is.null(df_mod)==FALSE){
        plot_name <- paste0("GLM", ".", df_mod$name[which(df_mod[str_which(df_mod$mod, "GLM"),]$val == k)])
        list_names <- append(list_names, plot_name)
      }else{list_names <- append(list_names, paste0("GLM.k",k))}
    }
  }
  
  
  ###------ GAM ------###
  if("gam" %in% models){  # when current model is GAM
    print_delineator("GAM in process", max_length = 30, delin.type = "-", bl.space = 1)
    for(s in s.gam){  # scan the selected fitting smooth degrees
      print(paste0("GAM s=", s))
      
      mod_gam_s <- fire_mod_simple(df_occ_var_train, df_occ_var_test, 
                                   list_predictors=pred_list, num_rep=num_rep,
                                   wts, model_type="gam", s.gam=s)
      
      list_models_out <- append(list_models_out, list(GAM = mod_gam_s))  # add GAM results to model result list
      
      # Adding model name to name list
      if(is.null(df_mod)==FALSE){
        plot_name <- paste0("GAM", ".", df_mod$name[which(df_mod[str_which(df_mod$mod, "GAM"),]$val == s)])
        list_names <- append(list_names, plot_name)
      }else{list_names <- append(list_names, paste0("GAM.s",s))}
    }
  }
  
  
  ###------ GBM ------###
  if("gbm" %in% models){  # when current model is GBM
    print_delineator("GBM in process", max_length = 30, delin.type = "-", bl.space = 1)
    for(cplxty in tc.gbm){  # scan the selected fitting tree complexities
      print(paste0("GBM tree complexity=", cplxty))
      
      mod_gbm_tc <- fire_mod_simple(df_occ_var_train, df_occ_var_test, 
                                    list_predictors=pred_list, num_rep=num_rep,
                                    wts, model_type="gbm", tc.gbm=cplxty)
      
      list_models_out <- append(list_models_out, list(GBM = mod_gbm_tc))  # add GBM results to model result list
      
      # Adding model name to name list
      if(is.null(df_mod)==FALSE){
        plot_name <- paste0("GBM", ".", df_mod$name[which(df_mod[str_which(df_mod$mod, "GBM"),]$val == cplxty)])
        list_names <- append(list_names, plot_name)
      }else{list_names <- append(list_names, paste0("GBM",cplxty))}
    }
  }
  
  
  names(list_models_out) <- list_names  # attributes right names to all models in the list
  list_models_out <- list_models_out[order(names(list_models_out))]  # reorder elements by name
  return(list_models_out)
}






## fire_mod_simple() ---
# Core modelling function used in fire_mod_single_list()
# Perform only 1 model type (out of GLM, GAM, GBM) with k replicates (~65% sampled presence points)
# Calculates assessment metrics
# Predict ffs map
fire_mod_simple <- function(df_occ_var_train,  # dataset with Pres/Abs of training points, and predictor values
                               df_occ_var_test,   # dataset with Pres/Abs of testing points, and predictor values
                               list_predictors,   # list of variables in raster format
                               wts,     # weights for class imbalance
                               model_type,  # model selection within GLM, GAM and GBM (+RF)
                               num_rep = 2,  # number of replicates
                               poly.glm.k = NULL,  # polynomial order of GLM
                               s.gam = NULL,   # smooth coeff for GAM
                               tc.gbm = NULL,
                               mod_spl_cplx_name = NULL){
  
  
  list_out <- list()  # list to combine the mean of the replicates
  list_rep_tot <- list()  # list to save output of all replicates
  
  
  for(r in 1:num_rep){
    list_rep_r <- list()  # empty list to put all mod outputs per replicate, then aggregated into list_out
    
    ### Replicates data attribution (randomnessbtw replicates)
    if(num_rep == 1){df_occ_var_train_rep_i <- df_occ_var_train}  # case with no replicates or first replicate
    else{df_occ_var_train_rep_i <- df_occ_var_train[c(sample(which(df_occ_var_train[,1]==1), replace=T), 
                                                      which(df_occ_var_train[,1]==0)),]}  # keep ~65% of presences per replicate, final number is the same as input but with some duplicated points
    
    
    ###------ Model fitting ------###
    formula <- pred_to_formula(list_predictors, model_type=model_type, poly.glm.k=poly.glm.k, s.gam=s.gam)  # defining modelling formula for GLM and GAM (no influence with GBM)
    if(model_type=="glm"){
      name_mod <- paste0("GLM.k", poly.glm.k)  # save model name
      mod_glm <- glm(formula, data = df_occ_var_train_rep_i, family = 'binomial', weights = wts)   # perform GLM model, binomial family for Pres/Abs data
      mod_out <- step(mod_glm, directions = 'both', trace = FALSE)  # forward-backward variables selection
      list_rep_r <- list(mod_res = mod_out)  # save model replicate
    }
    
    if(model_type=="gam"){
      name_mod <- paste0("GAM.s", s.gam)
      mod_out <- mgcv::gam(formula, data = df_occ_var_train_rep_i, family = 'binomial', weights = wts)  # perform GAM model, binomial family for Pres/Abs data
      list_rep_r <- list(mod_res = mod_out)
    }
    
    if(model_type=="gbm"){
      name_mod <- paste0("GBM")  # save model name
      mod_out <- gbm.step(data=df_occ_var_train_rep_i, 
                          gbm.y = 1,
                          gbm.x = 4:ncol(df_occ_var_train_rep_i), 
                          family = "bernoulli", 
                          site.weights = wts,
                          tree.complexity = tc.gbm,
                          learning.rate = 0.005,  # issues if higher for VD predictor list
                          bag.fraction = 0.5,
                          n.folds = 10,  # cross validation folding number
                          silent = T)  # don't show console output
      list_rep_r <- list(mod_res = mod_out)
    }
    
    
    
    ###------ Model predictions ------###
    ## Predicting Presence/Absence with independent test dataset and the created model
    if(model_type %in% c("glm","gam")){
      mod_prd <- predict(mod_out, newdata=df_occ_var_test, type="response")  
    }
    
    if(model_type == "gbm"){
      mod_prd <- predict(mod_out, df_occ_var_test, n.trees=mod_out$gbm.call$best.trees, type="response")
    }
    
    
    ## Fire danger prediction map
    if(model_type == "gbm"){
      raster_prd <- predict(rast(list_predictors), mod_out, 
                            n.trees = mod_out$gbm.call$best.trees, 
                            type = "response", na.rm=T)
    }else{raster_prd <- predict(rast(list_predictors), mod_out, type = "response")}    # GLM and GAM case
    list_rep_r <- append(list_rep_r, list(raster=raster_prd))  # add predicted raster to the output list
    
    
    ## XY sampling points / replicates
    list_rep_r <- append(list_rep_r, list(smp_xy = df_occ_var_train_rep_i[,c("x","y")]))
    
    
    
    ###------ Model assessment ------###
    ## Dataset preparation for assessment, gather observed presences and predicted probability presences (mod_prd)
    df_prd <- data.frame(ID = 1:nrow(df_occ_var_test), 
                         Obs = df_occ_var_test$occ,  # occurrences from test dataset
                         Prd = as.data.frame(mod_prd))  # model predictions
    
    ## Threshold-dependent metrics
    prd_thres <- optimal.thresholds(df_prd)  # metrics thresholds
    cm_prd <- cmx(df_prd , threshold = prd_thres[4,2])  # binarization with MaxKappa ([4,2])
    eval_kappa <- kappa(cm_prd)  # calculates kappa based on selected threshold
    eval_tss <- tss(cm_prd)   
    
    ## Threshold-independent metrics (no cm_prd)
    eval_auc <- AUC::auc(roc_wrap(df_prd)[[1]])
    eval_boyce <- boyce_wrap(df_prd)$cor
    
    ## returning model assessment
    list_rep_r <- append(list_rep_r, list(kappa=eval_kappa, tss=eval_tss, auc=eval_auc, boyce=eval_boyce))  # add all metrics to the replicate output info
    list_rep_tot <- append(list_rep_tot, list(list_rep_r))   # save replicate r infos into a list
  }
  
  
  
  ###------ Combining replicates ------###
  list_out$models <- map(list_rep_tot, "mod_res")  # saving models of all replicates
  list_out$smp <- map(list_rep_tot, "smp_xy")  # Saving sampling points of all replicates
  
  list_out$raster_mean <- mean(rast(map(list_rep_tot, "raster")))  # extract all rasters and calculates the mean raster of all replicate risk maps
  list_out$raster_range <- max(rast(map(list_rep_tot, "raster"))) - (min(rast(lapply(list_rep_tot, "[[", "raster"))))   # range (incertitude across risk maps)
  
  names(list_out$raster_mean) <- name_mod
  names(list_out$raster_range) <- name_mod
  
  list_out$kappa <- mean(unlist(map(list_rep_tot, "kappa")))  # calculates the mean of the kappa across all replicates
  list_out$tss <- mean(unlist(map(list_rep_tot, "tss")))
  list_out$auc <- mean(unlist(map(list_rep_tot, "auc")))
  list_out$boyce <- mean(unlist(map(list_rep_tot, "boyce")), na.rm=TRUE)
  if(is.na(list_out$boyce)==TRUE){list_out$boyce <- 0}
  
  return(list_out)
}






## mod_ensembling() ---
## Look at the output models and keep the best ones for ensembling
# function used in batch_mod_multi_list2
mod_ensembling <- function(result_mod){
  
  ## Keeping the best ~60-70% rasters for ensembling, for 6 submodels keep the best 4
  list_good_mod <- result_mod[names(sort(unlist(map(result_mod,"boyce")), decreasing=TRUE)[1:ceiling(length(result_mod)*0.6)])]
  
  result_mod$ENS$raster_range <- max(rast(map(list_good_mod,"raster_mean"))) - min(rast(map(list_good_mod,"raster_mean")))  # take the range of all submodels range
  result_mod$ENS$raster_mean <- mean(rast(map(list_good_mod, "raster_mean")))  # mean of the risk maps per model
  names(result_mod$ENS$raster_range) <- "ENS"
  names(result_mod$ENS$raster_mean) <- "ENS"
  
  ## Average all metrics across replicates
  result_mod$ENS$kappa <- mean(unlist(map(list_good_mod, "kappa")))
  result_mod$ENS$tss <- mean(unlist(map(list_good_mod, "tss")))
  result_mod$ENS$auc <- mean(unlist(map(list_good_mod, "auc")))
  result_mod$ENS$boyce <- mean(unlist(map(list_good_mod, "boyce")))
  
  return(result_mod)
}








## compile_assessment() ---
## Making a list of dataframes with the results of all models from fire_mod_single_list()
# The input is the source folder containing the models RDS file outputs from fire_mod_single_list()
# Important function used in fire_mod_multi_list() for plotting metrics
compile_assessment <- function(path_folder){
  list_rds <- list()
  list_names <- list()
  
  ## Compile all rds files into one list
  for(i in list.files(path_folder, full.names = T)){  # scan all folders where models are stored
    path_rds <- str_subset(list.files(paste0(i,"/"), full.names = TRUE), "\\.rds")  # check files with ".rds"
    if(length(path_rds)!=0){  # in the case some folders don't have rds files
      path_rds_eval <- str_subset(path_rds, "eval.rds")  # only keep assessment rds file
      rds_file_i <- read_rds(path_rds_eval)  # open assessment
      list_names <- append(list_names, str_split(tail(str_split(path_rds_eval, "/")[[1]], n=1), ".rds")[[1]][1])  # extract rds main model name
      list_rds <- append(list_rds, list(rds_file_i))  # add the eval of model i to the others
    }
  }
  
  names(list_rds) <- list_names  # attributes the right names
  mod_names <- names(list_rds[[1]])
  
  ## Getting metrics assessments for each model
  list_assessment <- list()  # template empty list for saving output
  for(i in 1:length(list_rds)){
    list_assessment$kappa <- append(list_assessment$kappa, list(lapply(list_rds[[i]], "[[", "kappa")))  # extract all kappa values
    list_assessment$tss <- append(list_assessment$tss, list(lapply(list_rds[[i]], "[[", "tss")))  # extract all tss values
    list_assessment$auc <- append(list_assessment$auc, list(lapply(list_rds[[i]], "[[", "auc")))  # extract all auc values
    
    list_boyce <- flatten(lapply(list_rds[[i]], "[[" , "boyce"))  # special function because some differences inside list_rds, extract all boyce values
    names(list_boyce) <- mod_names
    list_assessment$boyce <- append(list_assessment$boyce, list(list_boyce))
  }
  
  ## Turn results into a list of dataframes
  list_assessment <- lapply(1:length(list_assessment), function(x) {as.data.frame(rbindlist(list_assessment[[x]], fill=TRUE))})
  names(list_assessment) <- c("kappa", "tss", "auc", "boyce")
  
  # Change rownames of dataframes
  for(i in 1:length(list_assessment)){
    rownames(list_assessment[[i]]) <- unlist(list_names)
  }
  return(list_assessment)
}





## pred_ffs_nwVar() ---
# The function enable to extract former models saved as RDS file and
# to predict a fire map with an updated set of the same predictors new predictors
pred_ffs_nwVar <- function(pred_list, path_rds){
  
  ## Loading RDS files
  rds_eval <- readRDS(str_subset(list.files(path_rds, full.names = TRUE), "eval.rds"))
  rds_mod <- readRDS(str_subset(list.files(path_rds, full.names = TRUE), "mod.rds"))
  
  rds_eval_no_ens <- rds_eval[-str_which(names(rds_eval), "ENS")]
  list_good_mod <- names(rds_eval_no_ens[names(sort(unlist(map(rds_eval_no_ens,"boyce")), decreasing=TRUE)[1:ceiling(length(rds_eval_no_ens)*0.6)])])  # selecting the best 60% models
  
  ## Predicting risk map with new predictors
  rep <- length(rds_mod[list_good_mod[1]][[1]])  # number of replicates
  list_rast <- list()
  for(m in list_good_mod){
    print_delineator(m, max_length = 20, delin.type = "-", bl.space = 1)
    for(i in 1:rep){  # Predict for each replicate
      print(paste0(i, " out of ", rep))
      if(str_detect(m, "GLM")){
        mod_i <- rds_mod[str_subset(m[[1]], "GLM")][[1]][i]  # extract replicate
        raster_prd <- predict(rast(pred_list), mod_i, type = "response")
        names(raster_prd) <- paste0(m,"_",i)
        list_GLM <- list(raster_prd)
        names(list_GLM) <- m
        list_rast <- append(list_rast, list_GLM)
      }
      if(str_detect(m, "GAM")){
        mod_i <- rds_mod[str_subset(m[[1]], "GAM")][[1]][i]  # extract replicate
        raster_prd <- predict(rast(pred_list), mod_i, type = "response")
        names(raster_prd) <- paste0(m,"_",i)
        list_GAM <- list(raster_prd)
        names(list_GAM) <- m
        list_rast <- append(list_rast, list_GAM)
      }
      if(str_detect(m, "GBM")){
        mod_i <- rds_mod[str_subset(m[[1]], "GBM")][[1]][i]  # extract replicate
        raster_prd <- predict(rast(prd_lst_ALL_mixt_m), mod_i, 
                              n.trees = mod_i$gbm.call$best.trees, 
                              type = "response", na.rm=T)
        names(raster_prd) <- paste0(m,"_",i)
        list_GBM <- list(raster_prd)
        names(list_GBM) <- m
        list_rast <- append(list_rast, list_GBM)
      }
    }
  }
  
  list_rast_ens_m <- lapply(1:length(list_good_mod), function(x) mean(rast(keep(list_rast, names(list_rast)==list_good_mod[x]))))   # select the replicates of each model and take mean
  raster_nw_pred <- mean(rast(list_rast_ens_m))
  
  return(raster_nw_pred)
}








## model_weights() ---
# Provides the weights to deal with class imbalance in models
model_weights <- function(fire_dataset, list_predictors, backgrd_pts){
  model_matrix <- prep_df_occ_var(fire_dataset, list_predictors, backgrd_pts, na.rm=TRUE)
  weights = rep(1,nrow(model_matrix))
  weights[which(model_matrix[,1]==0)] = 1
  weights[which(model_matrix[,1]==1)] = round(length(which(model_matrix[,1]==0))/length(which(model_matrix[,1]==1)))
  return(weights)
}







## pred_to_formula() ---
## Transform a predictor list to a model formula
pred_to_formula <- function(list_predictors, Y="occ", model_type, poly.glm.k=NULL, s.gam=NULL){
  
  if(class(list_predictors)=="list"){list_predictors <- rast(stack(lapply(list_predictors, raster)))}
  if(class(list_predictors)=="RasterStack"){list_predictors <- rast(list_predictors)}
  var_names <- unlist(lapply(list_predictors, names))
  
  if(model_type=="glm"){
    var_form <- c()
    for (i in 1:poly.glm.k){
      var_form <- append(var_form, paste0("I(", var_names, "^", i, ")"))
    }
    formula_glm <- reformulate(var_form, Y)
    return(formula_glm)
  }
  
  if(model_type=="gam"){
    formula_gam <- reformulate(paste0("s(",var_names,",k=",s.gam, ")"), Y)
    return(formula_gam)
  }
}








#* ----
# II) ASSESSMENT FUNCTIONS  #####################################################
# Kappa and tss functions have to be used with confusion matrix output
## kappa() ---

kappa <- function(cm){
  a <- cm[1,1]; b = cm[1,2]; c = cm[2,1]; d = cm[2,2]
  n <- a + b + c + d
  out <- ((a+d)/n-((a +b)*(a +c)+(c+d)*(d+b))/n^2)/(1-((a+b)*(a+c)+(c+d)*(d+b))/n^2)
  return(out)
}


## tss() ---
tss <- function(cm){
  a <- cm[1,1]; b = cm[1,2]; c = cm[2,1]; d = cm[2,2]
  out <- a/(a+c)+d/(b+d)-1
  return(out)
}


## roc_wrap() ---
# Small wrapper function that reformats the data to fit the requirements of the 'roc' function
roc_wrap <- function(prd){
  l_roc <- list()
  # Loop over model predictions
  obs <- as.factor(prd$Obs)
  for(i in 3:ncol(prd)){
    l_roc[[i-2]] <- AUC::roc(prd[,i],obs)
  }
  names(l_roc) <- colnames(prd)[3:ncol(prd)]
  return(l_roc)
}



## boyce_wrap() ---
# Boyce function, only uses presence data
boyce_wrap <- function(prd, ...){
  boyce <- ecospat.boyce(fit = prd[,3], # all predictions
                         obs = prd[which(prd$Obs == 1),3],  # predictions at presence points
                         PEplot = FALSE)
  return(boyce)
}







#*----
# III) SHAPLEY FUNCTIONS  #####################################################
# Note: The functions shap.score.rank, shap_long_hd and plot.shap.summary were 
# originally published at https://liuyanguu.github.io/post/2018/10/14/shap-visualization-for-xgboost/
# All the credits to the author.

## shap.score.rank() ---
## Calcultates the Shapley scores/contributions based on XGB model prediction
# return matrix of shap score and mean ranked score list
shap.score.rank <- function(xgb_model = xgb_mod, shap_approx = TRUE, 
                            X_train = mydata$train_mm){
  require(xgboost)
  require(data.table)
  shap_contrib <- predict(xgb_model, X_train, predcontrib = TRUE, approxcontrib = shap_approx)  # extract contribution
  shap_contrib <- as.data.table(shap_contrib)
  shap_contrib[,BIAS:=NULL]
  # cat('make SHAP score by decreasing order\n\n')
  mean_shap_score <- colMeans(abs(shap_contrib))[order(colMeans(abs(shap_contrib)), decreasing = T)]
  return(list(shap_score = shap_contrib,
              mean_shap_score = (mean_shap_score)))
}




## shap.std() ---
# a function to standardize feature values into same range
shap.std <- function(x){
  return ((x-min(x, na.rm = T)) / (max(x, na.rm = T)-min(x, na.rm = T)))
}




## shap.prep() ---
# Prepare data of Shapley results
shap.prep <- function(shap  = shap_result, X_train = mydata$train_mm, top_n){
  require(ggforce)
  # descending order
  if (missing(top_n)) top_n <- dim(X_train)[2] # by default, use all features
  if (!top_n%in%c(1:dim(X_train)[2])) stop('supply correct top_n')
  require(data.table)
  shap_score_sub <- as.data.table(shap$shap_score)
  shap_score_sub <- shap_score_sub[, names(shap$mean_shap_score)[1:top_n], with = F]
  shap_score_long <- melt.data.table(shap_score_sub, measure.vars = colnames(shap_score_sub))
  
  # feature values: the values in the original dataset
  fv_sub <- as.data.table(X_train)[, names(shap$mean_shap_score)[1:top_n], with = F]
  # standardize feature values
  fv_sub_long <- melt.data.table(fv_sub, measure.vars = colnames(fv_sub))
  fv_sub_long[, stdfvalue := shap.std(value), by = "variable"]
  
  # SHAP value: value
  # raw feature value: rfvalue; 
  # standarized: stdfvalue
  names(fv_sub_long) <- c("variable", "rfvalue", "stdfvalue" )
  shap_long2 <- cbind(shap_score_long, fv_sub_long[,c('rfvalue','stdfvalue')])
  shap_long2[, mean_value := mean(abs(value)), by = variable]
  setkey(shap_long2, variable)
  return(shap_long2) 
}





## plot.shap.summary() ---
plot.shap.summary <- function(data_long, title=""){
  x_bound <- max(abs(data_long$value))
  require('ggforce') # for `geom_sina`
  plot1 <- ggplot(data = data_long)+
    coord_flip() + 
    # sina plot: 
    geom_sina(aes(x = variable, y = value, color = stdfvalue)) +
    # print the mean absolute value: 
    geom_text(data = unique(data_long[, c("variable", "mean_value"), with = F]),
              aes(x = variable, y=-Inf, label = sprintf("%.3f", mean_value)),
              size = 3, alpha = 0.7,
              hjust = -0.2, 
              fontface = "bold") + # bold
    # # add a "SHAP" bar notation
    # annotate("text", x = -Inf, y = -Inf, vjust = -0.2, hjust = 0, size = 3,
    #          label = expression(group("|", bar(SHAP), "|"))) + 
    scale_color_gradient(low="#FFCC33", high="#6600CC", 
                         breaks=c(0,1), labels=c("Low","High")) +
    theme_bw() + 
    ggtitle(title) +
    theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(), # remove axis line
          legend.position="bottom") + 
    geom_hline(yintercept = 0) + # the vertical line
    scale_y_continuous(limits = c(-x_bound, x_bound)) +
    # reverse the order of features
    scale_x_discrete(limits = rev(levels(data_long$variable)) 
    ) + 
    labs(y = "SHAP value (impact on model output)", x = "", color = "Feature value") 
  return(plot1)
}





## plot.shap.summary() ---
shap.var.importance <- function(shap_result, top_n = 10, title = NULL){
  var_importance <- tibble(var=names(shap_result$mean_shap_score), importance=shap_result$mean_shap_score)
  var_importance <- var_importance[1:top_n,]
  
  ggplot(var_importance, aes(x=reorder(var,importance), y=importance)) + 
    geom_bar(stat = "identity") + 
    coord_flip() + 
    theme_light() + 
    ggtitle(title) +
    theme(title = element_text(size = 14),
          axis.title.x = element_text(size = 12), 
          axis.title.y = element_blank(),
          axis.text.y = element_text(size = 12))
  
}




## xgb.plot.shap_M() ---
# Modified function from xgboost:::xgb.plot.shap()
xgb.plot.shap_M <- function (data, 
                             data_occ = NULL,  # same dataset as data but with a filter on occurrences
                             features = NULL, 
                             shap_contrib = NULL, 
                             model = NULL, 
                             top_n = 1, 
                             trees = NULL, 
                             target_class = NULL, 
                             approxcontrib = FALSE, 
                             subsample = NULL, 
                             n_col = 1, 
                             col = rgb(0, 0, 1, 0.2),  #col of points (blue-purple)
                             pch = ".", 
                             discrete_n_uniq = 5, 
                             discrete_jitter = 0.01, 
                             ylab = "SHAP", 
                             plot_NA = TRUE, 
                             col_NA = rgb(0.7, 0, 1, 0.6),  # NA col of pts (purple)
                             pch_NA = ".", 
                             pos_NA = 1.07, 
                             plot_loess = TRUE, 
                             col_loess = 2, 
                             span_loess = 0.5, 
                             plot = TRUE, ...){
  
  ## Prepare data fro SHAP plot
  data_list <- xgboost:::xgb.shap.data(data = data, shap_contrib = shap_contrib, 
                                       features = features, top_n = top_n, model = model, trees = trees, 
                                       target_class = target_class, approxcontrib = approxcontrib, 
                                       subsample = subsample, max_observations = 1e+05)
  
  data <- data_list[["data"]]
  shap_contrib <- data_list[["shap_contrib"]]
  features <- colnames(data)
  
  if (n_col > length(features)){n_col <- length(features)}
  
  op <- par(mfrow = c(ceiling(length(features)/n_col), n_col), 
            oma = c(0, 0, 0, 0) + 0.2, 
            mar = c(3.5, 3.5, 0, 0) + 0.1, 
            mgp = c(1.7, 0.6, 0))
  
  ## Scan all variables and plot the dependency plot
  for (f in features) {
    ord <- order(data[, f])
    x <- data[, f][ord]
    y <- shap_contrib[, f][ord]
    x_lim <- range(x, na.rm = TRUE)
    y_lim <- range(y, na.rm = TRUE)
    do_na <- plot_NA && any(is.na(x))
    
    if (do_na) {
      x_range <- diff(x_lim)
      loc_na <- min(x, na.rm = TRUE) + x_range * pos_NA
      x_lim <- range(c(x_lim, loc_na))
    }
    
    x_uniq <- unique(x)
    x2plot <- x
    if (length(x_uniq) <= discrete_n_uniq){
      x2plot <- jitter(x, amount = discrete_jitter * 
                         min(diff(x_uniq), na.rm = TRUE))
    }
    
    plot(x2plot, y, pch = pch, xlab = f, col = col, 
         xlim = x_lim, ylim = y_lim, ylab = ylab, ...)
    grid()
    
    points(x=data_occ[,f], y=rep(y_lim[1], nrow(data_occ)), pch = "I", col = "#00000050", cex=2)  # add occurrence points
    
    
    if (plot_loess) {
      zz <- data.table(x = signif(x, 3), y)[, .(.N, 
                                                y = mean(y)), x]
      if (nrow(zz) <= 5) {
        lines(zz$x, zz$y, col = col_loess)
      }
      else {
        lo <- stats::loess(y ~ x, data = zz, weights = zz$N, 
                           span = span_loess)
        zz$y_lo <- predict(lo, zz, type = "link")
        lines(zz$x, zz$y_lo, col = col_loess)
      }
    }
    
    if (do_na) {
      i_na <- which(is.na(x))
      x_na <- rep(loc_na, length(i_na))
      x_na <- jitter(x_na, amount = x_range * 0.01)
      points(x_na, y[i_na], pch = pch_NA, col = col_NA)
    }
  }
  par(op)
  
  invisible(list(data = data, shap_contrib = shap_contrib))
}










#*----
# IV) COMPLEMENTARY FUNCTIONS  #####################################################
## prep_df_occ_var() ---
# Creates a dataframe with the fires Pres/pseudo-Abs and the related predictor values
prep_df_occ_var <- function(fires_dataset, predictor_stack, backgrd_pts, res, na.rm=FALSE){
  
  if(class(predictor_stack)=="list"){predictor_stack <- rast(stack(lapply(predictor_stack, raster)))}
  if(class(predictor_stack)=="RasterStack"){predictor_stack <- rast(predictor_stack)}
  
  # Extracting predictor values at ignition points
  fires_be_env <- bind_cols(fires_dataset[,c("x","y")], terra::extract(predictor_stack, as.matrix(fires_dataset[,c("x", "y")])))  # extract pred values for xy ccords
  fires_be_env$fires <- 1
  fires_be_env <- relocate(fires_be_env, "fires", .before="x")
  
  # Sampling pseudo-absences
  fires_be_env_abs <- terra::spatSample(predictor_stack, backgrd_pts, method = "random", na.rm = TRUE, as.df = TRUE, xy = TRUE)
  fires_be_env_abs$fires <- 0
  fires_be_env_abs <- relocate(fires_be_env_abs, "fires", .before="x")
  
  # Combining presence and pseudo-abs datasets
  df_fire_be_sub <- rbind(fires_be_env, fires_be_env_abs)
  if(na.rm==TRUE){df_fire_be_sub <- na.omit(df_fire_be_sub)}
  colnames(df_fire_be_sub)[1] <- "occ"
  return(df_fire_be_sub)
}









## predictor_selection() ---
## Performs predictor comparison to help select the best variables
# Important point for D2ajd: complicated version that separates rasters with and without forest mask
# necessary to avoid model weights issues with different number of NA values

predictor_selection <- function(nested_compare_lists,  # lists of variables to compare
                                df_fires,  # training dataset
                                cor_plot = TRUE,
                                VIF = TRUE,
                                D2adj = FALSE,  # univariate D2adj
                                glm = FALSE,
                                glm.k = 2,
                                glm.plot = FALSE,
                                glm.step = FALSE,
                                ecospat.D2adj_mean = FALSE,
                                D2adj_mean_rep = 10)  # number of replicates for the mean D2adj
{
  length_list <- length(nested_compare_lists)
  nested_pred_stack <- lapply(X=1:length_list, function(X) stack(lapply(unlist(nested_compare_lists[X]), FUN=raster)))  # turning each pred list into a stack of rasters
  for(i in 1:length_list){names(nested_pred_stack[[i]]) <- unlist(lapply(nested_compare_lists[[i]], names))}  # attributing right names
  bck_pts <- 10000
  
  ###------ VIF and Spearman correlation ------###
  if(cor_plot==TRUE | VIF==TRUE){
    print_delineator("VIF", max_length = 50, delin.type = "=")
    for(i in 1:length_list){
      pred_stack_df <- as.data.frame(terra::extract(nested_pred_stack[[i]], dismo::randomPoints(nested_pred_stack[[i]], n = 100000))) %>% na.omit() # extract values for a subset of n points
      
      ## VIF
      print_delineator(names(nested_compare_lists)[i], max_length = 40, delin.type = "-")
      print(usdm::vif(pred_stack_df))
      cat("\n")
      
      
      ## Correlation
      par(mfrow = c(1,1), oma = c(0,8,8,0), mar = c(0,0,0,0), ps = 8, cex = 1.5, xpd = NA)
      plot(1, 1, xlim = c(0, ncol(pred_stack_df)-.5), ylim = c(0.5, ncol(pred_stack_df)), 
           xaxs = "i", yaxs = "i", type = "n", xaxt = "n", yaxt = "n", bty = "n", 
           ylab = "", xlab = "")
      correlation_plot(pred_stack_df, as.matrix(cor(pred_stack_df)))
    }
  }
  
  ###------ Explanatory power ------###
  if(D2adj == TRUE){
    print_delineator("D2 adjusted", max_length = 50, delin.type = "=")
    
    # New version
    for(i in 1:length_list){
      print_delineator(names(nested_compare_lists)[i], max_length = 40, delin.type = "-")
      
      bck_pts <- 10000
      list_prd <- nested_compare_lists[[i]]
      
      # Extract number of forest and non forest rasters
      na_pix <- mapply(X=1:length(list_prd), function(X) table(is.na(list_prd[[X]][])))
      index_no_forest <- which(na_pix[2,] < 1000000)
      index_forest <- which(na_pix[2,] > 1000000)
      
      d2adj <- rep(NA, length(list_prd))
      names(d2adj) <- unlist(lapply(list_prd, names))
      
      for(j in index_no_forest){
        wts_no_forest <- model_weights(df_fires, list_prd[j], backgrd_pts=bck_pts)
        df_no_forest <- prep_df_occ_var(df_fires, list_prd[j], bck_pts, na.rm=TRUE)
        glm.bi <- glm(df_no_forest$occ ~ df_no_forest[,4] + I(df_no_forest[,4]^2), family = 'binomial', weights = wts_no_forest)
        d2adj[j] <- ecospat.adj.D2.glm(glm.bi)* 100
      }
      for(f in index_forest){
        wts_forest <- model_weights(df_fires, list_prd[f], backgrd_pts=bck_pts)
        df_forest <- prep_df_occ_var(df_fires, list_prd[f], bck_pts, na.rm = TRUE)
        glm.bi <- glm(df_forest$occ ~ df_forest[,4] + I(df_forest[,4]^2), family = 'binomial', weights = wts_forest)
        d2adj[f] <- ecospat.adj.D2.glm(glm.bi)* 100
      }
      print(data.frame(d2adj = sort(d2adj, decreasing=T)))
      cat("\n")
    }
  }
  
  ###------ GLM ------###
  if(glm == TRUE){
    print_delineator("GLM assessment", max_length = 50, delin.type = "=")
    
    for(i in 1:length_list){
      ## GLM
      print_delineator(paste0("GLM - ",names(nested_compare_lists)[i]), max_length = 40, delin.type = "-")
      wts_i <<- model_weights(df_fires, nested_pred_stack[[i]], bck_pts)  # wide selection with soil moisture
      df_occ_var <- prep_df_occ_var(df_fires, nested_pred_stack[[i]], backgrd_pts=bck_pts, na.rm=TRUE)
      form <- pred_to_formula(nested_pred_stack[[i]], model_type = "glm", poly.glm.k = glm.k)  # select the glm order
      mod_glm <- glm(form, data = df_occ_var, family = 'binomial', weights = wts_i)  # modelling part
      
      if(glm.plot == TRUE){print(summary(mod_glm)) ; cat("\n")}
      if(glm.step == TRUE){
        print_delineator(paste0("GLM step - ",names(nested_compare_lists[i])), max_length = 40, delin.type = "-")
        if(glm.plot == TRUE){print(step(mod_glm, directions = 'both', trace = FALSE))}
        cat("\n")
      }
      
      ## Adjusted explained deviance
      if(ecospat.D2adj_mean == FALSE){
        print_delineator(paste0("Adj. expl. deviance - ",names(nested_compare_lists)[i]), max_length = 40, delin.type = "-")
        print(ecospat.adj.D2.glm(mod_glm)) 
      }else{  # apply n time the ecospat D2adj and returns the mean
        print_delineator(paste0("D2adj mean - ",names(nested_compare_lists)[i]), max_length = 40, delin.type = "-")
        return(mean(mapply(function(x){
          df_occ_var <- prep_df_occ_var(df_fires, nested_pred_stack[[i]], backgrd_pts=x, na.rm=TRUE)
          form <- pred_to_formula(nested_pred_stack[[i]], model_type = "glm", poly.glm.k = 3)
          mod_glm <- glm(form, data = df_occ_var, family = 'binomial', weights = wts_i)
          ecospat.adj.D2.glm(mod_glm)
        }, rep(bck_pts, D2adj_mean_rep))))  # repeat k times the d2adj and take mean
      }
    }
  }
  par(mfrow = c(1,1))
}






## correlation_plot() ---
correlation_plot <- function(predictor_dataframe, cormat){
  # Loop over the upper left half of the correlation matrix and plot the values
  # function developped by Philipp
  for(i in 1:(ncol(predictor_dataframe)-1)){
    
    if(i<ncol(predictor_dataframe)){
      text(i-.5, ncol(predictor_dataframe)+.3, colnames(predictor_dataframe)[i ], pos=2, offset=0, srt=-90) #x-axis labels
    }
    
    for(j in (i+1):ncol(predictor_dataframe)){
      # Define color code: green = OK, orange = problematic, red = big problem
      cl <- ifelse(abs(cormat[i,j]) < .7, "#91cf60", 
                   ifelse(abs(cormat[i,j]) < .9, "#fc8d59", "#d73027"))
      points(i-.5, j-.5, cex = 5, pch = 16, col = cl)  # change circle size: cex
      # Add Pearson correlation coefficients
      text(i-.5, j-.5, round(cormat[i,j], digits = 2), cex = .9)  # change size of text in circle: cex
      
      if(i==1){
        text(i-.5, j-.5, colnames(predictor_dataframe)[j], pos = 2, offset = 2) # y-axis labels
      }
    }
  }
}








## fire_return_time() ---
# This function indicates the frequency of fire/pixel depending on
# the thresholds used on the FFS modelled map and the fire occurrences.
fire_return_time <- function(raster_FFS, fire_dataset, thr_low, thr_high, nb_yr){
  raster_FFS_thr <- raster_FFS
  raster_FFS_thr[raster_FFS_thr < thr_low] <- NA
  raster_FFS_thr[raster_FFS_thr > thr_high] <- NA
  table_TF <- table(is.na(raster_FFS_thr[]))  # check non NA values
  
  if(length(table_TF)==2){  # TRUE and FALSE
    nb_pix <- table_TF[[1]]
    nb_fires <- nrow(bind_cols(df_fire, terra::extract(raster_FFS_thr, as.matrix(df_fire[c("x","y")]))) %>% na.omit())
    fire_prob <- (nb_fires/nb_pix)/nb_yr  # return time calculation
    fire_prob <- formatC(fire_prob, format = "e", digits = 2)
  }
  return(fire_prob)
}








## response_curves() ---
# Response curve (partial plots) function
# Possibility to change label names
response_curves <- function(df_fire_choice,      # fire occurrence dataset
                            list_predictor,      # list of covariate rasters
                            lab_predictors=NULL, # 2 col dataset with current and modified names if wished
                            model_type,          # glm and/or gam
                            poly.glm.k=NULL,     # vector of polynomial order values
                            s.gam=NULL,          # vector of gam smooth values
                            plot.points=FALSE){  # adding occ to RC plots
  
  ### ---- Data preparation ---
  bckg_pts <- 10000
  
  # Extracting predictor values for presences and pseudo-absences
  model_matrix <- prep_df_occ_var(df_fire_choice, list_predictor, bckg_pts, na.rm=TRUE)
  
  # Weights to manage class imbalance
  wts <<- model_weights(df_fire_choice, list_predictor, bckg_pts) 
  
  
  ### ---- Color preparation ---
  pal_glm <- rev(cartography::carto.pal("blue.pal", n1=10))  # blue for GLM
  pal_gam <- rev(cartography::carto.pal("red.pal", n1=10))  # green for GAM
  poly.glm.k <- sort(poly.glm.k)  # reorder values in case they are not
  s.gam <- sort(s.gam)
  col_glm <- col_gam <- c()
  
  if(!is.null(poly.glm.k) & !is.null(s.gam)){  # GLM and GAM models
    # here colors not to close are selected, the value 3 is to avoid too dark colors
    for(i in 0:(length(poly.glm.k)-1)){col_glm <- append(col_glm, pal_glm[3 + i*floor(10/length(poly.glm.k))])}
    for(i in 0:(length(s.gam)-1)){col_gam <- append(col_gam, pal_gam[3 + i*floor(10/length(s.gam))])}
    
    # Dataset with all color information
    col_var <- data.frame(model = c(rep("GLM", length(poly.glm.k)), rep("GAM", length(s.gam))),
                          cplx=c(paste0("k=", poly.glm.k), paste0("s=", s.gam)),
                          col = c(col_glm, col_gam))
  }
  
  if(is.null(poly.glm.k) & !is.null(s.gam)){  # only GAM model
    for(i in 0:(length(s.gam)-1)){col_gam <- append(col_gam, pal_gam[3 + i*floor(10/length(s.gam))])}
    col_var <- data.frame(model = rep("GAM", length(s.gam)), 
                          cplx=c(paste0("s=", s.gam)),
                          col = c(col_gam))
  }
  if(!is.null(poly.glm.k) & is.null(s.gam)){  # only GLM model
    for(i in 0:(length(poly.glm.k)-1)){col_glm <- append(col_glm, pal_gam[3 + i*floor(10/length(poly.glm.k))])}
    col_var <- data.frame(model = c(rep("GLM", length(poly.glm.k))), 
                          cplx=c(paste0("k=", poly.glm.k)),
                          col = c(col_glm))
  }
  
  
  ### ---- Extracting values ---
  predictor_values <- values(rast(list_predictor))
  
  # Calculates probabilities per col : minima, maxima, and medians
  var_stat <- apply(predictor_values, 2, quantile, probs = c(0,.5,1), na.rm = TRUE) 
  
  # Replicates the mean value 200 times for each predictor
  var_raw <- as.data.frame(var_stat[rep(2,200),])  
  
  
  
  ### ---- Fitting model ---
  # Prepare plotting window for each variable
  par(mfrow=mfrow_choice(length(list_predictor)), cex = 1, mar=c(5,2,2,1), oma=c(5,5,1,1)) 
  
  # Loop variables to perform response curve
  for(i in 1:length(list_predictor)){
    
    # Prepare sequence of variable values from min to max
    # only variable i (var_i[,i]) hasn't fixed values
    var_i <- var_raw
    var_i[,i] <- as.data.frame(seq(var_stat[1,i], var_stat[3,i], length.out = 200)) 
    
    # Setting up variable name
    if(!is.null(lab_predictors)){var_nm <- lab_predictors[i,2]
    }else{var_nm <- colnames(var_i)[i]}
    
    
    # New plot window for variable i
    plot(1,1, ylim=c(0,1) ,xlim=var_stat[c(1,3),i], type = "n", 
         ylab = "", xlab=var_nm, cex.lab=1.3)
    
    index_glm <- index_gam <- 1  # indices for color matching
    
    # Curves if GLM model selected
    if("glm" %in% model_type){
      col_var_glm <- col_var[which(col_var$model == "GLM"),]  # filtering GLM rows for colors
      for(k in poly.glm.k){
        # Set up formula with function
        formula <- pred_to_formula(list_predictor, model_type="glm", poly.glm.k=k, s.gam=NULL)
        
        # Fit variable to fire observations and predict proba for each point of the sequence
        mod_out <- glm(formula, data = model_matrix, family = 'binomial', weights=wts)
        prd_fire_i <- predict(mod_out, newdata = var_i, type = "response", se.fit = TRUE)
        
        # Plot predicted curve and standard error
        lines(x=var_i[,i], y=prd_fire_i$fit, lwd=1.5, col=col_var_glm$col[index_glm])
        polygon(c(var_i[,i], rev(var_i[,i]), var_i[1,i]),
                c(prd_fire_i$fit + prd_fire_i$se.fit,  rev(prd_fire_i$fit - prd_fire_i$se.fit),
                  prd_fire_i$fit[1] + prd_fire_i$se.fit[1]),
                col = "#00b7e515", border = FALSE)
        
        index_glm <- index_glm + 1
      }
    }
    
    # Curves if GAM model selected
    if("gam" %in% model_type){
      col_var_gam <- col_var[which(col_var$model == "GAM"),]
      for(s in s.gam){
        formula <- pred_to_formula(list_predictor, model_type="gam", poly.glm.k=NULL, s.gam=s)
        mod_out <- gam(formula, data = model_matrix, family = 'binomial', weights = wts)
        prd_fire_i <- predict(mod_out, newdata = var_i, type = "response", se.fit = TRUE)
        
        lines(x=var_i[,i], y=prd_fire_i$fit, lwd=1.5, col=col_var_gam$col[index_gam])
        polygon(c(var_i[,i], rev(var_i[,i]), var_i[1,i]),
                c(prd_fire_i$fit + prd_fire_i$se.fit,  rev(prd_fire_i$fit - prd_fire_i$se.fit),
                  prd_fire_i$fit[1] + prd_fire_i$se.fit[1]),
                col = "#00b7e515", border = FALSE)
        
        index_gam <- index_gam + 1
      }
    }
    
    # Add observation points to plot, if TRUE
    if(plot.points==TRUE){
      dataset_num_pres <- length(which(model_matrix[,1]==1))  # number of presences
      dataset_num_abs <- length(which(model_matrix[,1]==0))  # number of absences
      num_samp_pts <- 1/10*dataset_num_abs
      point_sampling <- c(1:dataset_num_pres, sample(dataset_num_pres:nrow(model_matrix), num_samp_pts))  # take all pres and a sample of abs
      points(x=model_matrix[point_sampling,3+i], y=model_matrix[point_sampling,1], pch = "I", col = "#00000050")
    }
    box()  # Redraw panel box
  }
  
  
  # Add Y label to final plot
  mtext("Occurrence probability", side=2, line=2, cex=2, outer=TRUE)
  
  # Add legend to final plot
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
  plot(0, 0, type = 'l', bty = 'n', xaxt = 'n', yaxt = 'n')
  legend("bottom", inset = 0,
         legend = paste0(col_var$model, " ", col_var$cplx), 
         col = col_var$col, lwd = 2, bty = "n", cex = 1.6,
         xpd = TRUE, ncol = 2)
}











## raster_rescale() ---
# Function to rescale raster values with new values
raster_rescale <- function(x, new.min = 0, new.max = 1) {
  x.min <- min(values(x, na.rm = TRUE))
  x.max <- max(values(x, na.rm = TRUE))
  new.min + (x - x.min) * ((new.max - new.min) / (x.max - x.min))
}







## mfrow_choice() ---
# Enabling responsive plot according to the number of output predicted variables
mfrow_choice <- function(num_var){
  if(num_var==1){return(c(1,1))}
  if(num_var==2){return(c(1,2))}
  if(num_var==3 | num_var==4){return(c(2,2))}
  if(num_var==5 | num_var==6){return(c(2,3))}
  if(num_var>6 & num_var<=9){return(c(3,3))}
  if(num_var>9 & num_var<=12){return(c(3,4))}
  if(num_var>12 & num_var<=16){return(c(4,4))}
  if(num_var>16 & num_var<=20){return(c(4,5))}
  if(num_var>20 & num_var<=25){return(c(5,5))}
  if(num_var>25 & num_var<=30){return(c(5,6))}
}







## print_delineator() ---
# Offers a clean console printing for separators/titles
print_delineator <- function(text, max_length, delin.type, bl.space=1){
  num_delim <- round((max_length - nchar(text) - 2*bl.space)/2)
  print(paste0(paste0(rep(delin.type, num_delim), collapse=""), paste0(rep(" ", bl.space),collapse=""), 
               text, 
               paste0(rep(" ", bl.space),collapse=""), paste0(rep(delin.type, num_delim), collapse="")))
}














