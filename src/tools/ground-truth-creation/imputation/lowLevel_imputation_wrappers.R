imp.amelia.wrapper = function(pupil_df, t, y, error_frac, weights_norm, filecodes, param_imp) {
  
  # for documentation, see:
  # https://cran.r-project.org/web/packages/Amelia/vignettes/amelia.pdf
  
  # take a subset and keep only time and pupil
  pupil_df_subset = pupil_df[c('time', 'pupil', 'subject')]
  a.out <- amelia(pupil_df_subset, m = 5, ts = "time", cs = "subject")
  
    # Amelia Error Code:  61 
    # There is only 1 column of data after removing the ts, cs and idvars. Cannot impute without adding polytime. 
    
    # PETTERI: In other words AMELIA is meant for multivariate time series, and we would by default have been measuring 
    # something else at the same time as the pupil size
  
}

imp.imputeTS.wrapper = function(pupil_df, t, y, error_frac, weights_norm, filecodes, 
                                method_name = 'kalman', param_imp = list(), 
                                debugON = FALSE, verbose = FALSE)  {
  
  # imputeTS offers 18 different methods for imputing univariate time series
  # see: https://pdfs.semanticscholar.org/cf38/7a44ef973ac37568a8ca482b4add12b646eb.pdf
  # and: https://github.com/SteffenMoritz/imputeTS
  
    # "In general, for most time series one algorithm out of 
    # na.kalman, na.interpolation and na.seadec will yield the best results.
  
    # Convert to a time series object 
    # https://www.statmethods.net/advstats/timeseries.html
    # https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ts.html
    ts_pupil = convert.vectors.to.time.series.object(t, y, filecodes, error_frac, weights_norm, param_imp[['fps']])

    na_found = is.na(ts_pupil)
    # sum(na_found)
    
    if (grepl('kalman', method_name)) {
      
      if (verbose) {
        cat('   .. Imputing missing data with Kalman option from imputeTS package')
      }
      
      options(warn=-1)
      if (grepl('StructTS', method_name)) {
        y_na = na.kalman(ts_pupil, model ="StructTS")
      } else if (grepl('auto.arima', method_name)) {
        y_na = na.kalman(ts_pupil, model ="auto.arima")
      } else {
        warning('option for Kalman Imputution not recognized, has to be either "StructTS" or "auto.arima", you gave = ', 
                method_name, ' is this a typo?')
      }
      options(warn=0)
      if (verbose) {
        cat('DONE!\n')
      }
      
      # TODO! Uncertainty estimate!
      
    } else if (grepl('interpolation', method_name)) {
      y_na = na.interpolation(y_ts)
      
    } else if (grepl('seadec', method_name)) {
      y_na = na.seadec(y_ts)
      
    } else if (grepl('all3', method_name)) {
      
      y_na1 = na.kalman(y_ts)
      y_na2 = na.interpolation(y_ts)
      y_na3 = na.seadec(y_ts)
      
      
    } else {
      warning('method_name "', method_name, '" not implemented yet or was this a typo?')
    }
    
    if (debugON) {
      # ind_to_plot = param_imp[['debug_series']] 
      # debug.imputation.ts(t[,ind_to_plot], ts_pupil[,ind_to_plot], y_na[,ind_to_plot], na_found[,ind_to_plot])  
    }
  
    # name the columns by subjects if the input is a matrix
    if (length(dim(y_na)) != 0) {
      colnames(y_na) = filecodes  
    } 
    
    
    # Return the imputed time series with the errors and weights
    data_out = list()
    y_na2 = data.matrix(y_na)
    y_na = as.vector(y_na2)
    data_out[['pupil']] = y_na
    
    # TODO! Now these are more like placeholders
    data_out[['error']] = error_frac
    data_out[['weights']] = weights_norm
    
    return(data_out)
}

debug.imputation.ts = function(t_t, y_ts, y_na_t, na_found_t) {
  
  # zoom
  i1 = which(t_t == 32)
  i2 = which(t_t == 40)
  
  y_na_vis = y_na_t # only display the imputed values
  y_na_vis[!na_found_t] = NA
  
  plot(t_t[i1:i2], y_ts[i1:i2])
  points(t_t[i1:i2], y_na_vis[i1:i2], col='red') 
  # Kalman gives a bit smoother imputation than other tested methods
  
}

impute.with.MissForest = function(vars_as_matrices, pupil_col = 'pupil_toBeImputed', miss_forest_parallelize) {
  
  library(missForest)
  library(doParallel)  
  
  # registerDoParallel(cores=2)  
  getDoParWorkers()
  
  # a data matrix with missing values. 
  # The columns correspond to the variables and
  # the rows to the observations.
  matrix_with_missing_value = vars_as_matrices[[pupil_col]]
  
  start_time <- Sys.time()
  filled_matrix <- missForest(matrix_with_missing_value, parallelize=miss_forest_parallelize) 
  end_time <- Sys.time(); end_time - start_time
  # missForest iteration 5 in progress...done!
  # Time difference of 2.00318 hours
  # OOBerror = 0.0566
  
  matrix_imputed <- filled_matrix$ximp
  OOBerror <- filled_matrix$OOBerror
 
  return(list(matrix_imputed, OOBerror)) 
  
}