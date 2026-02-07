denoise.TSrepr.wrapper = function(t, ts_in, filecodes, method_name, param_den) {
  
  y = ts_in$pupil
  error = ts_in$error
  
}

denoise.pdSpecEst.wrapper = function(t, ts_in, filecodes, method_name, param_den) {
  
  y = ts_in$pupil
  error = ts_in$error
  
  # https://cran.r-project.org/web/packages/pdSpecEst/vignettes/wavelet_est_clust.html
  # TODO!
  
}

denoise.tvd.wrapper =  function(t, ts_in, filecodes, method_name, param_den) {
  
  y = ts_in$pupil
  error = ts_in$error
  
  # https://bitbucket.org/marpin/r-tvd
  # https://cran.rstudio.com/web/packages/tvd/tvd.pdf
  
  trace_to_plot = 18 # FOR LOOP ALL THE TRACES
  
  t_ts = t[,trace_to_plot]
  data_ts = ts_in$pupil[,trace_to_plot]
  error_ts = ts_in$error[,trace_to_plot]
  title_string = colnames(ts_in$pupil)[trace_to_plot]
  
  x.denoised = tvd1d(data_ts, lambda = 10, method = "Condat")
  plot.pair.diff(t_ts, data_ts, x.denoised, title_string, 'tvd1d')
  
}

denoise.gam.wrapper =  function(t, ts_in, filecodes, method_name, param_den) {
  
  y = ts_in$pupil
  error = ts_in$error
  
  # Simplish library to start with
  # https://cran.r-project.org/web/packages/robustgam/robustgam.pdf
  
  # More options in MGCV on top of which Robustgam is built
  # https://cran.r-project.org/web/packages/mgcv/mgcv.pdf
  
  # TODO!
  
}

denoise.bsmm.wrapper =  function(t, ts_in, filecodes, method_name, param_den) {
  
  # bssm: Bayesian Inference of Non-linear and
  # Non-Gaussian State Space Models in R
  # https://cran.r-project.org/web/packages/bssm/vignettes/bssm.pdf
  
  y = ts_in$pupil
  error = ts_in$error
  
  trace_to_plot = 18 # FOR LOOP ALL THE TRACES
  
  t_ts = t[,trace_to_plot]
  data_ts = ts_in$pupil[,trace_to_plot]
  error_ts = ts_in$error[,trace_to_plot]
  
  # library("bssm")
  data("nhtemp", package = "datasets")
  prior <- normal(1, 10)
  bsm_model <- bsm(y = data_ts, sd_y = error_ts, sd_level = prior,
                   sd_slope = prior)
  
  # TODO!
  
}

denoise.local_projection.wrapper =  function(t, ts_in, filecodes, method_name, param_den) {
  
  # https://rdrr.io/rforge/fractal/man/localProjection.html
  # TODO!
  
}

denoise.wmtsa.wrapper =  function(t, ts_in, filecodes, method_name, param_den, debug = FALSE) {
  
  # wavShrink Nonlinear denoising via wavelet shrinkage
  y = ts_in$pupil
  error = ts_in$error
  col_names = colnames(ts_in$pupil)
  mother_wavelet = strsplit(method_name, '_')[[1]][2]
  
  nrows_in = dim(ts_in$pupil)[1]
  ncols_in = dim(ts_in$pupil)[2]
  y_denoised = matrix(nrow=nrows_in, ncol=ncols_in)
 
  cat('Denoising subjects with mother wavelet = ', mother_wavelet, '\n')
  
  for (i in 1 : length(col_names)) {
  
    cat(col_names[i], ' ')  
    
    # t_ts = t[,i]
    data_per_subject = ts_in$pupil[,i]
    # error_ts = ts_in$error[,i]
    
    y_denoised[,i] = wavShrink(data_per_subject, wavelet=mother_wavelet,
                            n.level=ilogb(length(data_per_subject), base=2),
                            shrink.fun="hard", thresh.fun="universal", threshold=NULL,
                            thresh.scale=1, xform="modwt", noise.variance=-1.0,
                            reflect=TRUE)
  }
  cat('\n')
  cat('\n')
    
  if (debug == TRUE) {
    # trace_to_plot = which(filecodes %in% param_den[['debug_subject_to_plot']])
    trace_to_plot = which(pupil_df$subject %in% param_den[['debug_subject_to_plot']])
    title_string = colnames(ts_in$pupil)[trace_to_plot]
    plot.pair.diff(t_ts, data_ts, y2, title_string, paste('wavShrink_', mother_wavelet, sep=''))
  }
  
  denoised_out = list()
  denoised_out[['pupil']] = y_denoised
  
  # PLACEHOLDER AGAIN AS NOW UNCERTAINTY ESTIMATE DONE
  denoised_out[['error']] = ts_in$error
  
  return(denoised_out)
  
}

denoise.demo.TSrepr = function(t, ts_in, filecodes, method_name, param_den) {
  
  trace_to_plot = 18
  
  t_ts = t[,trace_to_plot]
  
  data_ts = ts_in$pupil[,trace_to_plot]
  error_ts = ts_in$error[,trace_to_plot]
  
  # DIMENSION REDUCTION -----------------------------------------------------
  
  data_DWT <- repr_dwt(data_ts, level = 1)
  data_PAA <- repr_paa(data_ts, q = 4, func = mean)
  
  # why amplitude changes?
  ratio_DWT = min(data_ts, na.rm = TRUE) / min(data_DWT, na.rm = TRUE)
  ratio_PAA = min(data_ts, na.rm = TRUE) / min(data_PAA, na.rm = TRUE)
  data_DWT = ratio * data_DWT
  data_PAA = ratio * data_PAA
  
  
  in_plot <- data.frame(Value = data_ts,
                        Time = t[,trace_to_plot], Method = 'Imputed_in')
  
  data_df_DWT <- data.frame(Value = c(data_DWT),
                            Time = seq(from = t_ts[1], to=tail(t_ts,1), length = length(data_DWT)),
                            Method = 'DWT')
  
  data_df_PAA <- data.frame(Value = c(data_PAA),
                            Time = seq(from = t_ts[1], to=tail(t_ts,1), length = length(data_PAA)),
                            Method = 'PAA')
  
  ggplot(in_plot, aes(Time, Value, color = Method)) +
    geom_line(alpha = 0.80, size = 0.8) +
    geom_line(data=data_df_DWT, aes(Time, Value, color = Method)) +
    geom_line(data=data_df_PAA, aes(Time, Value, color = Method)) +
    theme_bw()
}



