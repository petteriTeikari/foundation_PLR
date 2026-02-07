changepoint.detection = function(df, method = 'meanvar') {
  
  library(changepoint)
  plot(df$time, df$pupil, type='l')
  
  options(warn = -1)
  ts = convert.vectors.to.time.series.object(t = df$time, y = df$pupil)
  options(warn = 0)
  
  if (identical(method, 'meanvar')) {
    mvalue = cpt.meanvar(ts, method="PELT") #mean changepoints using PELT  
  } else if (identical(method, 'mean')) {
    mvalue = cpt.mean(ts, method="PELT") #mean changepoints using PELT  
  } else if (identical(method, 'var')) {
    mvalue = cpt.var(ts, method="PELT") #mean changepoints using PELT  
  }
  
  cpts = attributes(mvalue)$cpts
  change_times = df$time[cpts]
  
  for (i in seq(1, length(cpts), 2)) {
    
    if (!is.na(cpts[i]) & !is.na(cpts[i+1])) {
      ind1 = cpts[i]
      ind2 = cpts[i+1]
      # cat(ind1, ' ', ind2, '\n')
      df$pupil[ind1:ind2] = NA
    }
    
  }
  
  points(df$time, df$pupil, col='red', pch = 12, cex = .3)
  
  return(df)
  
}