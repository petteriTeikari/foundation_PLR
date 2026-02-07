compute.PLR.features = function(data_frame_in, bins, color, 
                                normalize_on, normalize_method, normalize_indiv_colors, 
                                baseline_period) {
  
  bin_names = bins$Name
  bin_methods = bins$Method
  bin_timingMethod = bins$StartString
  
  # Define when light was on
  verbose = FALSE # TODO! now hard-coded
  light_range = define.whenLightWasOn(data_frame_in, verbose)
    
  # Initialize as list
  features <- list()
  data_bins <- list()
  
  # Cycle through each bin
  options(warn = -1) # suppress warnings [Warning in rma(y ~ t, abs(uncert_in), method = "FE") :
                                         # Studies with NAs omitted from model fitting.]
  
  # GO THROUGH THE FEATURED DEFINED in bins.csv
  for (i in 1:length(bin_names)) {
    
    # debug
    bin_name = bin_names[i]
    method = bins$Method[i]
    timingMethod = bin_timingMethod[i]
    startTime = bins$Start[i]
    endTime = bins$End[i]
    startString = bins$StartString[i]
    
    # Get the corresponding points of the bin
    # i.e. Abstractify the rather messy code to determine this
    data_points_df = get.datapoints.per.bin(data_frame_in, timingMethod, startTime, endTime,
                                            startString, light_range, color)
    
    out = process.all.bins(data_frame_in, bin_name, method, data_points_df,
                           timingMethod, startTime, endTime,
                           bins, i, light_range, color, features, data_bins)
    
    features = out[[1]]
    data_bins = out[[2]]
    
  }
  
  options(warn = 1) # switch back the warnings
  return(list(features, data_bins))
  
} 


# SUBFUNCTION for
# GO THROUGH THE BINS HERE
process.all.bins = function(data_frame_in, bin_name, method, data_points_df, 
                            timingMethod, startTime, endTime, 
                            bins, i, light_range, color, features, data_bins) {
  
  # Compute then the feature on this subset of the whole PLR
  # depending on the method given for this feature
  feature = compute.indiv.feature(data_points_df, method, bin_name, timingMethod)
  
  # Add this to the previous features computed on previous loop iterations
  # https://stackoverflow.com/questions/17499013/how-do-i-make-a-list-of-data-frames
  features[[i]] = feature
  data_bins[[i]] = data_points_df
  
  return(list(features, data_bins))
  
}


