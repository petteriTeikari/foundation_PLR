augment.traces.for.deep.learning = function(list_in, list_full, t, y, error,
                                            classes, subject_code_augm, method = '1st_try',
                                            increase_by_nr = 10, range = c(-0.3, 0.3),
                                            train_indices = train_indices,
                                            debug_plot_per_subject = FALSE,
                                            debug_plot_per_class = TRUE,
                                            stat_correction_augm = 'normalize_mean_keepMaxConstrTheSame') {
  
  # stat_correction_augm = 'zStandardize_over_all_subj'
  # stat_correction_augm = 'zStandardize_indiv_subj'
  
  cat('Augmentation method is = ', method, '\n')
  cat('  The size of the input training data is now = ', dim(y),'\n')
  cat('     Trim the incoming data in lists to training data indices\n')
  cat('        Stastical correction for each "synthetic subject" =', stat_correction_augm, '\n')
  
  method_strings = list()
  method_str = list()
  
  for (nam in 1 : length(list_in)) {
    list_in[[nam]] = list_in[[nam]][,train_indices]
  }
  
  for (nam in 1 : length(list_full)) {
    list_full[[nam]] = list_full[[nam]][,train_indices]
  }
  
  # TODO! verify the correct lengths of all! n = 241 for the PLR dataset
  list_in_corr = dim(list_in$time)[2] == dim(y)[1]
  list_full_corr = dim(list_full$time)[2] == dim(y)[1]
  classes_corr = length(classes) == dim(y)[1]
  subjects_corr = length(subject_code_augm) == dim(y)[1]
  
  if (list_in_corr & list_full_corr & classes_corr & subjects_corr) {
    cat('')
  } else {
    warning('Discrepancies in your data dimensions, you need to do debugging :(')
  }
  
  # these are pre-computed in Matlab
  if (identical(method, 'w_matlab')) {
    matlab_augm = matlab.augm.wrapper()
  }
  
  no_of_subjects_in = dim(y)[1] # e.g. 241
  
  if (identical(method, '1st_try') | identical(method, 'w_matlab')) {
    
    for (subj in 1 : no_of_subjects_in) {
      
      i = 1
      method_str[i] = 'Input'
      
      cat('    augmenting subject #', subj, '/', dim(y)[1], '\n')
      y_subj = y[subj,] # NOTE! the matrix is transposed in relation to the list_full items
      # plot(y_subj)
      # cat('       ... trace length = ', length(y_subj), '\n')
      
      hiFreq = list_full$hiFreq[,subj]
      loFreq = list_full$loFreq[,subj] 
      noise = rowSums(cbind(list_full$noiseNorm[,subj], list_full$noiseNonNorm[,subj]), na.rm = TRUE)
      base = list_full$base_osc[,subj] 
      
      # loFreq
      i = i + 1
      multip1 = runif(1, range[1], range[2])
      y_augm1 = rowSums(cbind(y_subj, (multip1*loFreq)), na.rm = TRUE)
      method_str[i] = 'loFreq'
      
      # loFreq + hiFreq
      i = i + 1
      multip1 = runif(1, range[1], range[2])
      multip2 = runif(1, range[1], range[2])
      y_augm2 = rowSums(cbind(y_subj, (multip1*loFreq), (multip2*hiFreq)), na.rm = TRUE)
      method_str[i] = 'loFreq_hiFreq'
      
      # loFreq + noise
      i = i + 1
      multip1 = runif(1, range[1], range[2])
      multip2 = runif(1, range[1], range[2])
      y_augm3 = rowSums(cbind(y_subj, (multip1*loFreq), (multip2*noise)), na.rm = TRUE)
      method_str[i] = 'loFreq_noise'
      
      # hiFreq
      i = i + 1
      multip1 = runif(1, range[1], range[2])
      y_augm4 = rowSums(cbind(y_subj, (multip1*hiFreq)), na.rm = TRUE)
      method_str[i] = 'hifreq'
      
      # hiFreq + noise
      i = i + 1
      multip1 = runif(1, range[1], range[2])
      multip2 = runif(1, range[1], range[2])
      y_augm5 = y_subj + (multip1*hiFreq) + (multip2*noise)
      method_str[i] = 'hifreq_noise'
      
      # hiFreq + loFreq + noise
      i = i + 1
      multip1 = runif(1, range[1], range[2])
      multip2 = runif(1, range[1], range[2])
      multip3 = runif(1, range[1], range[2])
      y_augm6 = y_subj + (multip1*hiFreq) + (multip2*loFreq) + (multip3*noise)
      method_str[i] = 'hifreq_lofreq_noise'
      
      # hiFreq + loFreq + noise + base
      i = i + 1
      multip1 = runif(1, range[1], range[2])
      multip2 = runif(1, range[1], range[2])
      multip3 = runif(1, range[1], range[2])
      multip4 = runif(1, range[1], range[2])
      y_augm7 = y_subj + (multip1*hiFreq) + (multip2*loFreq) + (multip3*noise) + (multip4*base)
      method_str[i] = 'hifreq_lofreq_noise_base'
      
      # Smooth1
      i = i + 1
      loess_model_deg2 = loess(hiFreq~t, span = 0.05, degree = 2)
      residual_hi_deg2 = hiFreq - loess_model_deg2$fitted
      y_augm8 = y_subj - residual_hi_deg2
      method_str[i] = 'loess_smooth1'
      
      # Smooth2
      i = i + 1
      loess_model_lo = loess(loFreq~t, span = 0.1, degree = 2)
      residual_lo = loFreq - loess_model_lo$fitted
      y_augm9 = y_augm8 - residual_lo
      method_str[i] = 'loess_smooth2'
      
      # Smooth3
      i = i + 1
      loess_smooth = loess(y_augm9~t, span = 0.1, degree = 2)
      residual_smooth = y_augm9 - loess_smooth$fitted
      y_augm10 = y_augm9 - residual_smooth
      method_str[i] = 'loess_smooth3'
      
      no_of_cols = i
      y_new_per_subject = cbind(y_subj, y_augm1, y_augm2, y_augm3, y_augm4, y_augm5,
                                y_augm6, y_augm7, y_augm8, y_augm9, y_augm10)
      
      if (identical(method, 'w_matlab')) {
        
        parsed_list = parse.matlab.aug.per.subj(t, matlab_augm)
        matlab_data = t(parsed_list[['data']])
        y_new_per_subject = cbind(y_new_per_subject, matlab_data)
        no_of_subjects_now = dim(y_new_per_subject)[2]
        no_of_cols = parsed_list[['no_of_cols_after_matlab']]
        method_str = c(method_str, parsed_list[['methods']])
        
      }
      
      # Inspect now each augmented subject
      y_new_per_subject = inspect.augmented.subject(y_new_per_subject, method_str, debug_plot_per_subject,
                                                    stat_correction = stat_correction_augm)
      
      labels_new_per_subject = rep(classes[subj], no_of_cols)
      subjectcode_new_per_subject = rep(subject_code_augm[subj], no_of_cols)
      
      if (subj == 1) {
        
        y_out = y_new_per_subject
        labels_out = labels_new_per_subject
        subjects_out = subjectcode_new_per_subject
        method_strings = method_str
        
      } else {
        
        y_out = cbind(y_out, y_new_per_subject)
        labels_out = c(labels_out, labels_new_per_subject)
        subjects_out = c(subjects_out, subjectcode_new_per_subject)
        method_strings = c(method_strings, method_str)
        
        cat('         synthetic dataset size = ', dim(y_out), '\n')
        cat('           ... synthetic labels = ', length(labels_out), '\n')
        cat('           ... ... synthetic subject codes = ', length(subjects_out), '\n')
        
      }
      
    } # end of for loop
    # from 241 subject, we should now have 2651 synthetic subjects (11x)
    cat('\nIN THE END WE HAVE A DATASET with the size of = ', dim(y_out)[1], ' samples x', dim(y_out)[2], ' synthetic subjects')
    
    # TODO! works only for binary labels
    no_of_glaucoma_labels =  sum(labels_out %in% 'Glaucoma')
    cat('\n   with', no_of_glaucoma_labels, 'glaucoma labels (', round(100*(no_of_glaucoma_labels/length(labels_out)), digits = 2), '% of all the labels )') 
    
    # DEBUG ALL TRACES
    if (debug_plot_per_class) {
      debug.plot.y_out(t, y_out, labels_out)
    }
    
    # Check that nothing funky happened above
    if ((no_of_cols*dim(y)[1]) != (dim(y_out)[2])) {
      warning('We do not have correct number of synthetic data out?')
    }
    
    if (no_of_cols*dim(y)[1] != length(labels_out)) {
      warning('We do not have correct number of synthetic labels out?')
    }
    
    if (no_of_cols*dim(y)[1] != length(subjects_out)) {
      warning('We do not have correct number of synthetic subjects out?')
    }
  }
  
  list_out = list(y_out, labels_out, subjects_out, method_strings)
  names(list_out) = c('y', 'labels', 'subjectCodes', 'method_strings')
  
  return(list_out)
  
}

matlab.augm.wrapper = function() {
  
  if (identical(.Platform$OS.type, 'windows')) {
    base_data_path = 'C:\\Users\\petteri-sda1\\Dropbox\\manuscriptDrafts\\deepPLR\\code\\data_for_deepLearning\\archives\\'
    folder_freq_bands = file.path(base_data_path, 'SERI_PLR_Glaucoma_augmFreqBands', fsep = '\\')
    folder_smooth = file.path(base_data_path, 'SERI_PLR_Glaucoma_augmSmoothed', fsep = '\\')
  } else {
    base_data_path = '/home/petteri/Dropbox/manuscriptDrafts/deepPLR/code/data_for_deepLearning/archives/'
    folder_freq_bands = file.path(base_data_path, 'SERI_PLR_Glaucoma_augmFreqBands', fsep = .Platform$file.sep)
    folder_smooth = file.path(base_data_path, 'SERI_PLR_Glaucoma_augmSmoothed', fsep = .Platform$file.sep)
  }
  
  # TODO! check that the data actually is found from INPUT
  # i.e. base_data_path = '/home/petteri/Dropbox/manuscriptDrafts/deepPLR/code/data_for_deepLearning/archives/'
  
  data_freq_bands_list = gather.correct.traces.from(folder_in = folder_freq_bands, subject_code_augm, header_on = TRUE, dset = 'freq_bands')
  data_smooth_list = gather.correct.traces.from(folder_in = folder_smooth, subject_code_augm, header_on = FALSE, dset = 'smooth')
  
  data_freqs_matrix = data_freq_bands_list[['Matrix']]
  data_freqs_labels = data_freq_bands_list[['Labels']]
  data_freqs_methods = data_freq_bands_list[['Method Names']]
  
  data_smooth_matrix = data_smooth_list[['Matrix']]
  data_smooth_labels = data_smooth_list[['Labels']]  
  data_smooth_methods = data_smooth_list[['Method Names']]
  
  # TODO! you can't combine the rows as now we assume above, that
  # 7 consecutive traces belong to the same subject
  # matlab_data = rbind(data_freqs_matrix, data_smooth_matrix)  
  # matlab_labels = c(data_freqs_labels, data_smooth_labels)  
  # matlab_methods = c(data_freqs_methods, data_smooth_methods)  
  
  list_out = list(data_freqs_matrix, data_freqs_labels, data_freqs_methods,
                  data_smooth_matrix, data_smooth_labels, data_smooth_methods)
  
  names(list_out) = c('freqs_data', 'freqs_labels', 'freqs_methods',
                      'smooth_data', 'smooth_labels', 'smooth_methods')
  
  return(list_out)
  
}

gather.correct.traces.from = function(folder_in, subject_code_augm, header_on = TRUE, dset = NA) {
  
  filepaths = list.files(path=folder_in, pattern='*', recursive=FALSE, full.names = TRUE)
  files = list.files(path=folder_in, pattern='*', recursive=FALSE, full.names = FALSE)
  
  the_data_ind = grepl('TRAIN.txt', files)
  the_codes_ind = grepl('_SUBJECTCODES.csv', files)
  the_methods_ind = grepl('_augmMethods', files)
  
  data_in = read.csv(filepaths[the_data_ind], header = header_on)
  codes_in = read.csv(filepaths[the_codes_ind], header = FALSE) 
  dims = dim(data_in)
  no_of_subjects = dims[1]
  
  # if (!require("R.matlab")) install.packages("R.matlab"); library("R.matlab")
  # rawData = readMat(filepaths[the_methods_ind ])# Will assume version 5
  # TODO! fix the export as string .csv, as .MAT version needs to be 6 for saving strings on .MAT
  
  # Hard-coded now due to Matlab string processing greatness :(
  if (identical(dset, 'freq_bands')) {
    methods_hardcoded = c("Fatique Waves", "Respiration", "PSV HF", "PSV Total", "Alert Oscillation")
      
  } else if (identical(dset, 'smooth')) {
    methods_hardcoded = c('BilaterFilter', 'L0andGuidedFilter')
    
  } else {
    warning('Do not know yet how to parse dset =', dset)
    
  }
  
  if (no_of_subjects %% length(methods_hardcoded) != 0) {
    warning('You probably have a glitch with your header boolean as you do not have integer multiple of augmentation methods')
    if (no_of_subjects %% length(methods_hardcoded) == 1) {
      cat(' ... your header should be TRUE?, and now it is = ', header_on)
    } else if (no_of_subjects %% length(methods_hardcoded) == 4) {
      cat(' ... your header should be FALSE?, and now it is = ', header_on)
    }
  }
  
  labels_in = list()
  traces_matrix = matrix(, nrow =no_of_subjects, ncol = dims[2]-1)
  
  # reshape the df into a matrix 
  # inefficent transparent shaping
  cat('\n  Reshaping', dims[1]-1, 'subjects in data.frame into a matrix\n\t\t')
  for (subj in 1 : dims[1]) {
    
    if ((subj %% 100) == 0) {
      cat(subj, ' ')
    } 
    labels_in = c(labels_in, data_in[subj,1])
    trace_in = unlist(data_in[subj, 2:dims[2]])
    traces_matrix[subj,] = trace_in
  }
  
  method_names = rep(methods_hardcoded, dims[1]/length(methods_hardcoded))
  plot(traces_matrix[1,])
  
  list_out = list(traces_matrix, unlist(labels_in), method_names)
  names(list_out) = c('Matrix', 'Labels', 'Method Names')
  
  return(list_out)
  
}

debug.plot.y_out = function(t, y_out, labels_out) {
  
  if (is.numeric(labels_out)) {
    control_ind = labels_out == 0
    labels_out[control_ind] = 'Control'
    glauc_ind = labels_out == 1
    labels_out[glauc_ind] = 'Glaucoma'
  }
  
  glaucoma_indices = labels_out %in% 'Glaucoma'
  control_indices = labels_out %in% 'Control'
  
  y_subset1 = y_out[,control_indices]
  df_control = data.frame(t = t, y = rowMeans(y_subset1), err = apply(y_subset1, 1, sd))
  y_subset2 = y_out[,glaucoma_indices]
  df_glaucoma = data.frame(t = t, y = rowMeans(y_subset2), err = apply(y_subset2, 1, sd))
  
  ggplot() +
    geom_line(data=df_control, aes(x=t, y=y), color = 'deepskyblue3', size = 1) +
    geom_ribbon(data=df_control, aes(x = t, ymin=y-err, ymax=y+err), fill = 'deepskyblue3', alpha = 0.1) +
    geom_line(data=df_glaucoma, aes(x=t, y=y), color = 'deeppink3', size = 1) + 
    geom_ribbon(data=df_glaucoma, aes(x = t, ymin=y-err, ymax=y+err), fill = 'deeppink3', alpha = 0.1) + 
    theme_bw()
  
}

parse.matlab.aug.per.subj = function(t, matlab_augm) {

  # Feb 2019: We have 7 methods to add
  no_of_samples_matlab = dim(matlab_augm[['freqs_data']])[2]
  
  if (no_of_samples_matlab != length(t)) {
    warning("Matlab data augmentation does not have the same number of samples as the input data?!")
  }
  
  # the matlab augmentations already had a random weight assigned to them, so just copy
  
  # FREQS
  unique_matlab_methods1 = unique(matlab_augm[['freqs_methods']])
  base_subj_mlab_idx = ((subj-1)*length(unique_matlab_methods1))
  idxs1 = c(base_subj_mlab_idx+1, subj*length(unique_matlab_methods1))
  
  y_matlab1 = matlab_augm[['freqs_data']][idxs1[1]:idxs1[2],]
  labels_matlab1 = matlab_augm[['freqs_labels']][idxs1[1]:idxs1[2]]
  methods_matlab1 = matlab_augm[['freqs_methods']][idxs1[1]:idxs1[2]]
  
  # SMOOTH
  unique_matlab_methods2 = unique(matlab_augm[['smooth_methods']])
  base_subj_mlab_idx2 = ((subj-1)*length(unique_matlab_methods2))
  idxs2 = c(base_subj_mlab_idx2+1, subj*length(unique_matlab_methods2))
  
  y_matlab2 = matlab_augm[['smooth_data']][idxs2[1]:idxs2[2],]
  labels_matlab2 = matlab_augm[['smooth_labels']][idxs2[1]:idxs2[2]]
  methods_matlab2 = matlab_augm[['smooth_methods']][idxs2[1]:idxs2[2]]
  
  y_matlab = rbind(y_matlab1, y_matlab2)
  labels_matlab = c(labels_matlab1, labels_matlab2)
  methods_matlab = c(methods_matlab1, methods_matlab2)
  
  no_of_cols = i + length(unique_matlab_methods1) + length(unique_matlab_methods2)
  
  list_out = list(y_matlab, labels_matlab, methods_matlab, no_of_cols)
  names(list_out) = c('data', 'labels', 'methods', 'no_of_cols_after_matlab')
  
  return(list_out)

}

inspect.augmented.subject = function(y_new_per_subject, method_str, debug_plot_per_subject,
                                     stat_correction = 'normalize_mean') {
  
  # Now is your chance to force the synthetic subject to have certain properties
  # i.e. stats over all the augmented versions of the same subject
  y_new_per_subject_out = y_new_per_subject
  no_of_samples = dim(y_new_per_subject)[1]
  no_of_subjects = dim(y_new_per_subject)[2]
  
  if (identical(stat_correction, 'globalZ')) {
    
    subject_mean = mean(y_new_per_subject)
    subject_stdev = sd(y_new_per_subject)
    y_new_per_subject_out = (y_new_per_subject - subject_mean) / subject_stdev
  
  } else if (identical(stat_correction, 'indivZ')) {
    
    # pretty much destroys all amplitude information that the signal might have
    # as it clearly has in our PLR. You can test this or feed along with other methods
    # for stricter shape classification free from all the amplitude stuff
    
    subject_means = colMeans(y_new_per_subject)
    subject_stdevs = apply(y_new_per_subject, 2, sd)
    
    for (subj in 1 : no_of_subjects) {
    
      # maybe these would be broadcasted correctly without the vector creation
      mean_as_vector = rep(subject_means[subj], no_of_samples)
      y_new_per_subject_out[,subj] = y_new_per_subject[,subj] - mean_as_vector
    
      stdev_as_vector = rep(subject_stdevs[subj], no_of_samples)
      y_new_per_subject_out[,subj] = y_new_per_subject_out[,subj] / stdev_as_vector
      
    }
    
    
  } else if (identical(stat_correction, 'meanNorm') |
             identical(stat_correction, 'meanNorm_keepMaxConstrTheSame')) {
    
    # We want all the variations to have the same mean now, which seems
    # the most reasonable assumption, as the augmentation methods naturally
    # increase the variance, and the increased amplitude is more local due
    # the oscillatory nature of our components that we manipulate
    
    fixed_ind = 1
    fixed_max_constriction_indices = c(500, 740)
    subject_means = colMeans(y_new_per_subject)
    subject_stdevs = apply(y_new_per_subject, 2, sd)
    
    for (subj in 1 : no_of_subjects) {
      
      blue_constriction_in = min(y_new_per_subject_out[fixed_max_constriction_indices[1]:
                                                         fixed_max_constriction_indices[2], fixed_ind],
                                 na.rm = TRUE)
      
      # Subtract mean of this particular variation
      # maybe these would be broadcasted correctly without the vector creation
      mean_as_vector = rep(subject_means[subj], no_of_samples)
      y_new_per_subject_out[,subj] = y_new_per_subject[,subj] - mean_as_vector
      # the non-augmented file is the first
      # and add back the input mean
      input_mean_as_vector = rep(subject_means[fixed_ind], no_of_samples)
      # TODO! hard-coded bin now
      # see "df_feats = reshape.traces.to.feat.df(traces, time_vec, bin_name, bins, timingMethod, dataset, split_set, use_precomp)"
      # later for more consistent implementation
      
      y_new_per_subject_out[,subj] = y_new_per_subject_out[,subj] + input_mean_as_vector
      # mean(residual_PLR) # becomes zero now
      
      if (identical(stat_correction, 'meanNorm_keepMaxConstrTheSame')) {
        
        blue_constriction_augmented = min(y_new_per_subject_out[fixed_max_constriction_indices[1]:
                                                                fixed_max_constriction_indices[2], subj],
                                          na.rm = TRUE)
        
        ratio_in_augm = blue_constriction_in / blue_constriction_augmented 
        y_new_per_subject_out[,subj] = y_new_per_subject_out[,subj] * ratio_in_augm
        
        blue_constriction_corr = min(y_new_per_subject_out[fixed_max_constriction_indices[1]:
                                                                  fixed_max_constriction_indices[2], subj],
                                          na.rm = TRUE)
        
        # cat(blue_constriction_in, ' | ', blue_constriction_augmented, ' | ', blue_constriction_corr, '\n')
        
      }
      
    }
    
  } else if (identical(stat_correction, 'noCorr')) {
    cat('')
    
  } else {
    warning('Not recocgnized your stat_correction method = ', stat_correction)
  }
  
  # DEBUG EACH SUBJECT to detect possible FUNKINESS
  # useful when debugging and playign around with this, not so much
  # when you are batch processing this all
  if (debug_plot_per_subject) {
    
    input_PLR = y_new_per_subject_out[,1]
    mean_PLR = rowMeans(y_new_per_subject_out)
    residual_PLR = input_PLR - mean_PLR
    
    df = data.frame(t = t, y = mean_PLR,
                    y_orig = input_PLR,
                    y_residual = residual_PLR)
    
    # mean of all augmentations
    ggplot(df) + geom_line(aes(x = t, y = y))
    
    # orig input
    ggplot(df) + geom_line(aes(x = t, y = y_orig))
  
    # orig input - mean_of_all
    # TODO! For example you could constrain the variations in the end to 
    # to be so that the distribution from the augmentation is the same as 
    # for input?
    ggplot(df) + geom_line(aes(x = t, y = y_residual))
  
  }
  
  # TODO!
  # Save individual synthetic subjects to disk, if you would be interested for
  # example doing some "repeated measures" statistics for these later on
  # Gaussian Processes for modeling group differences, inter-individual differences, 
  # and intra-individual differences. 
  
  # Note! that PLR has significant intra-individual
  # variability and in our dataset we only have measured each individual once,
  # so these augmented dataset could be used to model that as well, or build some
  # quantitative framework to model the intra-individual differences when you actually
  # get some data.
  
  # see for example:
  # Visual Neuroscience  |   July 2012
  # Circadian and Wake-Dependent Effects on the Pupil Light Reflex in Response to Narrow-Bandwidth Light Pulses
  # Mirjam Munch; Lorette Leon; Sylvain V. Crippa; Aki Kawasaki
  # http://doi.org/10.1167/iovs.12-9494
  # cited by 52 articles:
  # https://scholar.google.com/scholar?um=1&ie=UTF-8&lr&cites=12039663125012918621
  
  return(y_new_per_subject_out)
  
  
}
