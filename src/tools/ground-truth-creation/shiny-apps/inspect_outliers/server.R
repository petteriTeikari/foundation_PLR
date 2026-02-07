server <- function(input, output) {

# BORING INIT -------------------------------------------------------------

  # Docker-compatible path detection
  # When running in Docker container, use fixed paths
  # Otherwise, try rstudioapi (RStudio) or fall back to working directory

  fsep = .Platform$file.sep

  if (Sys.getenv("DOCKER_CONTAINER") == "TRUE") {
    # Running in Docker - use fixed Shiny Server paths
    path_base <- "/srv/shiny-server/ground-truth-creation"
    cat('Running in Docker container\n')
    cat('path_base = ', path_base, '\n\n')

  } else if (requireNamespace("rstudioapi", quietly = TRUE) &&
             tryCatch(rstudioapi::isAvailable(), error = function(e) FALSE)) {
    # Running in RStudio - use rstudioapi
    library(rstudioapi)
    full_path_script = rstudioapi::getActiveDocumentContext()$path
    script.dir = strsplit(full_path_script, split = fsep, fixed=TRUE)[[1]]
    if (identical(.Platform$OS.type, 'windows')) {
      # script.dir = strsplit(full_path_script, split = '\\', fixed=TRUE)[[1]]
      # just to make sure that this is correctly split
    }
    just_the_file = tail(script.dir,1)
    cat('   --- just_the_file = ', just_the_file, '\n')
    cat('   --- --- full_path_script = ', full_path_script, '\n\n')
    script.dir = gsub(just_the_file, '', full_path_script)
    path_base = script.dir

    # remove the last separator
    if (substr(path_base, nchar(path_base), nchar(path_base)) == '/') {
      script.dir = substr(path_base, 1, nchar(path_base)-1)
    } else if (substr(path_base, nchar(path_base), nchar(path_base)) == '/') {
      script.dir = substr(path_base, 1, nchar(path_base)-1)
    }
    path_base = file.path(script.dir, '..', '..', fsep = .Platform$file.sep)

  } else {
    # Fallback - assume running from shiny-apps directory
    # Try to detect based on working directory
    wd <- getwd()
    if (grepl("inspect_outliers", wd)) {
      path_base <- file.path(wd, '..', '..', fsep = fsep)
    } else {
      # Assume we're in the ground-truth-creation directory
      path_base <- wd
    }
    cat('Running without rstudioapi\n')
    cat('path_base = ', path_base, '\n\n')
  }
  
  # Opening the default paths out
  path_configs = file.path(path_base, 'config', fsep = .Platform$file.sep)
  config_full_path = file.path(path_configs, 'paths.csv', fsep = .Platform$file.sep)
  paths_cfg = read.csv(config_full_path, header = FALSE, stringsAsFactors = FALSE)
  win_indices = paths_cfg$V2 == 'windows'
  
  if (identical(.Platform$OS.type, 'windows')) {
    paths_win = paths_cfg$V3[win_indices]
    paths_data_in = paths_win[1]
    paths_data_out = paths_win[2]
  } else {
    paths_unix = paths_cfg$V3[!win_indices]
    paths_data_in = paths_unix[1]
    paths_data_out = paths_unix[2]
  }
  
  # examples from:
  # FOR ZOOM:
  # https://gallery.shinyapps.io/105-plot-interaction-zoom/
  
  library(ggplot2)
  library(Cairo)   # For nicer ggplot2 output when deployed on Linux
  # https://gykovacsblog.wordpress.com/2017/05/15/installing-cairo-for-r-on-ubuntu-17-04/
  # apt-get install libcairo2-dev libgtk2.0-dev xvfb xauth xfonts-base libxt-dev

  # Annoying manual switch whether you want to do outlier or imputation fixing
  # TODO!
  # mode = 'outlier'
  mode = 'imputation'
  
  warning('\n\nYou are running the Quality Review app in "', mode, '" mode, correct?   
              Still at the moment you need to manually choose that on lines 60,61
              Sorry for the inconvenience of this\n')
  
  if (identical(mode, 'outlier')) {
    path_in = file.path(paths_data_out, 'outlier_free', fsep = .Platform$file.sep) 
    path_out = file.path(path_in, 'outlier_free_corrected', fsep = .Platform$file.sep) 
  } else if (identical(mode, 'imputation')) {
    path_in = file.path(paths_data_out, 'imputation_final', fsep = .Platform$file.sep) 
    path_out = file.path(path_in, 'recon_imputation_correction', fsep = .Platform$file.sep)
  }
  
  cat('REVIEWING FILES from the FOLLOWING ORDER:\n')
  cat(path_in,'\n\n')
  
  cat('SAVING THE REVIEWED FILES to:\n   ')
  cat(path_out, '\n')
  
  if (dir.exists(path_out) == FALSE) {
    cat('Directory for OUTPUT files for correction files do not exist so creating one\n')
    dir.create(path_out, showWarnings = TRUE, recursive = FALSE, mode = "0777")
  }
  
  IO_path = file.path(path_base, 'PLR_IO')
  source(file.path(IO_path, 'check_for_done_filecodes.R', fsep = .Platform$file.sep))
  files_fullpath = list.files(path=path_in, pattern='*.csv', recursive=FALSE, full.names = FALSE)

  # check undone better
  process_only_unprocessed = TRUE
  if (process_only_unprocessed) {
    check_path = file.path(path_out, fsep = .Platform$file.sep)
    indices_undone = check.for.done.filecodes(files_fullpath, check_path)
    files_fullpath = files_fullpath[indices_undone]
  }
  
  if (length(files_fullpath) == 0) {
    cat('\nYOU DO NOT HAVE ANY FILES TO REVIEW ANYMORE!\n')
    cat('You can move on\n\n')
  }
  
  filename_in = files_fullpath[1] # automatically always moves the done files away at the end of script then
  df_in = read.csv(file.path(path_in, filename_in, fsep = .Platform$file.sep))
  
  cat(paste('OPENING FILE:', filename_in, '\n'))
  cat(paste('   from:', path_in, '\n'))
  
  # Subsetting only two variables 
  if (grepl('SERI_2017', path_in)) {
    
    df_in$error = vector(, length(df_in$pupil))
    df_in$error[] = 0
    
    myvars_raw <- c("time", "denoised", "error")
    myvars_raw <- c("time_onsetZero", "pupil", "error")
    df_raw = df_in[myvars_raw]
    colnames(df_raw) <- c("time", "pupil", "error") # standardize names
    
    myvars_clean <- c("time_onsetZero", "pupil_outlier_corrected", "error")
    df_clean = df_in[myvars_clean]
    colnames(df_clean) <- c("time", "pupil", "error") # standardize names
    
  } else if (grepl('INSERM', path_in)) {
  
    myvars_raw <- c("time", "pupil_toBeImputed", "error")
    df_raw = df_in[myvars_raw]
    colnames(df_raw) <- c("time", "pupil", "error") # standardize names
    
    myvars_clean <- c("time", "pupil", "error")
    df_clean = df_in[myvars_clean]
    colnames(df_clean) <- c("time", "pupil", "error") # standardize names
    
  } else {
    
    myvars_raw <- c("time", "pupil_raw", "video_noise_total")
    df_raw = df_in[myvars_raw]
    colnames(df_raw) <- c("time", "pupil", "error") # standardize names
    
    myvars_clean <- c("time", "pupil_outlierfree", "error")
    df_clean = df_in[myvars_clean]
    colnames(df_clean) <- c("time", "pupil", "error") # standardize names
    
  }
  
  
  # Determine if this is the "2nd pass" correction, i.e. further correcting the once corrected
  # file.
  corrected_2ndPass = FALSE
  correction_for_imputation = FALSE
  output_corr_column = 'pupil_outlier_corrected'
  
  if (identical(mode, 'imputation')) {
      
    cat('MODE: Correcting the IMPUTATION\n')
    correction_for_imputation = TRUE    
    
    if (grepl('missForest', filename_in, fixed=TRUE)) {
      output_corr_column = 'missForest'
      toBeReplaced = '_missForest'
    } else {
      toBeReplaced = '_imputation.csv'
      output_corr_column = 'pupil_imputation_corrected'
    }
    
    myvars_corr <- c("time", "pupil_toBeImputed", "error")
    # myvars_corr <- c("time", "denoised", "error")
    df_corr = df_in[myvars_corr]
    colnames(df_corr) <- c("time", "pupil", "error") # standardize names
    
    myvars_clean <- c("time", "missForest", "error")
    myvars_clean <- c("time", "missForest", "error")
    df_clean = df_in[myvars_clean]
    colnames(df_clean) <- c("time", "pupil", "error") # standardize names
    
    # TODO! Rework the logic so that the imputation is actually shown in red color
    # for example so you can inspect whether the imputation itself added anything weird?
    
    # No need to display the raw anymore for 2nd pass
    df_raw = df_clean
    df_clean = df_corr
    
    filename_recon = gsub(toBeReplaced, '', filename_in) # PLR1002_BR_video_outlier_free_corrected_imputation.csv
    filename_recon = paste(filename_recon, '_CEEMD_signals', sep='')
    # filename_recon = paste(filename_recon, '_recon_recon_CEEMD_signals.csv', sep='') # PLR1002_BR_video_outlier_free_corrected_recon_recon_CEEMD_signals.csv
    
    path_recon = file.path(path_in, '..', '..', 'recon_EMD', 'IMF_fusion', fsep = .Platform$file.sep)
    fullfile = file.path(path_recon, filename_recon, fsep = .Platform$file.sep)
    exists = length(which(list.files(path_recon) == filename_recon)) > 0
    
    if (exists) {
      
      df_recon_in = read.csv(file.path(path_recon, filename_recon, fsep = .Platform$file.sep))
      
      # Now this contains the IMFs, so let's drop the first 4 IMFs to have a denoised version
      # TODO! do the combining elsewhere and read the smooth version now
      myvars_recon <- c("time", "pupil_recon", "error")
      df_recon = df_clean
      df_recon[['pupil']] = df_recon_in[['smooth']]
      
      # pupil_recon = df_recon_in$pupil_recon
      
    } else {
      
      corrected_2ndPass = FALSE
     
      myvars_corr <- c("time", "pupil_outlier_corrected", "error")
      df_corr = df_in[myvars_corr]
      colnames(df_corr) <- c("time", "pupil", "error") # standardize names
      
      # No need to display the raw anymore for 2nd pass
      df_raw = df_clean
      df_clean = df_corr
      
      myvars_recon <- c("time", "pupil_recon", "error")
      df_recon = df_clean
      
      # # TODO!
      # # now manual correspondence of the reconstructed file
      # filename_recon = gsub('.csv', '', filename_in)
      # filename_recon = paste(filename_recon, '_recon.csv', sep='')
      # 
      # path_recon = file.path(path, '..', '..', 'recon', fsep = .Platform$file.sep)
      # df_recon_in = read.csv(file.path(path_recon , filename_recon, fsep = .Platform$file.sep))
      # # pupil_recon = df_recon_in$pupil_recon
      # 
      # myvars_recon <- c("time", "pupil_recon", "error")
      # df_recon = df_recon_in[myvars_recon]
      # colnames(df_recon) <- c("time", "pupil", "error") # standardize names
      # # pupil_recon = df_recon_in$pupil_recon
      
    }
  } else {
    cat('    MODE: 1st pass correction of outlier correction\n')
  }

  # req(input$datafile)
  
  # Get the file input from the user
  observe({
    
    datafile= input$datafile
    if (is.null(datafile)) {
      return(NULL)
    }
    df_in  = read.csv(datafile$datapath)
    str('TODO! Selection not working')
  })
    
  # Start with throwing away the points which are the same both in "raw"
  # and in "clean"
  # sum(res, na.rm=TRUE)
  only_in_raw = (!is.na(df_raw$pupil) & !is.na(df_clean$pupil))
  df_raw$pupil[only_in_raw] = NA
  
  # https://stackoverflow.com/questions/14620972/how-to-combine-two-vectors-into-a-data-frame
  require(reshape2)
  
  # 1st pass correction
  if (corrected_2ndPass == FALSE) {
    df <- data.frame(df_raw$time, df_raw$pupil, df_clean$pupil)
    colnames(df) <- c('time', 'raw', 'cleaned')
    df_m = melt(df, id.vars="time")
    
    df_vis <- data.frame(df_raw$time, df_raw$pupil, df_clean$pupil)
    colnames(df_vis) <- c('time', 'raw', 'cleaned')
    
  # 2nd pass correction
  } else {
    
    # TODO! Plot the spline as well!
    df_vis <- data.frame(df_raw$time, df_raw$pupil, df_clean$pupil, df_recon$pupil)
    colnames(df_vis) <- c('time', 'cleaned', 'corrected', 'recon')
    
    df <- data.frame(df_raw$time, df_raw$pupil, df_clean$pupil)
    colnames(df) <- c('time', 'cleaned', 'corrected')
    df_m = melt(df, id.vars="time")
  }
  
  if (grepl('imputation', filename_in, fixed=TRUE)) {
    
    df_vis <- data.frame(df_raw$time, df_raw$pupil, df_clean$pupil, df_recon$pupil)
    colnames(df_vis) <- c('time', 'Outlier Algorithm', 'Manual Outlier Correction', 'EMD denoised')
  }
  
  # NOTE! Quite non-elegant, so if you know R better, this could be made a lot
  # more compact probably
  raw_NA = 1:length(df_raw$pupil)
  raw_NA = raw_NA * NA
  clean_NA = 1:length(df_clean$pupil) # should be the same size as raw
  clean_NA = clean_NA * NA
  
  df_clean = data.frame(df_raw$time, raw_NA, df_clean$pupil)
  colnames(df_clean) <- c('time', 'raw', 'cleaned')
  df_m_clean = melt(df_clean, id.vars="time")
  df_raw = data.frame(df_raw$time, df_raw$pupil, clean_NA)
  colnames(df_raw) <- c('time', 'raw', 'cleaned')
  df_m_raw = melt(df_raw, id.vars="time")
  
  # -------------------------------------------------------------------
  # Linked plots (left and right)
  ranges2 <- reactiveValues(x = NULL, y = NULL)
  
  # For storing which rows have been excluded
  # no_of_rows = nrow(df_m) / length(unique(df_m$variable))
  # Divide with the number of variables (i.e. 2 now if we have raw and outlier cleaned)
  # e.g. 2007 values per recording -> 4014 total rows -> div by 2 = 2007 rows
  no_of_rows = length(df_m)
  
  # The reactive values that we are now updating when interacting with the app
  # https://shiny.rstudio.com/tutorial/written-tutorial/lesson6/
  vals = reactiveValues(keeprows = rep(TRUE, no_of_rows),
                        exclude_rows = rep(FALSE, no_of_rows), 
                        include_rows = rep(FALSE, no_of_rows))
  
 
  
  # 3) Well how about looping through the files of a folder, so that after each disk save
  # the next file would be opened?
  # path = '/home/petteri/Dropbox/LABs/SERI/PLR_Folder/DATA_OUT/outlier_free'
  # path_out = file.path(path, '..', 'outlier_free_corrected', fsep = .Platform$file.sep)
  # list_of_files = list.files(path=path, pattern='*_free.csv', recursive=FALSE, full.names = TRUE)
    # https://stackoverflow.com/questions/39196743/interactive-directory-input-in-shiny-app-r
  
  
  output$plot1 <- renderPlot({
    
    ggplot(df_m, aes(x=time, y=value, group=variable)) +
      geom_point(size = 1, aes(colour = variable))
  })
  
  output$plot2 <- renderPlot({
    
    # We only want to work on the outlier points, and keep the raw points displayed
    # for reference, but we want to work on the "clean" vector by either adding points 
    # to it as a result of too aggressive outlier removal, or exclude points from it 
    # with the removal being too conservative
    # lean_subset = subset(df_m, subset=variable=='clean')
    
    # Plot the kept, excluded and included points as three separate data sets
    keep    <- df_m[vals$keeprows, , drop = FALSE]
    exclude <- df_m[vals$exclude_rows, , drop = FALSE]
    include <- df_m[vals$include_rows, , drop = FALSE]
    
    options(warn = -1) 
    ggplot(keep, aes(x=time, y=value, group=variable)) +
      # geom_line(aes(x = time, y=value, group = 3), color = "black", alpha = 0.40) +
      geom_point(size = 1, aes(colour = variable), alpha = 0.75) +
      geom_point(data = exclude, shape = 21, fill = NA, alpha = 0.750) +
      geom_point(data = include, shape = 25, color = 'black', alpha = 0.750) +
      coord_cartesian(xlim = ranges2$x, ylim = ranges2$y, expand = FALSE)
      
    
    
  })
  
  # LEFT PLOT, just sets a ROI for easier including/excluding
  observe({
    brush <- input$plot1_brush
    if (!is.null(brush)) {
      ranges2$x <- c(brush$xmin, brush$xmax)
      ranges2$y <- c(brush$ymin, brush$ymax)
      
    } else {
      ranges2$x <- NULL
      ranges2$y <- NULL
    }
  })
  
  # CENTER PLOT
  # When a double-click happens, check if there's a brush on the plot.
  # If so, pick the points from the brush bounds; if not, reset the zoom.
  observeEvent(input$plot2_dblclick, {
    brush <- input$plot2_brush
    
    if (!is.null(brush)) {
      
      # Toggle points that are brushed, when button is clicked
      res <- brushedPoints(df_m, input$plot2_brush, allRows = TRUE)
      no_of_TRUES = sum(res$selected_, na.rm=TRUE)
      cat('Number of all points selected =', no_of_TRUES, '\n')
      
      # now we have all the point selected by the brush, and if we are 
      # excluding, then we need to select only from the "cleaned" PLR, 
      # and when including, only from the "raw" PLR
      
      # https://stackoverflow.com/questions/41014958/ggplot-plot-interaction-exclude-prevent-excluded-points-reset
      
      # Set NAs to FALSE. As for some reason there are some?
      is_na_in_selection = is.na(res$selected)
      res$selected[c(is_na_in_selection)] = FALSE
      
      # Which elements are actually only present in the cleaned signal,
      # and not also in the raw vector
      if (input$radioMode == 1) { # EXCLUDE
        # Exclude only clean elements
        valid_mode_elements = !is.na(df_m_clean$value)
      } else if (input$radioMode == 2) { # INCLUDE
        valid_mode_elements = !is.na(df_m_raw$value)
      }
      
      # Now when both valid clean and selection overlap, we have our excluded element
      selected_per_mode = (res$selected == TRUE & valid_mode_elements == TRUE)
      no_affected_points = sum(selected_per_mode, na.rm=TRUE)
      if (input$radioMode == 1) { # EXCLUDE
        cat('  .. of which ', no_affected_points, ' point(s) are to be excluded from the "Clean PLR"\n')
        
        # For instance if you selected only one point, this should be 
        # only contrain that one point (add to the previous value, i.e
        # use logical/boolean OR operator)
        vals$exclude_rows = vals$exclude_rows | selected_per_mode
        
      } else if (input$radioMode == 2) { # INCLUDE
        cat('  .. of which ', no_affected_points, ' point(s) are to be included from the "RAW PLR"\n')
        vals$include_rows = vals$include_rows | selected_per_mode
      }
      
      # Update then the keep rows as well
      vals$keeprows = vals$keeprows | vals$include_rows | !vals$exclude_rows
      
      # For tracking that points are actually added or removed
      no_of_nans_before_df = sum(is.na(df_m$value), na.rm=TRUE)
      # no_of_nans_before_vals = sum(is.na(vals$keep), na.rm=TRUE)
        # is.na() applied to non-(list or vector) of type 'NULL'
      
      # Note! Now the exclude_rows only marks the excluded point, so 
      # we want to exclude this point from the "main data frame" df_m
      # and fit the smoothed curve to the fine-tuned pupil size vector      
      df_m$value[c(selected_per_mode)] = NA
      no_of_nans_after_df = sum(is.na(df_m), na.rm=TRUE)
      na_change = no_of_nans_after_df - no_of_nans_before_df
      
      cat('  .. .. excluded points = ', sum(vals$exclude_rows), 
                  'included points = ', sum(vals$include_rows), '\n')
      
    } else {
      cat('No brush on double-click\n')
    }

  })
  
  # https://antoineguillot.wordpress.com/2017/03/01/three-r-shiny-tricks-to-make-your-shiny-app-shines-33-buttons-to-delete-edit-and-compare-datatable-rows/
  observeEvent(input$exclude_reset,{
    str('TODO Reset!, i.e. clear all your selection(s) and start again for this file')}
  )
  
  observeEvent(input$exclude_visualize,{
    str('TODO Visualize, i.e. re-fit a spline!')}
  )
  
  # PLOT 3
  df_vis_m = melt(df_vis, id.vars="time")
  
  output$plot3 <- renderPlot({
    ggplot(df_vis_m, aes(x=time, y=value, group=variable)) +
      geom_line(size = 1, aes(colour = variable, alpha = variable))  + # the spline fit has the highest alpha
      coord_cartesian(xlim = ranges2$x, ylim = ranges2$y, expand = FALSE)
  })
  
  options(warn = 1) 
  
  # WRITE THE RESULT TO DISK
  # TODO! functionize the mess at some point
  # https://antoineguillot.wordpress.com/2017/03/01/three-r-shiny-tricks-to-make-your-shiny-app-shines-33-buttons-to-delete-edit-and-compare-datatable-rows/
  # https://stackoverflow.com/questions/34731975/how-to-listen-for-more-than-one-event-expression-within-a-shiny-eventreactive-ha
  observeEvent(input$finalize_savetodisk,{
    
    cat('\nAppend the result to the input data\n')
    
    if (corrected_2ndPass == TRUE) {
      filename_out = filename_in
    } else {
      filename_out = gsub(".csv", "_corrected.csv", filename_in)
    }
    
    path_out_with_filename = file.path(path_out, filename_out, fsep = .Platform$file.sep)
    cat(paste(' Saving it to: ', filename_out, '\n'))
    cat(paste('  to path: ', path_out, '\n'))
    
    # copy input to output
    df_in[[output_corr_column]] = df_in$pupil
    
    
    # exclude first from the conservative automation outlier removal
    
    # if no point ever has been included, the length of the boolean will be 3
    if (length(vals$exclude_rows) >= length(df_in[[output_corr_column]])) {
      no_of_values = (length(vals$exclude_rows) / 2)
      exclusion_logical_boolean = tail(vals$exclude_rows, no_of_values)
      cat('  .. .. no of excluded points = ', sum(exclusion_logical_boolean), 'out of', length(exclusion_logical_boolean), 'values \n')
      df_in[[output_corr_column]][exclusion_logical_boolean] = NA
      excluded_points = exclusion_logical_boolean
      
    } else {
      # All are FALSE
      excluded_points = vector(mode="logical", length=length(df_in[[output_corr_column]]))
    }

    # and then include back when the outlier removal was too aggressive as assessed by human annotator
    
    # if no point ever has been included, the length of the boolean will be 3
    if (length(vals$include_rows) >= length(df_in[[output_corr_column]])) {
      no_of_values = (length(vals$include_rows) / 2)
      inclusion_logical_boolean = vals$include_rows[1:no_of_values]
      cat('  .. .. no of included points = ', sum(inclusion_logical_boolean), 'out of', length(inclusion_logical_boolean), 'values \n')
      df_in[[output_corr_column]][inclusion_logical_boolean] = df_in$pupil_raw[inclusion_logical_boolean]
      included_points = inclusion_logical_boolean
      
    } else {
      # All are FALSE
      included_points = vector(mode="logical", length=length(df_in[[output_corr_column]]))
    }
      
    if (!correction_for_imputation) {
    
      if (corrected_2ndPass == FALSE) {
        
        df_in$included_points = included_points
        df_in$excluded_points = excluded_points
        
      } else {
        
        # now with the 2nd pass conditions, we already have some values
        # marked outliers and inliers so we need to add the values marked on
        # this 2nd pass to the 1st pass values
        
        cat('  .. .. 1st pass already had: excluded points = ', sum(df_in$excluded_points), 'out of', length(df_in$excluded_points), 'values \n')
        cat('  .. .. 1st pass already had: included points = ', sum(df_in$included_points), 'out of', length(df_in$included_points), 'values \n')
        
        # temp variables
        # in_excluded = df_in$excluded_points
        # in_included = df_in$included_points
        
        # if we actually made previously included point excluded now or vice versa
        # in_excluded = in_excluded - included_points
        # cat('  .. .. .. ', sum(df_in$excluded_points)-sum(in_excluded), ' previously manually excluded points came included now')
        # in_included = in_included - included_points
        # cat('  .. .. .. ', sum(df_in$included_points)-sum(in_included), ' previously manually included points came excluded now')
        
        # COMBINE LATER
        
        # OR operator
        df_in$included_points_2ndPass = as.integer(included_points)
        df_in$excluded_points_2ndPass = as.integer(excluded_points)
       
      }
    }
    
    if (correction_for_imputation) {
      
      # replace the pupil column as well
      df_in$pupil = df_in[[output_corr_column]]
      df_in$imputation_correction = df_in[[output_corr_column]]
      
      df_in$imp_included_points = included_points
      df_in$imp_excluded_points = excluded_points
      
      # convert to integer 0 and 1
      df_in$imp_included_points = as.integer(df_in$imp_included_points)
      df_in$imp_excluded_points = as.integer(df_in$imp_excluded_points)
      
      # Placeholder now. TODO! 
      # You can hand-place end points in Calc or something for now
      df_in$imp_handplaced_points = vector(mode='integer', length=length(df_in$imp_excluded_points))
      
    
    } else {
      
      # replace the pupil column as well
      df_in$pupil = df_in[[output_corr_column]]
      
      # convert to integer 0 and 1
      df_in$included_points = as.integer(df_in$included_points)
      df_in$excluded_points = as.integer(df_in$excluded_points)
      
      # Placeholder now. TODO! 
      # You can hand-place end points in Calc or something for now
      df_in$handplaced_points = vector(mode='integer', length=length(df_in$excluded_points))
    }
    
    # str(path_out_with_filename)
    
    # Write to disk
    write.csv(df_in, file=path_out_with_filename)
    
    # Move the done file to done
    # move_path = file.path(path, 'outlierCorrection_input_DONE', fsep = .Platform$file.sep)
    # if (dir.exists(move_path) == FALSE) {
    #   cat('Directory for DONE input files for correction files do not exist so creating one')
    #   dir.create(move_path, showWarnings = TRUE, recursive = FALSE, mode = "0777")
    # }
    # 
    # cat(paste('     -- Moving the input file to:',  move_path, '\n'))
    # from = file.path(path, filename_in, fsep = .Platform$file.sep)
    # to = file.path(move_path, filename_in, fsep = .Platform$file.sep)
    # file.rename(from, to)
    
    cat(paste('  \nWRITE DONE! You can open the next file!\n'))
    
    
  })
  
}


