library(ggplot2)
library(reshape2)
if (!require("moments")) install.packages("moments"); library("moments")
library(Cairo)   # For nicer ggplot2 output when deployed on Linux
  # https://gykovacsblog.wordpress.com/2017/05/15/installing-cairo-for-r-on-ubuntu-17-04/
  # apt-get install libcairo2-dev libgtk2.0-dev xvfb xauth xfonts-base libxt-dev

# Docker-compatible path detection
# When running in Docker container, use fixed paths
# Otherwise, try rstudioapi (RStudio) or fall back to working directory

fsep = .Platform$file.sep

if (Sys.getenv("DOCKER_CONTAINER") == "TRUE") {
  # Running in Docker - use fixed Shiny Server paths
  path_base <- "/srv/shiny-server/ground-truth-creation"
  base_dir <- path_base
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

  # remove the last separator
  if (substr(script.dir, nchar(script.dir), nchar(script.dir)) == '/') {
    script.dir = substr(script.dir, 1, nchar(script.dir)-1)
  } else if (substr(script.dir, nchar(script.dir), nchar(script.dir)) == '/') {
    script.dir = substr(script.dir, 1, nchar(script.dir)-1)
  }
  path_base = file.path(script.dir, '..', '..', fsep = .Platform$file.sep)
  base_dir = path_base

} else {
  # Fallback - assume running from shiny-apps directory
  # Try to detect based on working directory
  wd <- getwd()
  if (grepl("inspect_EMD", wd)) {
    path_base <- file.path(wd, '..', '..', fsep = fsep)
  } else {
    # Assume we're in the ground-truth-creation directory
    path_base <- wd
  }
  base_dir <- path_base
  cat('Running without rstudioapi\n')
  cat('path_base = ', path_base, '\n\n')
}

recon_dir = file.path(base_dir, 'PLR_reconstruction', 'subfunctions', fsep = .Platform$file.sep)
IO_dir = file.path(base_dir, 'PLR_IO', fsep = .Platform$file.sep)

# Opening the default paths out
path_configs = file.path(path_base, 'config', fsep = .Platform$file.sep)
config_full_path = file.path(path_configs, 'paths.csv', fsep = .Platform$file.sep)
paths_cfg = read.csv(config_full_path, header = FALSE, stringsAsFactors = FALSE)
win_indices = paths_cfg$V2 == 'windows'

if (identical(.Platform$OS.type, 'windows')) {
  paths_win = paths_cfg$V3[win_indices]
  paths_data_in = paths_win[2]
  paths_data_out = paths_win[2]
  
} else {
  paths_unix = paths_cfg$V3[!win_indices]
  paths_data_in = paths_unix[1]
  paths_data_out = paths_unix[2]
}

paths_data_in = file.path(paths_data_in, 'recon_EMD', fsep = .Platform$file.sep)

pattern = '*.csv'
# path_out = file.path(path, '..', 'reconstructed', fsep = .Platform$file.sep)
path_out = file.path(paths_data_in, 'IMF_fusion', fsep = .Platform$file.sep)
move_path = file.path(paths_data_in, 'DONE', fsep = .Platform$file.sep) # move input to, when done

# Source the subfunctions
source(file.path(recon_dir, 'post_process_decomposition_IMFs.R', fsep = .Platform$file.sep))
source(file.path(IO_dir, 'export_pupil_dataframe_toDisk.R', fsep = .Platform$file.sep))
source(file.path(IO_dir, 'check_for_done_filecodes.R', fsep = .Platform$file.sep))

cat('\nDATA IN: ', paths_data_in)
cat('\nDATA OUT: ', path_out)
cat('\n... moving done files to: ', move_path, '\n\n')


server = function(input, output, session) {
    
  # DATA --------------------------------------------------------------------

    # INPUT
    files_fullpath = list.files(path=paths_data_in, pattern=pattern, recursive=FALSE, full.names = TRUE)
    
    if (length(files_fullpath) == 0) {
      warning('No input files were found from DATA IN = "', paths_data_in, '"')
    }
    
    # check undone better
    process_only_unprocessed = TRUE
    if (process_only_unprocessed) {
      indices_undone = check.for.done.filecodes(files_fullpath, path_out)
      files_fullpath = files_fullpath[indices_undone]
    }
    
    filename_in = files_fullpath[1] # automatically always moves the done files away at the end of script then
    filecode = strsplit(tail(strsplit(filename_in, .Platform$file.sep)[[1]],1), '_')[[1]][1]
    cat(paste('Input file:', filename_in, '\n'))
    
    if (is.na(filename_in)) {
      warning('Well we have no input filename to open as no files were found from input path')
    }
    
    # Read in
    df_CEEMD = read.csv(filename_in)
   
    cleaned = pre.clean.EMD.df(df_CEEMD)
      df_IMFs = cleaned[[1]] 
      names_IMF = cleaned[[2]]
      names_IMF_in = cleaned[[3]]
      number_of_IMFs = cleaned[[4]]
    
    # Make IMFs plottable into a single ggplot plot with baseline offsets
    # df_plot_IMFs = add.offsets.for.plt(df_IMFs, names_IMF)
    
      col_max = apply(abs(df_IMFs[]),2,max)
      largest_col_value = max(col_max)
      col_height = 2*round(largest_col_value, digits=0)
      plot_height = length(col_max) * col_height  
    
      df_plot_IMFs = list()
      baseline_to_add = vector(mode='numeric', length(names_IMF_in))
      for (i in 1 : length(df_IMFs)) {
        baseline_to_add[i] = -1 * (col_height * (i-1))
        df_plot_IMFs[[names_IMF[i]]] = df_IMFs[[names_IMF_in[i]]] + baseline_to_add[i]
      }
      
      df_plot_IMFs[['time']] = df_CEEMD$time
      df_plot_IMFs = data.frame(df_plot_IMFs)
      
      IMFs_plot = melt(df_plot_IMFs, id = 'time')
    
    # define input
    # if (grepl('loFreq', path)) {
    #   input_type = 'loFreq'
    #   components = c('noise', 'loFreq_hi', 'loFreq_lo', 'base')
    # } else if (grepl('hiFreq', path)) {
    #   input_type = 'hiFreq'
    #   components = c('noise', 'hiFreq_hi', 'hiFreq_lo', 'base')
    # } else if (grepl('noise', path)) {
    #   input_type = 'noise'
    #   components = c('noise', 'spikes', 'base')
    # } else {
    input_type = '1stPass'
    components = c('noiseNorm', 'noiseNonNorm', 'hiFreq', 'loFreq', 'base')
    
    # Estimate the most likely combining of IMFs
    IMF_index_estimates = estimate.imf.combination.indices(df_IMFs, input_type) #, path = path)
    
    # convert to radiobutton selections
    IMF_radiobutton_indices = IMF.indices.into.radiobutton.indices(IMF_index_estimates, df_IMFs, input_type)
    
    # convert back
    IMF_index_estimates = IMF.indices.from.radiobutton.indices(IMF_radiobutton_indices, df_IMFs, components, input_type)
    
    # All IMFs and residue combined makes the input
    input_signal = rowSums(df_IMFs)
    df_input = data.frame(x = df_CEEMD$time, pupil = input_signal)
    
    # denoised signal
    smooth_indices = vector(, length = length(df_IMFs))
    smooth_indices[] = TRUE
    noise_indices = IMF_index_estimates$noise
    smooth_indices[noise_indices] = FALSE
    smooth_signal = df_IMFs[smooth_indices]
    smooth_signal = rowSums(smooth_signal)
    df_input[['denoised']] = smooth_signal
    
    signals = generate.signals.from.indices(df_IMFs, IMF_index_estimates, components)
    signals_df = data.frame(signals)
    signals_df[['time']] = df_CEEMD$time
    signals_df_plot = melt(signals_df, id = 'time')

    cat('                    ', filecode, '\n')

  # BEHAVIOR SETUP ----------------------------------------------------------

    # Track how user interacts with the radio buttons?
    # https://stackoverflow.com/questions/40631788/shiny-observe-triggered-by-dynamicaly-generated-inputs
    # https://stackoverflow.com/questions/41727621/shiny-switch-on-input-from-radio-buttons-causes-error-about-reactive-context
    # https://stackoverflow.com/questions/45315106/shiny-how-to-send-updated-data-with-renderui-and-eventreactive
    # https://community.rstudio.com/t/modularizing-an-app-with-dynamic-inputs-renderui/1454/2
    # https://stackoverflow.com/questions/40631788/shiny-observe-triggered-by-dynamicaly-generated-inputs
    
    # TODO!
      # None of the plots are updated now when you click the radiobuttons!
      
    # Linked plots (2nd and 3rd Column)
    ranges2 <- reactiveValues(x = NULL, y = NULL)
    
    # TODO!
      # Now the y-values are all wrong in the IMF plot, due to the baseline fix
      
    # MONITOR the IMF Plot for brush Zoom
    observe({
      brush <- input$plotIMF_brush
      if (!is.null(brush)) {
        ranges2$x <- c(brush$xmin, brush$xmax)
        ranges2$y <- c(brush$ymin, brush$ymax)
        
      } else {
        ranges2$x <- NULL
        ranges2$y <- NULL
      }
    })
    

  # PLOTTING ----------------------------------------------------------------
    
    # IMF SELECTION TABLE
    # 1st COLUMN
    
      # INITIALIZE THE RADIO BUTTONS
      # https://groups.google.com/forum/#!topic/shiny-discuss/xW8f5g5gm4s
      output$DynamicRadioButtons <- renderUI({
        LL <- vector("list",number_of_IMFs)
        
        # get the initial selections from IMF_index_estimates
        for(i in 1:number_of_IMFs){
          LL[[i]] <- list(radioButtons(inputId = names_IMF[i], label = names_IMF[i], 
                                       choices = components, selected = components[IMF_radiobutton_indices[i]], inline=T))
        }       
        return(LL)                      
      })
    
      # Save button
      
      # TODO!
      # Do not allow the save if you do not have anything on hiFreq, loFreq and base,
      # the annotator in this case did not pay attention, and this will cause problems
      # further down the line
    
      observeEvent(input$button_save, {
        cat('SAVE\n')
        # https://stackoverflow.com/questions/35022021/shiny-get-the-selected-radios-button-selected-value
        
        output_matrix = matrix(nrow = length(names_IMF), ncol = 3)
        for (i in 1 : length(names_IMF)) {
          id = names_IMF[[i]]
          # cat(input[[id]], '\n')
          output_matrix[i, 1] = id
          component_name = input[[id]]
          output_matrix[i, 2] = which(components %in% component_name)
          output_matrix[i, 3] = input[[id]]
        }
        
        str(output_matrix)
        
        col_names = c('Input_IMF', 'Input Index', 'Output_Signal')
        colnames(output_matrix) = col_names
        output_mapping = data.frame(output_matrix)
        names_selected = output_mapping[[col_names[[3]]]]
        indices = indices.from.component.names(names_selected, components)
        output_mapping = data.frame(output_matrix)
          # str(names_selected)
          # str(indices)
        
        str(output_mapping)
        
        IMF_index_estimates = IMF.indices.from.radiobutton.indices(indices, df_IMFs, components)
        signals = generate.signals.from.indices(df_IMFs, IMF_index_estimates, components)
        
        # smooth signal from non-noise values
        if (grepl('1stPass', input_type)) {
          smooth_list = signals[c(3,4,5)]  
          
        } else if (grepl('loFreq', input_type)) {
          smooth_list = signals[c(2,3,4)]  
          
        } else if (grepl('hiFreq', input_type)) {
          smooth_list = signals[c(2,3,4)]  
          
        } else if (grepl('noise', input_type)) {
          smooth_list = signals[c(2,3)]  
          
        }
        
        smooth = vector(,length(smooth_list[1]))
        for (i in 1 : length(smooth_list)) {
          
          temp = cbind(smooth, smooth_list[[i]])
          smooth = rowSums(temp, na.rm = TRUE)
        }
        
        
        
        # add to signals
        signals[['denoised']] = smooth
        signals_df = data.frame(signals) # and to dataframe
        
        # save to disk
        just_filename = paste0(filecode, '.csv')
        
        # if (grepl('SERI_2017', path)) {
        #   color = strsplit(tail(strsplit(filename_in, .Platform$file.sep)[[1]],1), '_')[[1]][2]
        #   just_filename = paste0(filecode, '_', color, '.csv')
        # }
        
        cat('Writing to folder = ', path_out, '\n')
        export.pupil.dataframe.toDisk(output_mapping, just_filename, path_out, 'mapping')
        export.pupil.dataframe.toDisk(signals_df, just_filename, path_out, 'signals')
        
        
        # Move the done file to done
        just_filename_in = tail(strsplit(filename_in, .Platform$file.sep)[[1]],1)
        
        cat(paste('     -- Moving the input file to:',  move_path, '\n'))
        from = file.path(paths_data_in, just_filename_in, fsep = .Platform$file.sep)
        to = file.path(move_path, just_filename_in, fsep = .Platform$file.sep)
        
        # check if they exist or need to be created
        if (dir.exists(move_path) == FALSE) {
          cat('Creating the subdirectory for the DONE: ', move_path)
          dir.create(move_path, showWarnings = TRUE, recursive = FALSE, mode = "0777")
        }
        
        file.rename(from, to)
        
        cat(paste('  WRITE DONE! You can open the next file!\n'))
        
      })
    
    output$plot_input <- renderPlot({
      ggplot(df_input, aes(x)) +
        geom_line(aes(y = pupil, colour = "Pupil")) + 
        geom_line(aes(y = denoised, colour = "Denoised"))
    })
    
    # IMF PLOT
    # 2nd COLUMN
      
      output$plotIMF <- renderPlot({
        ggplot(IMFs_plot, aes(time, value, colour=variable)) + geom_line() +
          scale_y_continuous(breaks=baseline_to_add, labels=names_IMF) + 
          theme(legend.position='none') +
          coord_cartesian(xlim = ranges2$x, ylim = ranges2$y, expand = FALSE)
      })
    
    # COMBINED IMFs 
    # 3rd COLUMN
      
      output$plotComp_base <- renderPlot({
        if (1 == 1) {
          ggplot(signals_df, aes(time, base)) + geom_line() +
            coord_cartesian(xlim = ranges2$x, ylim = ranges2$y, expand = FALSE)
        }
      })
      
      
      if (grepl('1stPass', input_type)) { # | (grepl('pupil', path)) ) {
      
        output$plotComp_loFreq <- renderPlot({
          ggplot(signals_df, aes(time, loFreq)) + geom_line() +
            coord_cartesian(xlim = ranges2$x, ylim = ranges2$y, expand = FALSE)
        })
        
        output$plotComp_hiFreq <- renderPlot({
          ggplot(signals_df, aes(time, hiFreq)) + geom_line() +
            coord_cartesian(xlim = ranges2$x, ylim = ranges2$y, expand = FALSE)
        })
          
        output$plotComp_noiseNonGaussian <- renderPlot({
          ggplot(signals_df, aes(time, noiseNonNorm)) + geom_line() +
            coord_cartesian(xlim = ranges2$x, ylim = ranges2$y, expand = FALSE)
        })
        
        output$plotComp_noise <- renderPlot({
          ggplot(signals_df, aes(time, noiseNorm)) + geom_line() +
            coord_cartesian(xlim = ranges2$x, ylim = ranges2$y, expand = FALSE)
        })
        
      } else if (grepl('loFreq', input_type)) {
        
        output$plotComp_loFreq <- renderPlot({
          ggplot(signals_df, aes(time, loFreq_lo)) + geom_line() +
            coord_cartesian(xlim = ranges2$x, ylim = ranges2$y, expand = FALSE)
        })
        
        output$plotComp_hiFreq <- renderPlot({
          ggplot(signals_df, aes(time, loFreq_hi)) + geom_line() +
            coord_cartesian(xlim = ranges2$x, ylim = ranges2$y, expand = FALSE)
        })
        
        output$plotComp_noise <- renderPlot({
          ggplot(signals_df, aes(time, noise)) + geom_line() +
            coord_cartesian(xlim = ranges2$x, ylim = ranges2$y, expand = FALSE)
        })
        
      } else if (grepl('hiFreq', input_type)) {
        
        output$plotComp_loFreq <- renderPlot({
          ggplot(signals_df, aes(time, hiFreq_lo)) + geom_line() +
            coord_cartesian(xlim = ranges2$x, ylim = ranges2$y, expand = FALSE)
        })
        
        output$plotComp_hiFreq <- renderPlot({
          ggplot(signals_df, aes(time, hiFreq_hi)) + geom_line() +
            coord_cartesian(xlim = ranges2$x, ylim = ranges2$y, expand = FALSE)
        })
        
        output$plotComp_noise <- renderPlot({
          ggplot(signals_df, aes(time, noise)) + geom_line() +
            coord_cartesian(xlim = ranges2$x, ylim = ranges2$y, expand = FALSE)
        })
        
      } else if (grepl('noise', input_type)) {
        
        output$plotComp_noise <- renderPlot({
          ggplot(signals_df, aes(time, spikes)) + geom_line() +
            coord_cartesian(xlim = ranges2$x, ylim = ranges2$y, expand = FALSE)
        })
        
        output$plotComp_hiFreq <- renderPlot({
          ggplot(signals_df, aes(time, noise)) + geom_line() +
            coord_cartesian(xlim = ranges2$x, ylim = ranges2$y, expand = FALSE)
        })
        
      }

    
}
