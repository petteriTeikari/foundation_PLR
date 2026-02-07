library(shiny)
library(Cairo)   # For nicer ggplot2 output when deployed on Linux
# library(DT)

imf_plot_height = 720
composite_height = floor(imf_plot_height / 5)

ui <- fluidPage(
    
    column(width = 12, class = "well",
           
           fluidRow(
             
             column(width = 4, h4("Select Components"),
                    
                    # creating this dynamically as we do not know how many IMFs 
                    # we are going to have per file
                    # https://groups.google.com/forum/#!topic/shiny-discuss/xW8f5g5gm4s
                    uiOutput("DynamicRadioButtons"),
                   
                    column(width = 4, h5("Buttons"),
                           actionButton("button_save", "SAVE")
                    ), # end of first inner column
                    
                    column(width = 8, h5("PLOT INPUT"),
                           plotOutput("plot_input", height = composite_height)
                    ) # end of second inner column
                    
             ), # end of column
             
             column(width = 4, h4("IMFs"),
                    plotOutput("plotIMF", height = imf_plot_height,
                               brush = brushOpts(id = "plotIMF_brush",
                                                 resetOnNew = TRUE)
                    ) # end of plotOutput
             ), # end of column
             
             column(width = 4, h4("Composites"),
                    
                    plotOutput("plotComp_base", height = composite_height),
                    plotOutput("plotComp_loFreq", height = composite_height),
                    plotOutput("plotComp_hiFreq", height = composite_height),
                    plotOutput("plotComp_noiseNonGaussian", height = composite_height),
                    plotOutput("plotComp_noise", height = composite_height)
             ) # end of column
             
           ) # end of fluidRow
           
    ) # end of column
    
) # end fluidPage 