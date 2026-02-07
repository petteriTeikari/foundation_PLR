#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)
# install.packages('Cairo')
library(Cairo)   # For nicer ggplot2 output when deployed on Linux

ui <- fluidPage(
  column(width = 12, class = "well",
           
       fluidRow(
             column(width = 4, h4("ROI Selection"),
                    plotOutput("plot1", height = 300,
                               brush = brushOpts(
                                 id = "plot1_brush",
                                 resetOnNew = TRUE)
                    )
             ),
             
             column(width = 4, h4("Include inliers / Exclude outliers"),
                    plotOutput("plot2", height = 300,
                               dblclick = "plot2_dblclick",
                               brush = brushOpts(
                                 id = "plot2_brush",
                                 resetOnNew = TRUE)
                    )
                           
             ),
             
             column(width = 4, h4("Visualize the Fit"),
                    plotOutput("plot3", height = 300,
                               dblclick = "plot3_dblclick",
                               brush = brushOpts(
                                 id = "plot3_brush",
                                 resetOnNew = TRUE)
                    )
                    
             )
             
        ),
         
        fluidRow(
            column(12, h5("Placeholder")
            ),
            
            # 1st COLUMN
            column(4, 
                   # Selector for file upload
                   # https://stackoverflow.com/questions/30072481/r-shiny-setting-the-folder-which-is-opened-in-fileinput
                   fileInput('datafile', 'Choose CSV file',
                             accept=c('*.csv',
                                      'text/csv', 
                                      'text/comma-separated-values,text/plain'))
            ),
            
            # 2nd COLUMN
            column(2,
                   radioButtons("radioMode", h3("Mode of Selection"),
                                choices = list("Exclude" = 1, "Include" = 2, "Erase" = 3),
                                selected = 1)
            ),
            
            column(2,
                   h3("Buttons"),
                   actionButton("exclude_reset", "Reset"),
                   actionButton("exclude_visualize", "Visualize")
            ),
            
            # 3rd COLUMN
            column(2,
                   h3("Finalize"),
                   actionButton("finalize_savetodisk", "Save to Disk")
            )
          )
  )
)
