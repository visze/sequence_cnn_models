library(tidyverse)
library(cowplot)
library(optparse)

option_list <- list(
    make_option(c("-i", "--input"), type = "character",
        help = "Input table. Multiple tables sebarated by comma.",
        dest = "input"),
    make_option(c("-r", "--region"), type = "character",
        help = "name of the region to plot.",
        dest = "region"),
    make_option(c("-o", "--output"), type = "character",
        help = "Output file of the SatMut plot")
)

arguments <- parse_args(OptionParser(option_list = option_list), positional_arguments = TRUE) # nolint

opt <- arguments$options

if (!"input" %in% names(opt)) {
  stop("--input parameter must be provided. See script usage (--help)")
}
if (!"region" %in% names(opt)) {
  stop("--region parameter must be provided. See script usage (--help)")
}
if (!"output" %in% names(opt)) {
  stop("--output parameter must be provided. See script usage (--help)")
}

data <- read_tsv(opt$input)

print(data %>% head)

standard_SatMut_region_style <- function() {
  ## styles
  size.text = 10;
  size.title = 12;
  size.line = 1.2;
  size.geom_line = 1;
  size.geom_point = 2
  standard_style <- theme_bw() + theme(plot.title = element_text(size = size.title, face="bold",hjust = 0.5),
                                       panel.grid.major = element_blank() , panel.grid.minor = element_blank(), panel.border = element_blank(),
                                       axis.text = element_text(colour = "black",size=size.text), axis.title = element_text(colour = "black",size=size.title), axis.ticks = element_line(colour = "black", linewidth=1), axis.line.y = element_line(color="black", linewidth = size.line), axis.line = element_line(colour = "black", linewidth=size.line),
                                       legend.key =  element_blank(), legend.text = element_text(size=size.text),
                                        legend.position="top", legend.box.just = "left",  legend.background = element_rect(fill = "transparent", colour = "transparent"), legend.margin = margin(0, 0, 0, 0),
                                       legend.key.size = unit(2, 'lines'), legend.title=element_text(size=size.text))+
    theme(axis.line.x = element_line(color="black", linewidth = size.line))
}

modify.filterdata <- function(data,barcodes=10, threshold=1e-5) {
  data <- data %>% # Position        Ref     Alt     ISM_delta       satmut_coefficient      pvalue  BCs
    filter(BCs >= barcodes) %>% 
    mutate(significance=if_else(pvalue<threshold,"Significant", "Not significant")) %>%
    mutate(printpos=if_else(Alt=="A",as.double(Position)-0.4,if_else(Alt=="T",as.double(Position)-0.2, if_else(Alt=="G",as.double(Position)+0.0,if_else(Alt=="C",as.double(Position)+0.2,as.double(Position)+0.4)))))
  return(data)
}


defaultColours=c("A"="#0f9447","C"="#235c99","T"="#d42638","G"="#f5b328","-"="#cccccc", 
                         "Significant"="#005500","Not significant"="red")
colorblindColors=c("A"="#1B9E77","C"="#7570B3","T"="#D95F02","G"="#E6AB02","-"="#A6761D", 
                   "Significant"="#66A61E","Not significant"="#E7298A")
getPlot <- function(data, score_name, y_name, colourPalette="default", legend=TRUE) {
  colours <- defaultColours
  
  if (colourPalette == "colorblind") {
    colours <- colorblindColors
  }
  
  refs <- data$Ref %>% unique()
  aesRefsValues <- c(if_else("A" %in% refs,15,NaN),if_else("C" %in% refs,16,NaN),if_else("G" %in% refs,17,NaN),if_else("T" %in% refs,18,NaN))
  aesRefsShape <- c(if_else("A" %in% refs,0,NaN),if_else("C" %in% refs,1,NaN),if_else("G" %in% refs,2,NaN),if_else("T" %in% refs,5,NaN))
  aesRefsValues <- aesRefsValues[!is.na(aesRefsValues)]
  aesRefsShape <- aesRefsShape[!is.na(aesRefsShape)]
  
  alts <- data$Alt %>% unique()
  sigs <- data$significance %>% unique()
  aesSize<-c(rep(7,length(alts)), rep(3,length(sigs)))
  aesLine<-c(rep(0,length(alts)), rep(1,length(sigs)))
  aesShape<-c(if_else("A" %in% alts,15,NaN),if_else("C" %in% alts,16,NaN),if_else("G" %in% alts,17,NaN),if_else("T" %in% alts,18,NaN), rep(32,length(sigs)))
  aesShape <- aesShape[!is.na(aesShape)]
  altBreaks<-c(as.character(alts), sigs)
  
  data <- data %>% select(printpos,all_of(score_name),significance,Alt,Ref) %>% dplyr::rename(Position=printpos)
  p <- ggplot() +
    geom_segment(data = data, aes(x=Position, xend=Position,y=0,yend=.data[[score_name]], colour=significance), linewidth=0.3, show.legend = TRUE) +
    geom_point(data= data, aes(x=Position,y=.data[[score_name]],colour=Alt, shape=Ref), size=1, show.legend = TRUE) +
    scale_shape_manual("",values=aesRefsValues, guide=guide_legend(override.aes = list(size=1, linetype=0, shape=aesRefsShape),nrow=2)) +
    scale_colour_manual("", values = colours, breaks=altBreaks, labels=altBreaks,
                        guide= guide_legend(override.aes = list(size=aesSize, linetype=aesLine, shape=aesShape))) +
    labs(x = "Position", y= y_name) + standard_SatMut_region_style()
  if (!legend){
    p <- p + theme(legend.position="none")
  }
  return(p)
}

data_filter <- modify.filterdata(data)



p_satmut <- getPlot(data_filter, "satmut_coefficient", "Log2 variant effect")
p_ism <- getPlot(data_filter, "ISM_delta", "ISM delta", legend=FALSE)

plot_final <- plot_grid(p_satmut,p_ism, 
          labels = c("Saturation mutagenesis MPRA", "ISM scores"),
          nrow=2)

# now add the title
title <- ggdraw() + 
  draw_label(
    opt$region,
    fontface = 'bold',
    x = 0,
    hjust = -12
  ) +
  theme(
    # add margin on the left of the drawing canvas,
    # so title is aligned with left edge of first plot
    plot.margin = margin(0, 0, 0, 7)
  )
plot_final <- plot_grid(
  title, plot_final,
  ncol = 1,
  # rel_heights values control vertical title margins
  rel_heights = c(0.1, 1)
)

ggsave(opt$output, plot_final, width=18, height=15)