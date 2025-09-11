if (!require("tidyverse")){
  install.packages("tidyverse")
}
if (!require("cowplot")){
  install.packages("cowplot")
}
if (!require("scales")){
  install.packages("scales")
}
if (!require("extrafont")){
  install.packages("extrafont")
}


library(tidyverse)
library(ggplot2)
library(cowplot)
library(scales)
library(extrafont) 
loadfonts()
theme_set(theme_classic(base_size = 22))

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))



# Load results and convert experiments to factor (+ correct naming)
dl_results <- read_csv(paste0("mae_dataframe.csv")) |>
  mutate(
    EXP = if_else(EXP == "covid", "COVID-19", 
                  if_else(EXP == "coh._rotifers_algae", "Rotifers-Algae\n(coherent)", 
                          if_else(EXP == "incoh._rotifers_algae", "Rotifers-Algae\n(incoherent)",
                                  "Lynx and Hares"))),
    EXP = factor(EXP, levels=c("COVID-19", "Rotifers-Algae\n(coherent)", "Rotifers-Algae\n(incoherent)", "Lynx and Hares")),
    `Dataset size` = as.factor(TS)
  )


(fig <- ggplot() +
    geom_hline(yintercept = 1, linewidth=1, color='#3E4989') + # This is for the DL baseline - normalized with itself -> 1
    geom_boxplot(data=dl_results, aes(x=EXP, y=MAE, fill=`Dataset size`, color=`Dataset size`), alpha=0.5, size=0.75) + # all results
    annotate("label", x=0.72, y=1.07, label="DL baseline", color='#3E4989', label.size = NA, family="LM Roman 10", size=6) +
    scale_color_manual(values=c('#4c1d4b', '#a11a5b', '#e83f3f', '#f69c73')) +
    scale_fill_manual(values=c('#4c1d4b', '#a11a5b', '#e83f3f', '#f69c73')) +
    ylab("Relative MAE") +
    theme_bw(base_size = 22) +
    theme(legend.position = "top",
          text = element_text(family="LM Roman 10"),
          axis.title.x = element_blank(),
          legend.margin=margin(0,0,0,0),
          legend.box.margin=margin(-10,-10,-10,-10),
          plot.margin =  margin(t = 8, r = 1, b = 1, l = 1, unit = "pt")))
  


ggsave("synthetic_datasets_size_boxplot_v11.pdf", fig, width=10, height=5.5, device=cairo_pdf)

 

