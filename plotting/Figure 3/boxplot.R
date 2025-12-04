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


plot_boxplot <- function(metric, ylab, x_dl, y_dl, x_ode, y_ode){
  lookup <- c(metric = "MAE", metric = "1-PTA")
  # Load results and convert experiments to factor (+ correct naming)
  dl_results <- read_csv(paste0(metric,"_dataframe.csv")) |>
    mutate(
      EXP = if_else(EXP == "covid", "COV-19*", 
                    if_else(EXP == "coh._rotifers_algae", "R-A\n(coh.)", 
                            if_else(EXP == "incoh._rotifers_algae", "R-A\n(incoh.)",
                                    "L-H"))),
      EXP = factor(EXP, levels=c("COV-19*", "R-A\n(coh.)", "R-A\n(incoh.)", "L-H")),
      `Dataset size` = as.factor(TS)
    ) |>
    rename(any_of(lookup))
  
  
  # Load ode baseline result and convert experiments to factor (+ correct naming)
  ode_baseline <- read_csv(paste0("../ode_baseline_",metric,".csv")) |>
    pivot_longer(everything(), names_to = "EXP", values_to = "metric") |>
    mutate(
      EXP = if_else(EXP == "covid", "COVID-19", 
                    if_else(EXP == "rotifers_algae_coherent", "Rotifers-Algae\n(coherent)", 
                            if_else(EXP == "rotifers_algae_incoherent", "Rotifers-Algae\n(incoherent)",
                                    "Lynx and Hares"))),
      EXP = factor(EXP, levels=c("COVID-19", "Rotifers-Algae\n(coherent)", "Rotifers-Algae\n(incoherent)", "Lynx and Hares")),
      x = c(0.5, 1.5, 2.5, 3.5),
      x_end = c(1.5, 2.5, 3.5, 4.5)
      ) 
  
  if(metric=="mae"){
    ode_baseline <- ode_baseline |> filter(EXP != "COVID-19")
  }
  
  scaleFUN <- function(x) sprintf("%.1f", x)
  
  (fig <- ggplot() +
    geom_hline(yintercept = 1, linewidth=1, color='#3E4989') + # This is for the DL baseline - normalized with itself -> 1
    geom_boxplot(data=dl_results, aes(x=EXP, y=metric, fill=`Dataset size`, color=`Dataset size`), alpha=0.5, size=0.75) + # all results
    geom_segment(data=ode_baseline, aes(x = x, xend = x_end, y = metric, yend = metric), linewidth=1, color='#51A537') + # ODE baselines
    annotate("label", x=x_dl, y=y_dl, label="DL baseline", color='#3E4989', label.size = NA, family="LM Roman 10", size=6) +
    annotate("label", x=x_ode, y=y_ode, label="ODE baseline", color='#51A537', label.size = NA, family="LM Roman 10", size=6) +
    scale_color_manual(values=c('#4c1d4b', '#a11a5b', '#e83f3f', '#f69c73')) +
    scale_fill_manual(values=c('#4c1d4b', '#a11a5b', '#e83f3f', '#f69c73')) +
    scale_y_continuous(labels=scaleFUN) +
    ylab(ylab) +
    theme_bw(base_size = 22) +
    theme(legend.position = "top",
          text = element_text(family="LM Roman 10"),
          axis.title.x = element_blank(),
          legend.margin=margin(0,0,0,0),
          legend.box.margin=margin(-10,-10,-10,-10),
          plot.margin =  margin(t = 8, r = 1, b = 1, l = 1, unit = "pt")))
  
}



left <- plot_boxplot("mae", ylab="Relative MAE", x_dl=1.1, y_dl=1.1 ,x_ode=3.2, y_ode=2.51) #+ theme(axis.text.x = element_blank())
right <- plot_boxplot("pta", ylab="Relative 1-PTA", x_dl=1.1, y_dl=0.93, x_ode=1.22, y_ode=1.75) #+ theme(legend.position = "none")

legend<-get_plot_component(left, "guide-box-top", return_all = FALSE)

fig <- plot_grid(left + theme(legend.position = "none", plot.margin =  margin(t = 8, r = 8, b = 1, l = 1, unit = "pt")), right + theme(legend.position = "none"), ncol = 2)

(fig_complete <- plot_grid(legend, fig, ncol=1, rel_heights = c(1, 25.5)))


ggsave("synthetic_datasets_size_boxplot_v10.pdf", fig_complete, width=10, height=6, device=cairo_pdf)

 

