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
if (!require("viridis")){
  install.packages("viridis")
}
if (!require("grid")){
  install.packages("grid")
}
if (!require("gridExtra")){
  install.packages("gridExtra")
}

library(tidyverse)
library(ggplot2)
library(cowplot)
library(scales)
library(extrafont) 
library(viridis)
library(grid)
library(gridExtra)
loadfonts()
theme_set(theme_classic(base_size = 12))

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

dl_baseline <- read_csv("heatmap_baseline.csv")

covid <- read_csv("heatmap_data/covid.csv") |>
  mutate(MAE = MAE / dl_baseline$covid)
coh_rot_algae <- read_csv("heatmap_data/rotifers_algae_coherent.csv") |>
  mutate(MAE = MAE / dl_baseline$rotifers_algae_coherent)
incoh_rot_algae <- read_csv("heatmap_data/rotifers_algae_incoherent.csv") |>
  mutate(MAE = MAE / dl_baseline$rotifers_algae_incoherent)
lynx_hares <- read_csv("heatmap_data/lynx_hares.csv") |>
  mutate(MAE = MAE / dl_baseline$lynx_hares)



prepare_df <- function(df){
  df |>
    mutate(TS = as.factor(as.integer(TS)),
           IC = if_else(IC==0, "S", 
                        if_else(IC==1, "M", 
                                if_else(IC==2, "L", "XL"
                                        )
                                )
                        ),
           IC = factor(IC, levels=c("S", "M", "L", "XL")),
           KP = if_else(KP==0, "S", 
                        if_else(KP==1, "M", 
                                if_else(KP==2, "L", "XL"
                                )
                        )
           ),
           KP = factor(KP, levels=c("S", "M", "L", "XL")),
           )
}

t <- quantile(lynx_hares$MAE, probs = c(0, 0.5, 1))
round(as.numeric(t), 1)
median(lynx_hares$MAE)

plot_col <- function(df, name){
  ggplot(data=df, aes(x=KP, y=IC, fill=MAE)) +
  geom_tile() +
  scale_fill_viridis(option = "rocket", direction = -1, breaks=round(as.numeric(quantile(df$MAE, probs = c(0, 0.5, 0.7))),1)) + 
  facet_wrap(~TS, 
             ncol = 1, 
             strip.position="right") +
  theme(axis.line = element_blank(),
        legend.position = "top",
        axis.title = element_blank(),
        axis.text.y = element_blank(),
        axis.text.x = element_text(size=11),
        axis.ticks.y = element_blank(),
        legend.text = element_text(size=11),
        plot.margin = margin(0,0,0,0, "pt"),
        strip.background = element_blank(),
        legend.key.width=unit(1.3,"lines"),
        legend.title.align=0.5,
        legend.box.margin=margin(0,-10,-10,-10),
        text = element_text(family="LM Roman 10"),
        ) +
  guides(fill = guide_colorbar(
    title = name,
    title.position = "top",
    label.position = "top"),
         )
}


(fig <- plot_grid(plot_col(prepare_df(covid), "\nCOVID-19") + theme(strip.text = element_blank(), axis.ticks.y = element_line(), axis.text.y = element_text(size=11)) , 
          plot_col(prepare_df(coh_rot_algae), "Rotifers-Algae\n(coherent)") + theme(strip.text = element_blank()), 
          plot_col(prepare_df(incoh_rot_algae), "Rotifers-Algae\n(incoherent)") + theme(strip.text = element_blank()), 
          plot_col(prepare_df(lynx_hares), "\nLynx and Hares") + theme(strip.text = element_text(size=11)), 
          nrow=1,
          rel_widths = c(1.17, 1, 1, 1.187),
          labels = c("Relative\n  MAE", "", "", ""),
          label_fontfamily = "serif",
          label_fontface = "plain",
          label_size = 11,
          hjust = 0.52, vjust = 3) +
    theme(plot.margin = margin(0,0,3,2))
  )


y.grob <- textGrob("IC sampling interval", 
                   gp=gpar(fontsize=15, fontfamily="LM Roman 10"), rot=90, hjust = 0.7, vjust=0.3)

x.grob <- textGrob("KP sampling interval", 
                   gp=gpar(fontsize=15, fontfamily="LM Roman 10"), vjust=0.5)

y2.grob <- textGrob("Dataset size", 
                   gp=gpar(fontsize=15, fontfamily="LM Roman 10"), rot=270, hjust = 0.25)


fig_complete <- grid.arrange(arrangeGrob(fig, left = y.grob, bottom = x.grob, right=y2.grob))

ggsave("multivariate-evaluation_v5_reverse.pdf", fig_complete, width=6.5, height=7, device=cairo_pdf)





