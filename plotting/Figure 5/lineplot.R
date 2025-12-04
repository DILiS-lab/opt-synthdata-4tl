if (!require("tidyverse")){
  install.packages("tidyverse")
}
if (!require("extrafont")){
  install.packages("extrafont")
}

library(tidyverse)
library(ggplot2)
library(extrafont) 
loadfonts()

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Returns the interval of the IQR as a string based on the standard deviation of the lognormal
get_interval_string <- function(sd){
  lower <- upper <- round(qlnorm(0.25, meanlog = 0, sdlog = as.numeric(sd)), 2) 
  upper <- round(qlnorm(0.75, meanlog = 0, sdlog = as.numeric(sd)), 2) 
  paste0("[", sprintf("%.2f", lower), ",", sprintf("%.2f", upper), "]")
}


# loads the normlized ODE baseline and renames experiments 
# ode_baseline <- read_csv("../ode_baseline.csv") |>
#   select(rotifers_algae_coherent, lynx_hares) |>
#   pivot_longer(everything(), names_to = "experiment", values_to = "mean_mae_norm") |>
#   mutate(experiment=if_else(experiment == "covid", "COVID-19", 
#           if_else(experiment == "rotifers_algae_coherent", "Rotifers-Algae (coherent)", 
#                   if_else(experiment == "rotifers_algae_incoherent", "Rotifers-Algae (incoherent)",
#                           "Lynx and Hares"))),
#         experiment = factor(experiment, levels=c("COVID-19", "Rotifers-Algae (coherent)", "Rotifers-Algae (incoherent)", "Lynx and Hares")),
#         noise_type="ODE baseline",
#         noise_level = "0.0")


# load results based on the adjusted output csv of the noise experiments
results_tl <- read_csv("real_world_noise_experiments_adjusted.csv") |>
  select(MAE, noise_std, experiment, noise_type) |>
  group_by(experiment, noise_type, noise_std) |>
  summarise(mean_mae = mean(MAE)) |>
  # remove additive noise experiments as we do not want to show these in the figure
  filter(!grepl("additive", noise_type, fixed = TRUE)) |>
  filter(noise_type!="tl") 


# removes baselines and renames experiments and noise types and normalizes values
results_tl_clean <- results_tl |>
  filter(noise_std != "baseline") |>
  left_join(results_tl |>
              filter(
                noise_std == "baseline"), 
            by=c("experiment"="experiment", "noise_type"="noise_type")) |>
  select(experiment, noise_type, noise_level=noise_std.x, mean_mae=mean_mae.x, dl_baseline=mean_mae.y) |>
  mutate(mean_mae_norm = mean_mae/dl_baseline) |>
  mutate(noise_type = if_else(noise_level=="baseline", "baseline", noise_type)) |>
  mutate(noise_type=if_else(noise_type=="tl_multiplicative", "Measurement", 
                            if_else(noise_type=="tl_multiplicative_derivative", "Environmental", noise_type))) |>
  mutate(experiment=if_else(experiment == "sir", "COVID-19", 
                            if_else(experiment == "rosenbaum_coherent_new", "Rotifers-Algae (coherent)", 
                                    if_else(experiment == "rosenbaum_incoherent", "Rotifers-Algae (incoherent)",
                                            "Lynx and Hares"))),
         experiment = factor(experiment, levels=c("COVID-19", "Rotifers-Algae (coherent)", "Rotifers-Algae (incoherent)", "Lynx and Hares"))) |>
  select(-dl_baseline)


# loads the DL baseline and renames experiments and normalizes values
dl_baseline <- results_tl |>
  filter(noise_std == "baseline") |>
  mutate(experiment=if_else(experiment == "sir", "COVID-19", 
                            if_else(experiment == "rosenbaum_coherent_new", "Rotifers-Algae (coherent)", 
                                    if_else(experiment == "rosenbaum_incoherent", "Rotifers-Algae (incoherent)",
                                            "Lynx and Hares"))),
         experiment = factor(experiment, levels=c("COVID-19", "Rotifers-Algae (coherent)", "Rotifers-Algae (incoherent)", "Lynx and Hares"))) |>
  mutate(noise_type = "DL baseline",
         noise_level="0.0",
         mean_mae_norm = mean_mae/mean_mae) |>
  select(-c("noise_std", "mean_mae"))

# Repeats the ODE and DL baselines for each standard deviation to simplify plotting
for (i in c("0.0625", "0.125", "0.25", "0.5", "1.0")) {
  # ode_baseline <- rbind(ode_baseline, ode_baseline |>
  #                         mutate(noise_level=i))
  dl_baseline <- rbind(dl_baseline, dl_baseline |>
                         mutate(noise_level=i))
}

# combines baselines and experiment results
df_combined <- rbind(results_tl_clean, dl_baseline) |>
  mutate(noise_type = factor(noise_type, levels=c("Measurement", "Environmental", "DL baseline")))  |>
  mutate(noise_level = get_interval_string(noise_level),)

# plots figure
(fig <- ggplot(data=df_combined, aes(x=reorder(noise_level, desc(noise_level)), y=mean_mae_norm, group=noise_type, color=noise_type)) +
  geom_line(linewidth=1) +
  geom_point() +
  facet_wrap(~experiment, 
             ncol = 2,
             scales= "free_y") +
  scale_color_manual(values=c('#e83f3f', '#f69c73', '#3E4989', '#51A537')) +
  xlab("Noise factor (IQR)") +
  ylab("Relative MAE") + 
  theme_bw(base_size = 22) +
  theme(legend.position = "top",
        axis.title = element_text(size=18),
        axis.text = element_text(size=15),
        axis.text.x = element_text(angle = 20, vjust=0.75),
        legend.title = element_blank(),
        text = element_text(family="LM Roman 10"),
        strip.background = element_blank(),
        strip.text = element_text(size=21),
        legend.key.width=unit(1.3,"lines"),
        legend.box.margin=margin(-10,-10,-25,-10),
        plot.margin =  margin(t = 0, r = 8, b = 0, l = 2, unit = "pt")))

   ggsave("RealWorldBoxplots_all_mae_2x2_v5.pdf", fig, width=10, height=6, device=cairo_pdf)
   