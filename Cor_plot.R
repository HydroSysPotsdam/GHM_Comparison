
#ISIMIP data vis
#Lina Stein

library(readr)
library(ggplot2)
library(ggrepel)
library(reshape2)
library(tidyverse)

#Load model correlation data
correlations_new_domains <- read_csv("C:/Users/stein/Nextcloud/Projects/ISIMIP_modelcomparison/plotting_data/correlations_new_domains_new.csv")

#Tidy and reshape model correlations
cor_model = melt(correlations_new_domains, id = c("Test", "output", "GHM", "Domain", "model_netrad"))%>%
  add_column(Number = NA)%>%
  add_column(Source = NA)%>%
  select(-Test)%>%
  select(-model_netrad)%>%
  rename(Variable = output, Forcing = variable, Input = GHM)%>%
  mutate(category = ifelse(Input == "Mean", "ensemble", "model"))


#Load data based correlations
data_based_correlations <- read_csv("C:/Users/stein/Nextcloud/Projects/ISIMIP_modelcomparison/plotting_data/data_based_correlations230705.csv", na = "-")


#Tidy and reshape data correlations to match cor_model tibble
cor_data = data_based_correlations %>%
  filter(Source!= "FLUXNET")%>%
  mutate(Number = as.integer(Number))%>%
  melt(id.vars = c("Variable", "Forcing", "Source", "Number"))%>%
  mutate(Input = variable)%>%
  rename(Domain = variable)%>%
  relocate(Variable, Input, Domain, Forcing, value, Number, Source)%>%
  mutate(category = "data")%>%
  filter(!is.na(Number))

cor_space = cor_data %>%
  filter(Input == "wet warm")%>%
  mutate(value = NA)%>%
  mutate(Input = "space")

#Combine model and data correlations
cor_tibble = rbind(cor_model, cor_data, cor_space)%>%
  mutate(Input = factor(Input, levels = rev(c("wet warm", "wet cold", "dry cold", "dry warm","space", "clm45", "cwatm", "h08", "jules-w1", "lpjml", "matsiro", "pcr-globwb", "watergap2", "Mean"))))


#Graphing details
shape_vec = c("model" = 19, "ensemble" = 15, "data" = 6)
palette = c("wet warm" = '#018571', "dry warm" = '#a6611a', "wet cold"= '#80cdc1', "dry cold"= '#dfc27d')
Variable.labs <- c("Actual evapotranspiration", "Groundwater recharge", "Total runoff")
names(Variable.labs) <- c("evap", "qr", "qtot")
Forcing.labs <- c("Precipitation", "Net radiation")
names(Forcing.labs) <- c("pr", "netrad")

annotate_df = data.frame(Input = rep("wet warm", 6), value = rep(-0.1, 6), Variable = factor(c("evap", "evap", "qr", "qr", "qtot", "qtot")), Forcing = factor(c("pr", "netrad","pr", "netrad", "pr",  "netrad")), label = paste0("(", letters[1:6], ")"), Domain = rep(NA, 6), category = rep(NA, 6), number = rep(NA, 6), source = rep(NA, 6))
options(ggrepel.max.overlaps = Inf)


#GGplot correlations
ggplot(cor_tibble, aes(Input, value, col = Domain, shape = category, size = (category == "data")))+
  geom_hline(yintercept = 0, color = "grey60")+
  annotate("rect", xmin = 9.5, xmax = 15, ymin = -Inf, ymax = Inf,
           alpha = .2,fill = "grey")+
  #annotate("rect", xmin = 0, xmax = 9.5, ymin = -Inf, ymax = Inf,
  #         alpha = .2,fill = "grey")+
  geom_point()+geom_text_repel(aes(label=Number), size = 4, point.size = NA, color = "black")+
  geom_line(data = cor_tibble%>%filter(category == "model"), aes(col = Domain, group = Domain), size = 0.5, linetype = "dashed")+
  scale_shape_manual(values = shape_vec)+
  scale_size_manual(values = c(2, 6))+ #2,5
  scale_colour_manual(values = palette)+
  coord_flip()+
  facet_grid(Variable~Forcing, labeller = labeller(Variable = Variable.labs, Forcing = Forcing.labs))+
  theme_bw()+
  theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(linewidth = 0.2, color = "grey60"), panel.grid.minor.x = element_line(linewidth  = 0.2, color = "grey60"), legend.position = "none", strip.background = element_rect(fill = "white"), axis.ticks.y = element_blank(), text = element_text(size = 16)) + 
  scale_x_discrete(
  "",
  labels = c(
    "wet warm" = "",
    "wet cold" = "Benchmark datasets",
    "dry warm" = "",
    "dry cold" = "and theory",
    "space" = "",
    "clm45" = "CLM4.5",
    "jules-w1" = "JULES-W1",
    "lpjml" = "LPJmL",
    "matsiro" = "MATSIRO",
    "pcr-globwb" = "PCR-GLOBWB",
    "watergap2" = "WaterGAP2",
    "h08" = "H08",
    "cwatm"= "CWatM",  
    "Mean" = "Ensemble Mean"
  )
)+
  ylab(expression("Rank correlation" ~ rho[s]))+
  geom_text(
    size    = 5,
    data    = annotate_df,
    mapping = aes(x = Inf, y = -Inf, label = label),
    hjust   = -0.2,
    vjust   = 1.3
  )




#Save Figure
ggsave("C:/Users/stein/Nextcloud/Projects/ISIMIP_modelcomparison/plotting_data/Figure3.pdf", dpi = "retina", width = 10, height = 10)
ggsave("C:/Users/stein/Nextcloud/Projects/ISIMIP_modelcomparison/plotting_data/Figure3.png", dpi = "retina", width = 10, height = 10)








