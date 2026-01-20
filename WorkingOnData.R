

# ---------------------------------
# 
# ---------------------------------

library(ggplot2)
library(reshape2)
library(tidyverse)
library(plotly)

load("~/MSC - SEM 3 THINGS/STAT 304 - Applied Bayesian Methods & Semiparametric Methods/Project/AppliedBayesianProject/pollution.Rdata")
names(pollution)
pollution
pollution = pollution[, c(1:3,15,4:14,16)]
names(pollution)
pollution_X = pollution[,-16]
View(pollution_X)
names(pollution_X)


cor_mat = cor(pollution_X, use = "complete.obs")
cor_long = reshape2::melt(cor_mat)



# ------------------------------------------
# Keep only upper triangle (including diagonal)
# ------------------------------------------
cor_long$Var1 <- factor(cor_long$Var1, levels = colnames(cor_mat))
cor_long$Var2 <- factor(cor_long$Var2, levels = colnames(cor_mat))

cor_upper <- cor_long[as.numeric(cor_long$Var1) <=
                        as.numeric(cor_long$Var2), ]

# ------------------------------------------
# Plot
# ------------------------------------------
p = ggplot(cor_upper, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white", linewidth = 0.6) +
  
  # 🔴 Highlight strong correlations (non-diagonal only)
  geom_tile(
    data = subset(
      cor_upper,
      abs(value) > 0.6 & Var1 != Var2
    ),
    color = "black",
    linewidth = 1,
    fill = NA
  ) +
  
  # Add correlation values
  geom_text(aes(label = sprintf("%.2f", value)),
            size = 2.5, color = "black", fontface = "bold") +
  
  # Professional diverging palette
  # scale_fill_viridis(
  #   option = "D",
  #   limits = c(-1, 1),
  #   name = "Correlation"
  # )+
  #scale_fill_gradientn(
  #   colours = rev(RColorBrewer::brewer.pal(11, "RdGy")),
  #   limits = c(-1, 1),
  #   name = "Correlation"
  # )+
  scale_fill_gradientn(
    colours = scales::alpha(viridisLite::viridis(256), 0.90),
    limits = c(-1, 1),
    name = "Correlation"
  )+
  # viridis::scale_fill_viridis(
  #   option = "C",
  #   direction = -1,
  #   limits = c(-1, 1),
  #   name = "Correlation"
  # )+
  
  coord_fixed() +
  
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 12),
    #panel.grid = element_blank(),
    legend.position = "right"
  ) +
  
  labs(
    #title = "Upper-Triangle Correlation Matrix",
    #subtitle = "Pearson correlation coefficients",
    x = "",
    y = ""
  )

p
ggplotly(p)




# Pollution data - EDA (Automated)

names(pollution)
str(pollution)
glimpse(pollution)



library(DataExplorer)
create_report(pollution)



# ===================================================================
# 
#  BAYESIAN RIDGE - LASSO- EN
# 
# ===================================================================

source("Bayesian_Ridge.R")  # done
saveRDS(results, file = "Ridge_Results_Pollu_Data.rds") # done
readRDS("ElasticNet_Results_Pollu_Data.rds")


source("Bayesian_Lasso.R")  # done
saveRDS(results, file = "Lasso_Results_Pollu_Data.rds") # done


source("Bayesian_ElasticNet.R")  # done
saveRDS(results, file = "ElasticNet_Results_Pollu_Data.rds") # done





# source("Bayesian_Lasso.R")  # done
# saveRDS(results, file = "Lasso_Results_Pollu_Data.rds") # done

# source("Bayesian_ElasticNet.R")
# saveRDS(results, file = "Enet_Results_Pollu_Data.rds")




# ===================================================================
# 
#  OBSERVING MULTICOLLINEARITY
# 
# ===================================================================

library(MASS)

X = scale(pollution[,-16])
y = scale(pollution[,16])

lm_model = lm(y ~ ., data = as.data.frame(X))
summary(lm_model)



library(car)
vif(lm_model)


















