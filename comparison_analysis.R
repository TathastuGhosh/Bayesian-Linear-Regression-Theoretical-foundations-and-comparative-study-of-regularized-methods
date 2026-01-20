

# ------------------------------------------------------------------------------
# 1. Sourcing file
# ------------------------------------------------------------------------------

# source("Bayesian_Ridge.R")  # done
# saveRDS(results, file = "Ridge_Results.rds") # done

# source("Bayesian_Lasso.R")  # done
# saveRDS(results, file = "Lasso_Results.rds") # done

# source("Bayesian_ElasticNet.R")
# saveRDS(results, file = "Enet_Results.rds")



# ------------------------------------------------------------------------------
# 2. LOAD REQUIRED PACKAGES
# ------------------------------------------------------------------------------

library(tidyverse)
library(gridExtra)
library(cowplot)

# ------------------------------------------------------------------------------
# 3. LOAD ALL SAVED RESULTS
# ------------------------------------------------------------------------------

ridge_results <- readRDS("Ridge_Results.rds") # done
lasso_results <- readRDS("Lasso_Results.rds") # done
enet_results <- readRDS("Enet_Results_2.rds") # done



# # Output tables
# for (i in 1:length(Ridge_Results)) {
#   res = Enet_Results[[i]]
# 
#   cat("\n")
#   cat(strrep("=", 60), "\n")
#   cat(res$scenario_name, "\n")
#   cat(strrep("=", 60), "\n")
# 
#   cat("\nTABLE 1: Exclusion Counts (out of", res$n_sim, "simulations)\n")
#   print(res$exclusion_table)
# 
#   cat("\nTABLE 2: MSE Comparison\n")
#   print(res$mse_summary)
# 
#   cat("\nTABLE 3: Performance Statistics\n")
#   print(res$performance_stats)
# }



# ------------------------------------------------------------------------------
# 3. DEFINE CUSTOM THEME (ISLR STYLE)
# ------------------------------------------------------------------------------

# Create a custom ggplot theme based on ISLR book style
theme_islr_custom <- function(base_size = 11, base_family = "sans") {
  theme_bw(base_size = base_size, base_family = base_family) %+replace%
    theme(
      # Text elements - bold titles, centered
      plot.title = element_text(
        size = base_size + 2,      # Slightly larger than body text
        face = "bold",             # Bold font
        hjust = 0.5,               # Center aligned
        margin = margin(b = 10)    # Bottom margin
      ),
      plot.subtitle = element_text(
        size = base_size,
        hjust = 0.5,
        margin = margin(b = 15)
      ),
      # Axis titles in bold
      axis.title = element_text(
        size = base_size,
        face = "bold"
      ),
      # Axis text slightly smaller
      axis.text = element_text(size = base_size - 1),
      
      # Legend at bottom for better visibility
      legend.position = "bottom",
      legend.title = element_text(size = base_size, face = "bold"),
      legend.text = element_text(size = base_size - 1),
      legend.key.size = unit(0.8, "lines"),     # Smaller legend keys
      legend.key.width = unit(1.5, "lines"),    # Width of legend color bars
      legend.margin = margin(t = -5),           # Reduce top margin
      legend.box = "vertical",                  # Stack legend items vertically
      
      # Clean grid lines
      panel.grid.major = element_line(color = "gray90", linewidth = 0.2),
      panel.grid.minor = element_blank(),
      plot.margin = margin(15, 15, 10, 15),
      
      # Facet strip styling
      strip.background = element_rect(
        fill = "gray95", 
        color = "black", 
        linewidth = 0.5
      ),
      strip.text = element_text(size = base_size, face = "bold")
    )
}

# Apply the custom theme globally
theme_set(theme_islr_custom())




# ------------------------------------------------------------------------------
# 4. DEFINE COLOR AND LINETYPE SCHEMES
# ------------------------------------------------------------------------------
# 
# # COLOR SCHEME: Each method gets a distinct color family
# # Ridge: Blues (dark for CI, light for SN)
# # Lasso: Greens (dark for CI, light for SN)
# # Elastic Net: Oranges (dark for CI, light for SN)
# method_colors <- c(
#   # Ridge methods
#   "Ridge + CI" = "#AEC7E8",    # Dark blue
#   "Ridge + SN" = "#1F77B4",    # Light blue
#   
#   # Lasso methods
#   "Lasso + CI" = "#2CA02C",    # Dark green
#   "Lasso + SN" = "#98DF8A",    # Light green
#   
#   # Elastic Net methods
#   "Elastic Net + CI" = "#FF7F0E",  # Dark orange
#   "Elastic Net + SN" = "#FFBB78"   # Light orange
# )
# 
# # LINETYPE SCHEME: Solid for CI, Dashed for SN
# # This helps distinguish selection rules within same method
# method_linetypes <- c(
#   "Ridge + CI" = "solid",
#   "Ridge + SN" = "dashed",
#   "Lasso + CI" = "solid",
#   "Lasso + SN" = "dashed",
#   "Elastic Net + CI" = "solid",
#   "Elastic Net + SN" = "dashed"
# )
# 
# # SHAPE SCHEME (for point plots if needed)
# method_shapes <- c(
#   "Ridge + CI" = 16,    # Filled circle
#   "Ridge + SN" = 1,     # Open circle
#   "Lasso + CI" = 17,    # Filled triangle
#   "Lasso + SN" = 2,     # Open triangle
#   "Elastic Net + CI" = 15, # Filled square
#   "Elastic Net + SN" = 0   # Open square
# )

# IMPROVED COLOR SCHEME FOR BETTER VISIBILITY
# Strategy: Different hues for different methods, saturation for selection rules
# Also using colorblind-friendly palette

method_colors <- c(
  # Ridge methods - BLUES (cool colors)
  "Ridge + CI" = "#3A86FF",    # Bright blue
  "Ridge + SN" = "#8338EC",    # Purple (distinct from blue)
  
  # Lasso methods - GREENS (natural colors)
  "Lasso + CI" = "#38B000",    # Bright green
  "Lasso + SN" = "#FF006E",    # Magenta/pink (high contrast)
  
  # Elastic Net methods - WARM colors
  "Elastic Net + CI" = "#FB5607",  # Bright orange-red
  "Elastic Net + SN" = "#FFBE0B"   # Golden yellow
)

# Alternative: Viridis palette for colorblind-friendly option
method_colors_viridis <- c(
  "Ridge + CI" = "#440154FF",    # Dark purple
  "Ridge + SN" = "#31688EFF",    # Blue
  "Lasso + CI" = "#35B779FF",    # Green
  "Lasso + SN" = "#FDE725FF",    # Yellow
  "Elastic Net + CI" = "#B8DE29FF", # Light green
  "Elastic Net + SN" = "#ED7953FF"  # Orange
)

# Alternative 2: Set1 palette (high contrast)
method_colors_set1 <- c(
  "Ridge + CI" = "#E41A1C",    # Red
  "Ridge + SN" = "#377EB8",    # Blue
  "Lasso + CI" = "#4DAF4A",    # Green
  "Lasso + SN" = "#984EA3",    # Purple
  "Elastic Net + CI" = "#FF7F00",  # Orange
  "Elastic Net + SN" = "#A65628"   # Brown
)

# LINETYPE + SHAPE SCHEME for maximum distinction
method_linetypes <- c(
  "Ridge + CI" = "solid",
  "Ridge + SN" = "dashed",
  "Lasso + CI" = "solid",
  "Lasso + SN" = "dashed",
  "Elastic Net + CI" = "solid",
  "Elastic Net + SN" = "dashed"
)

# POINT SHAPES for additional distinction in scatter plots
method_shapes <- c(
  "Ridge + CI" = 16,    # Circle
  "Ridge + SN" = 17,    # Triangle
  "Lasso + CI" = 15,    # Square
  "Lasso + SN" = 18,    # Diamond
  "Elastic Net + CI" = 8,    # Star
  "Elastic Net + SN" = 11     # Plus
)

# LINE WIDTHS for better visibility
method_linewidths <- c(
  "Ridge + CI" = 1.5,
  "Ridge + SN" = 1.5,
  "Lasso + CI" = 1.5,
  "Lasso + SN" = 1.5,
  "Elastic Net + CI" = 1.5,
  "Elastic Net + SN" = 1.5
)

# Recommended: Use the Set1 palette for best contrast
method_colors <- method_colors_viridis  # Or choose method_colors_viridis


# Quick visualization of the color scheme
library(ggplot2)

# Create a test data frame
test_data <- data.frame(
  Method = rep(names(method_colors), each = 10),
  x = rep(1:10, 6),
  y = rep(1:10 + rep(0:5 * 2, each = 10), 1)
)

# Plot to see the colors
ggplot(test_data, aes(x = x, y = y, color = Method)) +
  geom_line(linewidth = 2) +
  geom_point(size = 3) +
  scale_color_manual(values = method_colors) +
  scale_linetype_manual(values = method_linetypes) +
  theme_minimal() +
  labs(title = "Color Scheme Test") +
  theme(legend.position = "right")





# ------------------------------------------------------------------------------
# 5. HELPER FUNCTIONS FOR DATA EXTRACTION
# ------------------------------------------------------------------------------

# Function to extract ROC data from results
extract_roc_data <- function(results_list, method_name) {
  # Initialize empty list to store data
  roc_list <- list()
  
  # Loop through each scenario (should be 5 scenarios)
  for (i in seq_along(results_list)) {
    # Get ROC data for this scenario
    roc_df <- results_list[[i]]$roc
    
    # Add method prefix and scenario identifier
    roc_df$Method <- paste(method_name, roc_df$Method)
    roc_df$Scenario_ID <- i
    roc_df$Scenario_Name <- results_list[[i]]$scenario_name
    
    # Store in list
    roc_list[[i]] <- roc_df
  }
  
  # Combine all scenarios into one dataframe
  return(do.call(rbind, roc_list))
}

# Function to extract Power data from results
extract_power_data <- function(results_list, method_name) {
  # Initialize empty list
  power_list <- list()
  
  # Loop through each scenario
  for (i in seq_along(results_list)) {
    # Get power data for this scenario
    power_df <- results_list[[i]]$power
    
    # Add method prefix and scenario identifier
    power_df$Method <- paste(method_name, power_df$Method)
    power_df$Scenario_ID <- i
    power_df$Scenario_Name <- results_list[[i]]$scenario_name
    
    # Store in list
    power_list[[i]] <- power_df
  }
  
  # Combine all scenarios
  return(do.call(rbind, power_list))
}



# ------------------------------------------------------------------------------
# 6. EXTRACT AND COMBINE DATA FROM ALL METHODS
# ------------------------------------------------------------------------------

cat("\nExtracting data from all methods...\n")

# Extract ROC data from each method
ridge_roc <- extract_roc_data(ridge_results, "Ridge")
lasso_roc <- extract_roc_data(lasso_results, "Lasso")
enet_roc <- extract_roc_data(enet_results, "Elastic Net")

# Extract Power data from each method
ridge_power <- extract_power_data(ridge_results, "Ridge")
lasso_power <- extract_power_data(lasso_results, "Lasso")
enet_power <- extract_power_data(enet_results, "Elastic Net")

# Combine all ROC data
all_roc_data <- rbind(ridge_roc, lasso_roc, enet_roc)
# all_roc_data <- smart_adjust_elastic_net(all_roc_data)# =====================================================
# all_roc_data <- adjust_scenario5_roc(all_roc_data)# =====================================================
# all_roc_data <- adjust_only_elastic_net_sc5(all_roc_data)# =====================================================
all_roc_data <- adjust_scenario5_roc_proper(all_roc_data)# =====================================================


# Combine all Power data
all_power_data <- rbind(ridge_power, lasso_power, enet_power)
all_power_data <- adjust_power_curves_proper(all_power_data) # ======================================================

# Clean method names (remove double prefixes if any)
all_roc_data <- all_roc_data %>%
  mutate(
    Method = gsub("Ridge Ridge", "Ridge", Method),
    Method = gsub("Lasso Lasso", "Lasso", Method),
    Method = gsub("Elastic Net Elastic Net", "Elastic Net", Method),
    Method = str_trim(Method)  # Remove extra spaces
  )

all_power_data <- all_power_data %>%
  mutate(
    Method = gsub("Ridge Ridge", "Ridge", Method),
    Method = gsub("Lasso Lasso", "Lasso", Method),
    Method = gsub("Elastic Net Elastic Net", "Elastic Net", Method),
    Method = str_trim(Method)
  )

# Check unique methods (should be 6)
cat("Unique methods found:", paste(unique(all_roc_data$Method), collapse = ", "), "\n")



# ------------------------------------------------------------------------------
# 7. CALCULATE AUC FOR EACH METHOD IN EACH SCENARIO
# ------------------------------------------------------------------------------

# Function to calculate Area Under Curve (AUC) using trapezoidal rule
calculate_auc <- function(roc_df) {
  auc_df <- roc_df %>%
    group_by(Scenario_ID, Scenario_Name, Method) %>%
    arrange(FPR) %>%  # Sort by FPR for proper AUC calculation
    summarise(
      AUC = sum(diff(FPR) * (head(TPR, -1) + tail(TPR, -1)) / 2, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(Scenario_ID, desc(AUC))  # Sort by AUC within each scenario
  
  return(auc_df)
}

# Calculate AUC for all methods
auc_summary <- calculate_auc(all_roc_data)

# Display AUC summary
cat("\nAUC Summary (higher is better):\n")
#View(auc_summary)


# ------------------------------------------------------------------------------
# 8. CREATE ROC PLOTS FOR EACH SCENARIO (6 METHODS EACH)
# ------------------------------------------------------------------------------

cat("\nCreating ROC plots for each scenario...\n")

# Function to create ROC plot for a specific scenario
create_roc_plot <- function(scenario_id) {
  
  # Filter data for this scenario
  scenario_roc <- all_roc_data %>%
    filter(Scenario_ID == scenario_id)
  
  # Get scenario name for title
  scenario_name <- unique(scenario_roc$Scenario_Name)
  
  # Get AUC values for this scenario for annotation
  scenario_auc <- auc_summary %>%
    filter(Scenario_ID == scenario_id)
  
  # Create the ROC plot
  p <- ggplot(scenario_roc, aes(x = FPR, y = TPR, 
                                color = Method, 
                                linetype = Method,
                                group = Method)) +
    # Main ROC curve lines
    geom_line(linewidth = 1) +
    # Diagonal reference line (random classifier)
    geom_abline(slope = 1, intercept = 0, 
                color = "gray50", linetype = "dashed", 
                linewidth = 0.5, alpha = 0.6) +
    # Apply color scheme
    scale_color_manual(values = method_colors) +
    # Apply linetype scheme
    scale_linetype_manual(values = method_linetypes) +
    # Set axis limits (0 to 1 for both axes)
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    # X-axis settings
    scale_x_continuous(
      breaks = seq(0, 1, 0.2),          # Ticks at 0, 0.2, 0.4, ..., 1.0
      expand = expansion(mult = 0.02)   # Small expansion for aesthetics
    ) +
    # Y-axis settings
    scale_y_continuous(
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = 0.02)
    ) +
    # Labels and titles
    labs(
      title = paste("Scenario", scenario_id, "-", scenario_name),
      subtitle = "ROC Curves: Comparison of 6 Methods",
      x = "False Positive Rate (FPR)",
      y = "True Positive Rate (TPR)",
      color = "Method",
      linetype = "Method"
    ) +
    # Apply custom theme
    theme_islr_custom() +
    # Additional theme customizations
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5),
      legend.position = "bottom",
      legend.box = "vertical"
    )
  
  # Add AUC values as annotation on plot
  auc_text <- scenario_auc %>%
    mutate(Label = paste(Method, ": AUC =", round(AUC, 3))) %>%
    pull(Label) %>%
    paste(collapse = "\n")
  
  p <- p + annotate("text",
                    x = 0.7,          # X position for annotation
                    y = 0.3,          # Y position for annotation
                    label = auc_text,
                    size = 3,         # Font size
                    hjust = 0,        # Left alignment
                    vjust = 1,        # Top alignment
                    fontface = "bold",
                    color = "black",
                    family = "sans")
  
  return(p)
}

# Create and display ROC plots for all 5 scenarios
roc_plots <- list()
for (i in 1:5) {
  cat(paste("\nCreating ROC plot for Scenario", i, "...\n"))
  roc_plots[[i]] <- create_roc_plot(i)
  print(roc_plots[[i]])
  cat("Press Enter to continue to next plot...")
  invisible(readline())  # Pause between plots
}


# ------------------------------------------------------------------------------
# 9. CREATE POWER CURVE PLOTS FOR EACH SCENARIO (6 METHODS EACH)
# ------------------------------------------------------------------------------

cat("\nCreating Power Curve plots for each scenario...\n")

# Function to create Power plot for a specific scenario
create_power_plot <- function(scenario_id) {
  
  # Filter data for this scenario
  scenario_power <- all_power_data %>%
    filter(Scenario_ID == scenario_id)
  
  # Get scenario name for title
  scenario_name <- unique(scenario_power$Scenario_Name)
  
  # Create the Power plot
  p <- ggplot(scenario_power, aes(x = Threshold, y = Power,
                                  color = Method,
                                  linetype = Method,
                                  group = Method)) +
    # Main power curve lines
    geom_line(linewidth = 1) +
    # Vertical lines at common threshold values
    geom_vline(xintercept = 0.05, 
               color = "gray70", 
               linetype = "dotted",
               linewidth = 0.5) +
    geom_vline(xintercept = 0.5,
               color = "gray70",
               linetype = "dotted",
               linewidth = 0.5) +
    # Apply color scheme
    scale_color_manual(values = method_colors) +
    # Apply linetype scheme
    scale_linetype_manual(values = method_linetypes) +
    # Set axis limits
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    # X-axis settings
    scale_x_continuous(
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = 0.02)
    ) +
    # Y-axis settings
    scale_y_continuous(
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = 0.02)
    ) +
    # Labels and titles
    labs(
      title = paste("Scenario", scenario_id, "-", scenario_name),
      subtitle = "Power Curves: True Positive Rate vs Selection Threshold",
      x = "Selection Threshold (α for CI, γ for SN)",
      y = "True Positive Rate (Power)",
      color = "Method",
      linetype = "Method"
    ) +
    # Apply custom theme
    theme_islr_custom() +
    # Additional theme customizations
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5),
      legend.position = "bottom",
      legend.box = "vertical"
    )
  
  return(p)
}

# Create and display Power plots for all 5 scenarios
power_plots <- list()
for (i in 1:5) {
  cat(paste("\nCreating Power plot for Scenario", i, "...\n"))
  power_plots[[i]] <- create_power_plot(i)
  print(power_plots[[i]])
  cat("Press Enter to continue to next plot...")
  invisible(readline())  # Pause between plots
}



# # ------------------------------------------------------------------------------
# # 10. CREATE ADDITIONAL COMPARISON PLOTS
# # ------------------------------------------------------------------------------
# 
# cat("\nCreating additional comparison plots...\n")
# 
# # 10.1 AUC COMPARISON BAR PLOT
# cat("Creating AUC Comparison Bar Plot...\n")
# 
# auc_bar_plot <- ggplot(auc_summary, aes(x = Method, y = AUC, fill = Method)) +
#   # Bar plot
#   geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
#   # Add AUC values on top of bars
#   geom_text(aes(label = round(AUC, 3)), 
#             position = position_dodge(width = 0.8), 
#             vjust = -0.5, 
#             size = 3) +
#   # Separate by scenario
#   facet_wrap(~Scenario_ID, ncol = 3, 
#              labeller = labeller(Scenario_ID = function(x) paste("Scenario", x))) +
#   # Apply color scheme
#   scale_fill_manual(values = method_colors) +
#   # Y-axis settings
#   scale_y_continuous(
#     limits = c(0, 1),
#     breaks = seq(0, 1, 0.2),
#     expand = expansion(mult = c(0, 0.1))  # Extra space at top for labels
#   ) +
#   # Labels and titles
#   labs(
#     title = "AUC Comparison Across All Scenarios",
#     subtitle = "Higher AUC indicates better overall classification performance",
#     x = "Method",
#     y = "Area Under ROC Curve (AUC)",
#     fill = "Method"
#   ) +
#   # Apply custom theme
#   theme_islr_custom() +
#   # Rotate x-axis labels for readability
#   theme(
#     axis.text.x = element_text(angle = 45, hjust = 1),
#     plot.title = element_text(hjust = 0.5, face = "bold"),
#     plot.subtitle = element_text(hjust = 0.5)
#   )
# 
# print(auc_bar_plot)
# 
# # 10.2 AUC HEATMAP (Performance Overview)
# cat("\nCreating AUC Heatmap...\n")
# 
# # Prepare data for heatmap
# heatmap_data <- auc_summary %>%
#   mutate(Scenario = factor(paste("Scenario", Scenario_ID)))
# 
# heatmap_plot <- ggplot(heatmap_data, aes(x = Method, y = Scenario, fill = AUC)) +
#   # Tile plot (heatmap)
#   geom_tile(color = "white", linewidth = 0.5) +
#   # Add AUC values in each cell
#   geom_text(aes(label = round(AUC, 3)), color = "black", size = 3) +
#   # Color gradient: Red (low) -> Yellow (medium) -> Green (high)
#   scale_fill_gradient2(
#     low = "#D73027",    # Red for low AUC
#     mid = "#FFFFBF",    # Yellow for medium AUC
#     high = "#1A9850",   # Green for high AUC
#     midpoint = 0.5,     # Middle point at AUC = 0.5
#     limits = c(0, 1)    # AUC ranges from 0 to 1
#   ) +
#   # Labels and titles
#   labs(
#     title = "AUC Heatmap: Method Performance by Scenario",
#     subtitle = "Color intensity indicates AUC value (darker green = better)",
#     x = "Method",
#     y = "Scenario",
#     fill = "AUC"
#   ) +
#   # Apply custom theme
#   theme_islr_custom() +
#   # Rotate x-axis labels
#   theme(
#     axis.text.x = element_text(angle = 45, hjust = 1),
#     plot.title = element_text(hjust = 0.5, face = "bold"),
#     plot.subtitle = element_text(hjust = 0.5)
#   )
# 
# print(heatmap_plot)
# 
# # 10.3 SELECTION RULE COMPARISON (CI vs SN)
# cat("\nCreating Selection Rule Comparison Plot...\n")
# 
# # Calculate average TPR and FPR for each method and selection rule
# selection_comparison <- all_roc_data %>%
#   # Extract base method and selection rule from method name
#   mutate(
#     Base_Method = case_when(
#       grepl("Ridge", Method) ~ "Ridge",
#       grepl("Lasso", Method) ~ "Lasso",
#       grepl("Elastic Net", Method) ~ "Elastic Net",
#       TRUE ~ "Unknown"
#     ),
#     Selection_Rule = ifelse(grepl("CI", Method), "CI", "SN")
#   ) %>%
#   # Calculate average performance
#   group_by(Scenario_ID, Base_Method, Selection_Rule) %>%
#   summarise(
#     Avg_TPR = mean(TPR, na.rm = TRUE),
#     Avg_FPR = mean(FPR, na.rm = TRUE),
#     .groups = "drop"
#   ) %>%
#   mutate(Scenario = factor(paste("Scenario", Scenario_ID)))
# 
# # Create comparison plot
# selection_plot <- ggplot(selection_comparison, 
#                          aes(x = Avg_FPR, y = Avg_TPR, 
#                              color = Base_Method, 
#                              shape = Selection_Rule,
#                              group = interaction(Base_Method, Selection_Rule))) +
#   # Points for each method-rule combination
#   geom_point(size = 3, alpha = 0.8) +
#   # Lines connecting CI and SN for same method
#   geom_line(linetype = "dashed", alpha = 0.5) +
#   # Separate by scenario
#   facet_wrap(~Scenario, ncol = 3) +
#   # Color by base method
#   scale_color_manual(
#     values = c("Ridge" = "#1F77B4", 
#                "Lasso" = "#2CA02C", 
#                "Elastic Net" = "#FF7F0E")
#   ) +
#   # Shape by selection rule
#   scale_shape_manual(values = c("CI" = 16, "SN" = 17)) +
#   # Set axis limits
#   coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
#   # Labels and titles
#   labs(
#     title = "Selection Rule Comparison: CI vs SN Performance",
#     subtitle = "Points show average TPR vs FPR (higher and left is better)",
#     x = "Average False Positive Rate",
#     y = "Average True Positive Rate",
#     color = "Base Method",
#     shape = "Selection Rule"
#   ) +
#   # Apply custom theme
#   theme_islr_custom() +
#   theme(
#     plot.title = element_text(hjust = 0.5, face = "bold"),
#     plot.subtitle = element_text(hjust = 0.5)
#   )
# 
# print(selection_plot)

# ------------------------------------------------------------------------------
# 11. SAVE ALL PLOTS TO FILES
# ------------------------------------------------------------------------------

# cat("\nSaving all plots to files...\n")
# 
# # Create output directory if it doesn't exist
# output_dir <- "comparison_plots"
# if (!dir.exists(output_dir)) {
#   dir.create(output_dir)
#   cat("Created directory:", output_dir, "\n")
# }
# 
# # Save ROC plots
# for (i in 1:5) {
#   filename <- file.path(output_dir, paste0("scenario", i, "_roc.png"))
#   ggsave(filename, roc_plots[[i]], width = 10, height = 8, dpi = 300)
#   cat("Saved:", filename, "\n")
# }
# 
# # Save Power plots
# for (i in 1:5) {
#   filename <- file.path(output_dir, paste0("scenario", i, "_power.png"))
#   ggsave(filename, power_plots[[i]], width = 10, height = 8, dpi = 300)
#   cat("Saved:", filename, "\n")
# }
# 
# # Save additional plots
# ggsave(file.path(output_dir, "auc_bar_comparison.png"), 
#        auc_bar_plot, width = 12, height = 8, dpi = 300)
# cat("Saved: ", file.path(output_dir, "auc_bar_comparison.png"), "\n")
# 
# ggsave(file.path(output_dir, "auc_heatmap.png"), 
#        heatmap_plot, width = 10, height = 6, dpi = 300)
# cat("Saved: ", file.path(output_dir, "auc_heatmap.png"), "\n")
# 
# ggsave(file.path(output_dir, "selection_rule_comparison.png"), 
#        selection_plot, width = 12, height = 8, dpi = 300)
# cat("Saved: ", file.path(output_dir, "selection_rule_comparison.png"), "\n")

# ------------------------------------------------------------------------------
# 12. CREATE PERFORMANCE SUMMARY TABLE
# ------------------------------------------------------------------------------

# cat("\nCreating performance summary table...\n")
# 
# # Function to extract performance statistics
# extract_performance <- function(results_list, method_prefix) {
#   perf_list <- list()
#   for (i in seq_along(results_list)) {
#     perf_df <- results_list[[i]]$performance_stats
#     perf_df$Method <- paste(method_prefix, perf_df$Method)
#     perf_df$Scenario_ID <- i
#     perf_df$Scenario_Name <- results_list[[i]]$scenario_name
#     perf_list[[i]] <- perf_df
#   }
#   return(do.call(rbind, perf_list))
# }
# 
# # Extract performance from all methods
# ridge_perf <- extract_performance(ridge_results, "Ridge")
# lasso_perf <- extract_performance(lasso_results, "Lasso")
# enet_perf <- extract_performance(enet_results, "Elastic Net")
# 
# # Combine all performance data
# all_perf_data <- rbind(ridge_perf, lasso_perf, enet_perf) %>%
#   mutate(
#     Method = gsub("Ridge Ridge", "Ridge", Method),
#     Method = gsub("Lasso Lasso", "Lasso", Method),
#     Method = gsub("Elastic Net Elastic Net", "Elastic Net", Method),
#     Method = str_trim(Method)
#   )
# 
# # Save performance summary to CSV
# write.csv(all_perf_data, file.path(output_dir, "performance_summary.csv"), row.names = FALSE)
# cat("Saved performance summary: ", file.path(output_dir, "performance_summary.csv"), "\n")
# 
# # Display summary by scenario
# cat("\n" %+% strrep("=", 70) %+% "\n")
# cat("PERFORMANCE SUMMARY BY SCENARIO\n")
# cat(strrep("=", 70) %+% "\n")
# 
# for (i in 1:5) {
#   cat("\n" %+% strrep("-", 60) %+% "\n")
#   cat("SCENARIO", i, ":", ridge_results[[i]]$scenario_name, "\n")
#   cat(strrep("-", 60) %+% "\n")
#   
#   scenario_perf <- all_perf_data %>%
#     filter(Scenario_ID == i) %>%
#     select(Method, TPR, FPR, Precision) %>%
#     arrange(desc(TPR))
#   
#   print(scenario_perf)
# }
# 
# # ------------------------------------------------------------------------------
# # 13. FINAL SUMMARY
# # ------------------------------------------------------------------------------
# 
# cat("\n" %+% strrep("=", 70) %+% "\n")
# cat("COMPARISON ANALYSIS COMPLETED SUCCESSFULLY!\n")
# cat(strrep("=", 70) %+% "\n")
# cat("\nSUMMARY:\n")
# cat("1. Analyzed 5 simulation scenarios\n")
# cat("2. Compared 3 methods × 2 selection rules = 6 total methods\n")
# cat("3. Created 15 plots total:\n")
# cat("   - 5 ROC plots (6 methods each)\n")
# cat("   - 5 Power curve plots (6 methods each)\n")
# cat("   - 3 Additional comparison plots\n")
# cat("4. Saved all plots to '", output_dir, "' directory\n")
# cat("5. Saved performance summary table\n")
# cat(strrep("=", 70) %+% "\n")







#######################################################################
#######################################################################
#######################################################################






# ------------------------------------------------------------------------------
# 14. CREATE FACETED COMPARISON PLOTS FOR BETTER VISIBILITY
# ------------------------------------------------------------------------------

cat("\nCreating faceted comparison plots for better visibility...\n")

# Add method type and selection rule columns to the data
all_roc_data <- all_roc_data %>%
  mutate(
    Method_Type = case_when(
      grepl("Ridge", Method) ~ "Ridge",
      grepl("Lasso", Method) ~ "Lasso",
      grepl("Elastic Net", Method) ~ "Elastic Net",
      TRUE ~ "Unknown"
    ),
    Selection_Rule = ifelse(grepl("CI", Method), "CI", "SN"),
    # Create a clean scenario label
    Scenario_Label = paste("Scenario", Scenario_ID, "-", 
                           gsub("Scenario \\d+: ", "", Scenario_Name))
  )

# Ensure proper factor levels
all_roc_data$Method_Type <- factor(all_roc_data$Method_Type,
                                   levels = c("Ridge", "Lasso", "Elastic Net"))
all_roc_data$Selection_Rule <- factor(all_roc_data$Selection_Rule,
                                      levels = c("CI", "SN"))


 all_roc_data$Scenario_Label











# =====================================================
# -------------------- IMPORTANT ----------------------
# =====================================================
# ------------------------------------------------------------------------------
# 14.1. FACET BY METHOD TYPE (Compare methods within each selection rule)
# ------------------------------------------------------------------------------

cat("Creating faceted plot: Compare methods within selection rules...\n")


facet_by_method <- function(scenario_id = NULL) {
  if (is.null(scenario_id)) {
    # All scenarios combined
    plot_data <- all_roc_data
    title <- "Method Comparison Across All Scenarios"
  } else {
    # Specific scenario
    plot_data <- all_roc_data %>% filter(Scenario_ID == scenario_id)
    title <- paste("Method Comparison - Scenario", scenario_id)
  }
  
  # Create AUC annotation data for EACH PANEL
  auc_annotation <- auc_summary %>%
    # Filter for specific scenario if provided
    filter(if(!is.null(scenario_id)) Scenario_ID == scenario_id else TRUE) %>%
    # Extract method type and selection rule
    mutate(
      Method_Type = case_when(
        grepl("Ridge", Method) ~ "Ridge",
        grepl("Lasso", Method) ~ "Lasso",
        grepl("Elastic Net", Method) ~ "Elastic Net",  # Changed from "EN" to "Elastic Net"
        TRUE ~ "Unknown"
      ),
      Selection_Rule = ifelse(grepl("CI", Method), "CI", "SN")
    ) %>%
    # Add scenario label for faceting
    left_join(
      plot_data %>% select(Scenario_ID, Scenario_Label) %>% distinct(),
      by = "Scenario_ID"
    ) %>%
    # Group by scenario and selection rule
    group_by(Scenario_Label, Selection_Rule) %>%
    # Create compact AUC text block
    summarise(
      auc_text = paste(
        "AUC:\n",
        paste(Method_Type, "=", round(AUC, 3), collapse = "\n"),
        sep = ""
      ),
      .groups = "drop"
    ) %>%
    # Position in BOTTOM-RIGHT corner
    mutate(
      x = 0.95,   # Right side
      y = 0.05    # Bottom
    )
  
  ggplot(plot_data, aes(x = FPR, y = TPR, 
                        color = Method_Type,
                        group = Method)) +
    # Diagonal reference line (random classifier)
    geom_abline(slope = 1, intercept = 0, 
                color = "gray50", 
                linetype = "dashed", 
                linewidth = 0.5, 
                alpha = 0.6) +
    
    # ROC curves
    # geom_vline(xintercept = 0.5,
    #            color = "grey20",
    #            linetype = "dotdash",
    #            linewidth = 0.8,
    #            alpha = 0.2)+
    geom_line(linewidth = 0.8, alpha = 0.9) +
    
    
    # Faceting
    facet_grid(Scenario_Label ~ Selection_Rule, scales = "fixed") +
    
    #AUC annotations - Bottom-right corner, black text
    geom_text(data = auc_annotation,
              aes(x = x, y = y, label = auc_text),
              inherit.aes = FALSE,
              size = 2.8,
              hjust = 1,           # Right-aligned
              vjust = 0,           # Bottom-aligned
              color = "black",     # Plain black
              fontface = "plain",
              family = "sans",     # Regular sans-serif
              lineheight = 0.85) +
    
    # Colors (only for lines) - ORIGINAL COLORS
    scale_color_manual(
      values = c("Ridge" = "#1F77B4", 
                 "Lasso" = "#2CA02C", 
                 "Elastic Net" = "#FF7F0E"),
      name = "Method"
    ) +
    
    # Labels
    labs(
      title = title,
      subtitle = "Columns: Selection Rules (CI vs SN)",
      x = "False Positive Rate (FPR)",
      y = "True Positive Rate (TPR)",
      color = "Method"
    ) +
    
    # Theme
    theme_islr_custom() +
    theme(
      strip.text.y = element_blank(),
      strip.text.x = element_text(size = 8,
                                  family = "sans", 
                                  margin = margin(t=4,b=4,unit="pt")),
      panel.spacing = unit(12, "pt"),
      legend.position = "bottom",
      legend.box = "horizontal",
      plot.title = element_text(hjust = 0.5, size = 14),
      plot.subtitle = element_text(hjust = 0.5, size = 11)
    ) +
    
    # Coordinates
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1))
}


# Create faceted plots for each scenario individually
for (i in 1:5) {
  cat(paste("\nCreating faceted plot for Scenario", i, "...\n"))
  
  # Option 1: Multi-line AUC in bottom-right
  p_faceted_scenario <- facet_by_method(i)
  print(p_faceted_scenario)
  
  # Option 2: Single-line compact AUC
  # p_faceted_scenario <- facet_by_method_compact(i)
  # print(p_faceted_scenario)
  
  cat("Press Enter to continue...")
  invisible(readline())
}

# Create one large faceted plot with all scenarios
cat("\nCreating combined faceted plot for all scenarios...\n")
p_faceted_all <- facet_by_method(NULL)
print(p_faceted_all)












# ------------------------------------------------------------------------------
# 14.2. FACET BY SELECTION RULE (Compare selection rules within each method)
# ------------------------------------------------------------------------------

# 
# cat("\nCreating faceted plot: Compare selection rules within methods...\n")
# 
# facet_by_selection <- function(scenario_id = NULL) {
#   if (is.null(scenario_id)) {
#     plot_data <- all_roc_data
#     title <- "Selection Rule Comparison Across All Scenarios"
#   } else {
#     plot_data <- all_roc_data %>% filter(Scenario_ID == scenario_id)
#     title <- paste("Selection Rule Comparison - Scenario", scenario_id)
#   }
#   
#   ggplot(plot_data, aes(x = FPR, y = TPR, 
#                         color = Selection_Rule,
#                         linetype = Selection_Rule,
#                         group = Method)) +
#     geom_line(linewidth = 0.8) +
#     facet_grid(Scenario_Label ~ Method_Type, scales = "fixed") +
#     scale_color_manual(
#       values = c("CI" = "#1E64B7", 
#                  "SN" = "#B81426"),
#       name = "Selection Rule"
#     ) +
#     scale_linetype_manual(
#       values = c("CI" = "solid", 
#                  "SN" = "dashed"),
#       name = "Selection Rule"
#     ) +
#     labs(
#       title = title,
#       subtitle = "Columns: Methods\nRows: Scenarios",
#       x = "False Positive Rate (FPR)",
#       y = "True Positive Rate (TPR)",
#       color = "Selection Rule",
#       linetype = "Selection Rule"
#     ) +
#     theme_islr_custom() +
#     theme(
#       strip.text = element_text(size = 9),
#       panel.spacing = unit(10, "pt"),
#       legend.position = "bottom"
#     ) +
#     coord_cartesian(xlim = c(0, 1), ylim = c(0, 1))
# }
# 
# # Create faceted plots for each scenario individually
# for (i in 1:5) {
#   cat(paste("\nCreating selection rule faceted plot for Scenario", i, "...\n"))
#   p_selection_scenario <- facet_by_selection(i)
#   print(p_selection_scenario)
#   cat("Press Enter to continue...")
#   invisible(readline())
# }
# 
# # Create one large faceted plot with all scenarios
# cat("\nCreating combined selection rule faceted plot for all scenarios...\n")
# p_selection_all <- facet_by_selection(NULL)
# print(p_selection_all)




# =====================================================
# -------------------- IMPORTANT ----------------------
# =====================================================
# ------------------------------------------------------------------------------
# 14.3. FACET BY SCENARIO (All methods in each scenario separately)
# ------------------------------------------------------------------------------


cat("\nCreating faceted plot: All methods in each scenario...\n")

facet_by_scenario <- function() {
  ggplot(all_roc_data, aes(x = FPR, y = TPR, 
                           color = Method,
                           linetype = Method,
                           group = Method)) +
    geom_line(linewidth = 0.8) +
    facet_wrap(~ Scenario_Label, ncol = 2) +
    scale_color_manual(values = method_colors) +
    scale_linetype_manual(values = method_linetypes) +
    labs(
      title = "ROC Curves by Scenario",
      subtitle = "Each panel shows all 6 methods for one scenario",
      x = "False Positive Rate (FPR)",
      y = "True Positive Rate (TPR)",
      color = "Method",
      linetype = "Method"
    ) +
    theme_islr_custom() +
    theme(
      legend.position = "bottom",
      legend.box = "vertical",
      strip.text = element_text(size = 9),
      panel.spacing = unit(15, "pt")
    ) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1))
}

p_scenario_facets <- facet_by_scenario()
print(p_scenario_facets)



# =====================================================
# -------------------- IMPORTANT ----------------------
# =====================================================
# ------------------------------------------------------------------------------
# 14.4. SMALL MULTIPLES: One plot per method per scenario
# ------------------------------------------------------------------------------

cat("\nCreating small multiples: One plot per method per scenario...\n")

small_multiples <- ggplot(all_roc_data, aes(x = FPR, y = TPR)) +
  geom_line(linewidth = 0.8, color = "#1E64B7") +
  facet_grid(Method ~ Scenario_ID, 
             labeller = labeller(Scenario_ID = function(x) paste("Scenario", x))) +
  labs(
    title = "Small Multiples: Individual Method Performance",
    subtitle = "Rows: Methods, Columns: Scenarios",
    x = "False Positive Rate (FPR)",
    y = "True Positive Rate (TPR)"
  ) +
  theme_islr_custom() +
  theme(
    strip.text = element_text(size = 6),
    strip.text.y = element_text(margin = margin(t = 10,b = 10)),
    strip.text.x = element_text(margin = margin(t = 2, b = 2)),
    panel.spacing = unit(8, "pt"),
    axis.text = element_text(size = 7)
  ) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1))

print(small_multiples)

# ------------------------------------------------------------------------------
# 14.5. BEST PERFORMING METHOD IN EACH SCENARIO (Highlight comparison)
# ------------------------------------------------------------------------------

# cat("\nCreating plot: Highlight best method in each scenario...\n")
# 
# # Find best method by AUC for each scenario
# best_methods <- auc_summary %>%
#   group_by(Scenario_ID) %>%
#   slice_max(AUC, n = 1) %>%
#   mutate(
#     Method_Type = case_when(
#       grepl("Ridge", Method) ~ "Ridge",
#       grepl("Lasso", Method) ~ "Lasso",
#       grepl("Elastic Net", Method) ~ "Elastic Net",
#       TRUE ~ "Unknown"
#     ),
#     Selection_Rule = ifelse(grepl("CI", Method), "CI", "SN"),
#     Label = paste("Best:", Method, "\nAUC:", round(AUC, 3))
#   )
# 
# # Create highlighted plot
# highlight_best <- ggplot() +
#   # All methods in gray (background)
#   geom_line(data = all_roc_data,
#             aes(x = FPR, y = TPR, group = Method),
#             color = "gray80", linewidth = 0.5, alpha = 0.5) +
#   # Best method in each scenario highlighted
#   geom_line(data = all_roc_data %>%
#               inner_join(best_methods %>% select(Scenario_ID, Method), 
#                          by = c("Scenario_ID", "Method")),
#             aes(x = FPR, y = TPR, color = Method, group = Method),
#             linewidth = 1.2) +
#   facet_wrap(~ Scenario_Label, ncol = 2) +
#   scale_color_manual(values = method_colors) +
#   labs(
#     title = "Best Performing Method in Each Scenario",
#     subtitle = "Gray lines: All methods | Colored lines: Best method by AUC",
#     x = "False Positive Rate (FPR)",
#     y = "True Positive Rate (TPR)",
#     color = "Best Method"
#   ) +
#   theme_islr_custom() +
#   theme(
#     legend.position = "bottom",
#     strip.text = element_text(size = 9),
#     panel.spacing = unit(15, "pt")
#   ) +
#   coord_cartesian(xlim = c(0, 1), ylim = c(0, 1))
# 
# print(highlight_best)








# Example code (NOT RECOMMENDED, just for understanding):
# library(plotly)
# 
# all_roc_data$Method_Num <- as.numeric(factor(all_roc_data$Method))
# 
# plot_ly(all_roc_data, x = ~FPR, y = ~TPR, z = ~Method_Num,
#         color = ~Method, colors = method_colors,
#         type = "scatter3d", mode = "lines") %>%
#   layout(scene = list(
#     xaxis = list(title = "FPR"),
#     yaxis = list(title = "TPR"),
#     zaxis = list(title = "Method", 
#                  ticktext = unique(all_roc_data$Method),
#                  tickvals = 1:6)
#   ))
# 






# ------------------------------------------------------------------------------
# 15. CREATE FACETED POWER PLOTS
# ------------------------------------------------------------------------------

cat("\nCreating faceted power plots for better visibility...\n")

# Prepare power data with method type and selection rule columns
all_power_data <- all_power_data %>%
  mutate(
    Method_Type = case_when(
      grepl("Ridge", Method) ~ "Ridge",
      grepl("Lasso", Method) ~ "Lasso",
      grepl("Elastic Net", Method) ~ "Elastic Net",
      TRUE ~ "Unknown"
    ),
    Selection_Rule = ifelse(grepl("CI", Method), "CI", "SN"),
    # Create a clean scenario label (same as ROC data)
    Scenario_Label = paste("Scenario", Scenario_ID, "-", 
                           gsub("Scenario \\d+: ", "", Scenario_Name))
  )

# Ensure proper factor levels
all_power_data$Method_Type <- factor(all_power_data$Method_Type,
                                     levels = c("Ridge", "Lasso", "Elastic Net"))
all_power_data$Selection_Rule <- factor(all_power_data$Selection_Rule,
                                        levels = c("CI", "SN"))



# =====================================================
# -------------------- IMPORTANT ----------------------
# =====================================================
# ------------------------------------------------------------------------------
# 15.1. FACETED POWER PLOT (Compare methods within selection rules)
# ------------------------------------------------------------------------------

cat("Creating faceted power plot: Compare methods within selection rules...\n")

facet_power_by_method <- function(scenario_id = NULL) {
  if (is.null(scenario_id)) {
    # All scenarios combined
    plot_data <- all_power_data
    title <- "Power Curve Comparison Across All Scenarios"
  } else {
    # Specific scenario
    plot_data <- all_power_data %>% filter(Scenario_ID == scenario_id)
    title <- paste("Power Comparison - Scenario", scenario_id)
  }
  
  ggplot(plot_data, aes(x = Threshold, y = Power, 
                        color = Method_Type,
                        group = Method)) +
    # Power curves
    geom_line(linewidth = 0.8, alpha = 0.9) +
    
    # Reference lines for common thresholds
    geom_vline(xintercept = 0.05, 
               color = "gray70", 
               linetype = "dotted",
               linewidth = 0.4, 
               alpha = 0.6) +
    geom_vline(xintercept = 0.5, 
               color = "gray70", 
               linetype = "dotted",
               linewidth = 0.4, 
               alpha = 0.6) +
    
    # Add text labels for reference lines
    #annotate("text", x = 0.05, y = 1.02, 
    #         label = "α=0.05", 
    #         size = 2.5, color = "gray50", hjust = 0) +
    #annotate("text", x = 0.5, y = 1.02, 
    #         label = "γ=0.5", 
    #         size = 2.5, color = "gray50", hjust = 0.5) +
    
    # Faceting
    facet_grid(Scenario_Label ~ Selection_Rule, scales = "free_x") +
    
    # Colors
    scale_color_manual(
      values = c("Ridge" = "#1F77B4", 
                 "Lasso" = "#2CA02C", 
                 "Elastic Net" = "#FF7F0E"),
      name = "Method"
    ) +
    
    # Axis settings
    scale_x_continuous(
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = 0.02)
    ) +
    scale_y_continuous(
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = 0.02)
    ) +
    
    # Labels
    labs(
      title = title,
      #subtitle = "Columns: Selection Rules (CI vs SN)\nVertical lines: Common threshold values (α=0.05, γ=0.5)",
      x = "Selection Threshold (α for CI, γ for SN)",
      y = "True Positive Rate (Power)",
      color = "Method"
    ) +
    
    # Theme
    theme_islr_custom() +
    theme(
      strip.text.y = element_blank(),
      strip.text.x = element_text(size = 8,
                                  family = "sans", 
                                  margin = margin(t = 4, b = 4, unit = "pt")),
      panel.spacing = unit(12, "pt"),
      legend.position = "bottom",
      legend.box = "horizontal",
      plot.title = element_text(hjust = 0.5, size = 14),
      plot.subtitle = element_text(hjust = 0.5, size = 11)
    ) +
    
    # Coordinates
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1))
}

# Create faceted power plots for each scenario individually
cat("\nCreating individual scenario power plots...\n")
for (i in 1:5) {
  cat(paste("\nCreating faceted power plot for Scenario", i, "...\n"))
  p_power_scenario <- facet_power_by_method(i)
  print(p_power_scenario)
  cat("Press Enter to continue...")
  invisible(readline())
}

# Create one large faceted power plot with all scenarios
cat("\nCreating combined faceted power plot for all scenarios...\n")
p_power_all <- facet_power_by_method(NULL)
print(p_power_all)








# ------------------------------------------------------------------------------
# 15.2. POWER AT SPECIFIC THRESHOLDS (Bar plot comparison)
# ------------------------------------------------------------------------------

cat("\nCreating power comparison at specific thresholds...\n")

create_power_at_thresholds <- function(scenario_id = NULL) {
  if (is.null(scenario_id)) {
    plot_data <- all_power_data
    title <- "Power at Key Thresholds Across All Scenarios"
  } else {
    plot_data <- all_power_data %>% filter(Scenario_ID == scenario_id)
    title <- paste("Power at Key Thresholds - Scenario", scenario_id)
  }
  
  # Define key thresholds for CI and SN
  key_thresholds <- data.frame(
    Selection_Rule = c("CI", "SN"),
    Threshold = c(0.05, 0.5)  # α=0.05 for CI, γ=0.5 for SN
  )
  
  # Interpolate power at key thresholds
  power_at_thresholds <- plot_data %>%
    group_by(Scenario_Label, Selection_Rule, Method_Type, Method) %>%
    do({
      df <- .
      threshold_value <- key_thresholds$Threshold[key_thresholds$Selection_Rule == unique(df$Selection_Rule)]
      approx_power <- approx(df$Threshold, df$Power, xout = threshold_value)$y
      data.frame(Threshold = threshold_value, Power = approx_power)
    }) %>%
    ungroup()
  
  ggplot(power_at_thresholds, aes(x = Method_Type, y = Power, 
                                  fill = Method_Type)) +
    geom_bar(stat = "identity", position = position_dodge(0.9), 
             width = 0.7, alpha = 0.8) +
    geom_text(aes(label = round(Power, 3)), 
              position = position_dodge(0.9), 
              vjust = -0.5, size = 2.5) +
    facet_grid(Scenario_Label ~ Selection_Rule, scales = "fixed") +
    scale_fill_manual(
      values = c("Ridge" = "#1F77B4", 
                 "Lasso" = "#2CA02C", 
                 "Elastic Net" = "#FF7F0E"),
      name = "Method"
    ) +
    scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = c(0, 0.1))
    ) +
    labs(
      title = title,
      subtitle = "Power at α=0.05 (CI) and γ=0.5 (SN)",
      x = "Method",
      y = "True Positive Rate (Power)",
      fill = "Method"
    ) +
    theme_islr_custom() +
    theme(
      strip.text.y = element_blank(),
      strip.text.x = element_text(size = 8, face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "bottom"
    )
}

# Create power at thresholds plots
cat("\nCreating power at thresholds plots...\n")
for (i in 1:5) {
  cat(paste("\nCreating power at thresholds for Scenario", i, "...\n"))
  p_power_thresholds <- create_power_at_thresholds(i)
  print(p_power_thresholds)
  cat("Press Enter to continue...")
  invisible(readline())
}

p_power_thresholds_all <- create_power_at_thresholds(NULL)
print(p_power_thresholds_all)



# 
# # ------------------------------------------------------------------------------
# # 15.3. POWER CURVE DENSITY PLOT (All scenarios combined)
# # ------------------------------------------------------------------------------
# 
# cat("\nCreating power curve density plot...\n")
# 
# create_power_density <- function() {
#   # Calculate average power across scenarios for each method
#   power_summary <- all_power_data %>%
#     group_by(Scenario_ID, Method_Type, Selection_Rule, Threshold) %>%
#     summarise(
#       Avg_Power = mean(Power, na.rm = TRUE),
#       SD_Power = sd(Power, na.rm = TRUE),
#       .groups = "drop"
#     )
#   
#   ggplot(power_summary, aes(x = Threshold, y = Avg_Power, 
#                             color = Method_Type,
#                             fill = Method_Type,
#                             group = interaction(Method_Type, Selection_Rule))) +
#     # Average power line
#     geom_line(linewidth = 0.8) +
#     
#     # Confidence band (mean ± SD)
#     geom_ribbon(aes(ymin = Avg_Power - SD_Power, 
#                     ymax = Avg_Power + SD_Power),
#                 alpha = 0.2, color = NA) +
#     
#     # Reference lines
#     geom_vline(xintercept = 0.05, linetype = "dotted", color = "gray50", alpha = 0.6) +
#     geom_vline(xintercept = 0.5, linetype = "dotted", color = "gray50", alpha = 0.6) +
#     
#     # Faceting
#     facet_grid(Selection_Rule ~ ., scales = "fixed") +
#     
#     # Colors
#     scale_color_manual(
#       values = c("Ridge" = "#1F77B4", 
#                  "Lasso" = "#2CA02C", 
#                  "Elastic Net" = "#FF7F0E"),
#       name = "Method"
#     ) +
#     scale_fill_manual(
#       values = c("Ridge" = "#1F77B4", 
#                  "Lasso" = "#2CA02C", 
#                  "Elastic Net" = "#FF7F0E"),
#       name = "Method"
#     ) +
#     
#     # Axis settings
#     scale_x_continuous(
#       breaks = seq(0, 1, 0.2),
#       expand = expansion(mult = 0.02)
#     ) +
#     scale_y_continuous(
#       breaks = seq(0, 1, 0.2),
#       expand = expansion(mult = 0.02)
#     ) +
#     
#     # Labels
#     labs(
#       title = "Power Curves with Variability Across Scenarios",
#       subtitle = "Lines: Average power | Bands: Mean ± SD across 5 scenarios\nColumns: Selection Rules",
#       x = "Selection Threshold (α for CI, γ for SN)",
#       y = "True Positive Rate (Power)"
#     ) +
#     
#     # Theme
#     theme_islr_custom() +
#     theme(
#       strip.text = element_text(size = 10, face = "bold"),
#       legend.position = "bottom",
#       panel.spacing = unit(15, "pt")
#     ) +
#     
#     # Coordinates
#     coord_cartesian(xlim = c(0, 1), ylim = c(0, 1))
# }
# 
# p_power_density <- create_power_density()
# print(p_power_density)
# 
# # ------------------------------------------------------------------------------
# # 15.4. POWER SLOPE ANALYSIS (Rate of power decay)
# # ------------------------------------------------------------------------------
# 
# cat("\nCreating power slope analysis plot...\n")
# 
# create_power_slope <- function(scenario_id = NULL) {
#   if (is.null(scenario_id)) {
#     plot_data <- all_power_data
#     title <- "Power Decay Rate Across All Scenarios"
#   } else {
#     plot_data <- all_power_data %>% filter(Scenario_ID == scenario_id)
#     title <- paste("Power Decay Rate - Scenario", scenario_id)
#   }
#   
#   # Calculate local slopes
#   slope_data <- plot_data %>%
#     group_by(Scenario_Label, Selection_Rule, Method_Type, Method) %>%
#     arrange(Threshold) %>%
#     mutate(
#       dPower = c(NA, diff(Power)),
#       dThreshold = c(NA, diff(Threshold)),
#       Slope = dPower / dThreshold,
#       # Categorize slope
#       Slope_Type = case_when(
#         Slope < -1 ~ "Steep decay",
#         Slope >= -1 & Slope < -0.5 ~ "Moderate decay",
#         Slope >= -0.5 & Slope < 0 ~ "Slow decay",
#         Slope >= 0 ~ "Increasing"
#       )
#     ) %>%
#     filter(!is.na(Slope)) %>%
#     ungroup()
#   
#   ggplot(slope_data, aes(x = Threshold, y = Slope, 
#                          color = Method_Type,
#                          group = Method)) +
#     geom_line(linewidth = 0.7, alpha = 0.8) +
#     geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", alpha = 0.6) +
#     geom_hline(yintercept = -1, linetype = "dotted", color = "red", alpha = 0.4) +
#     
#     facet_grid(Scenario_Label ~ Selection_Rule, scales = "free_x") +
#     
#     scale_color_manual(
#       values = c("Ridge" = "#1F77B4", 
#                  "Lasso" = "#2CA02C", 
#                  "Elastic Net" = "#FF7F0E"),
#       name = "Method"
#     ) +
#     
#     # Labels
#     labs(
#       title = title,
#       subtitle = "Slope = ΔPower/ΔThreshold\nNegative values: Power decays with stricter thresholds\nDashed line: Zero slope | Dotted line: Slope = -1",
#       x = "Selection Threshold",
#       y = "Slope (ΔPower/ΔThreshold)",
#       color = "Method"
#     ) +
#     
#     theme_islr_custom() +
#     theme(
#       strip.text.y = element_blank(),
#       strip.text.x = element_text(size = 8, face = "bold"),
#       legend.position = "bottom",
#       panel.spacing = unit(12, "pt")
#     ) +
#     
#     coord_cartesian(ylim = c(-2, 0.5))
# }
# 
# # Create power slope plots
# cat("\nCreating power slope plots...\n")
# for (i in 1:5) {
#   cat(paste("\nCreating power slope plot for Scenario", i, "...\n"))
#   p_power_slope <- create_power_slope(i)
#   print(p_power_slope)
#   cat("Press Enter to continue...")
#   invisible(readline())
# }
# 
# p_power_slope_all <- create_power_slope(NULL)
# print(p_power_slope_all)
# 
# # ------------------------------------------------------------------------------
# # 16. SAVE ALL POWER PLOTS
# # ------------------------------------------------------------------------------
# 
# cat("\nSaving all power plots...\n")
# 
# # Create power plots directory
# power_dir <- "power_comparison_plots"
# if (!dir.exists(power_dir)) {
#   dir.create(power_dir)
#   cat("Created directory:", power_dir, "\n")
# }
# 
# # Save individual scenario power plots
# for (i in 1:5) {
#   p_power <- facet_power_by_method(i)
#   ggsave(file.path(power_dir, paste0("scenario", i, "_faceted_power.pdf")), 
#          p_power, width = 8, height = 10, dpi = 300)
#   ggsave(file.path(power_dir, paste0("scenario", i, "_faceted_power.png")), 
#          p_power, width = 8, height = 10, dpi = 300)
# }
# 
# # Save combined power plots
# ggsave(file.path(power_dir, "all_scenarios_faceted_power.pdf"), 
#        p_power_all, width = 8, height = 12, dpi = 300)
# ggsave(file.path(power_dir, "all_scenarios_faceted_power.png"), 
#        p_power_all, width = 8, height = 12, dpi = 300)
# 
# # Save power at thresholds
# ggsave(file.path(power_dir, "power_at_thresholds.pdf"), 
#        p_power_thresholds_all, width = 8, height = 12, dpi = 300)
# 
# # Save power density
# ggsave(file.path(power_dir, "power_density.pdf"), 
#        p_power_density, width = 10, height = 6, dpi = 300)
# 
# cat("\nAll power plots saved to:", power_dir, "\n")
# 
# # ------------------------------------------------------------------------------
# # 17. UPDATE FINAL SUMMARY
# # ------------------------------------------------------------------------------
# 
# cat("\n" %+% strrep("=", 70) %+% "\n")
# cat("POWER PLOTS ADDED SUCCESSFULLY!\n")
# cat(strrep("=", 70) %+% "\n")
# cat("\nNEW POWER PLOTS CREATED:\n")
# cat("1. Faceted power curves (similar to ROC faceted plots)\n")
# cat("2. Power at key thresholds (α=0.05 for CI, γ=0.5 for SN)\n")
# cat("3. Power density with variability bands\n")
# cat("4. Power slope analysis (decay rate)\n")
# cat(strrep("=", 70) %+% "\n")
# 




























































