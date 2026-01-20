
# ===========================================================================================
# ======= Bayesian Elastic Net Regression (Multiple Scenarios and bootstrapped SE) ==========
# ===========================================================================================


# ------------------------------------------------------------------
# 1. LOAD REQUIRED PACKAGES
# ------------------------------------------------------------------

library(rjags)      
library(coda)     
library(fda)         # For using inprod function
library(tidyverse)  
library(MASS)       


# ------------------------------------------------------------------
# 2. SCENARIO DEFINITIONS
# ------------------------------------------------------------------

# define_scenarios = function() {
#   scenarios = list()
#   
#   # -----------------------------------------------------------------
#   # SCENARIO 1: Simple independent predictors
#   # -----------------------------------------------------------------
#   scenarios[[1]] = list(
#     name = "Scenario 1: Independent predictors",
#     description = "All predictors independent with varying signal strengths",
#     p = 20,                           
#     beta = c(rep(2, 5), rep(1, 5), rep(0.5, 5), rep(0, 5)),              
#     sigma = 3,                        
#     data_gen = function(n, p, beta, sigma) {
#       # Independent predictors
#       X = matrix(rnorm(n * p), n, p)  
#       X = scale(X)                    
#       y = as.numeric(X %*% beta + rnorm(n, 0, sigma))  
#       
#       list(X = X, y = y)
#     }
#   )
#   
#   # -----------------------------------------------------------------
#   # SCENARIO 2: Group structure (correlated within groups)
#   # -----------------------------------------------------------------
#   scenarios[[2]] = list(
#     name = "Scenario 2: Within-group correlation",
#     description = "Predictors form groups with high correlation within groups (ρ=0.8)",
#     p = 30,                          
#     beta = c(rep(3, 6), rep(-3, 6), rep(0, 18)),
#     sigma = 10,                       
#     data_gen = function(n, p, beta, sigma) {
#       
#       # Create 3 groups: first two have signals, last is noise
#       X = matrix(0, n, p)
#       
#       # Group 1: variables 1-6, correlated with ρ=0.8
#       for (i in 1:6) {
#         X[, i] = rnorm(n)
#         if (i > 1) {
#           X[, i] = 0.8 * X[, 1] + sqrt(1 - 0.8^2) * rnorm(n)
#         }
#       }
#       
#       # Group 2: variables 7-12, correlated with ρ=0.8
#       for (i in 7:12) {
#         X[, i] = rnorm(n)
#         if (i > 7) {
#           X[, i] = 0.8 * X[, 7] + sqrt(1 - 0.8^2) * rnorm(n)
#         }
#       }
#       
#       # Group 3: variables 13-30, independent noise
#       for (i in 13:30) {
#         X[, i] = rnorm(n)
#       }
#       
#       X = scale(X)
#       y = as.numeric(X %*% beta + rnorm(n, 0, sigma))
#       
#       list(X = X, y = y)
#     }
#   )
#   
#   # -----------------------------------------------------------------
#   # SCENARIO 3: Autoregressive correlation (Zou & Hastie style)
#   # -----------------------------------------------------------------
#   scenarios[[3]] = list(
#     name = "Scenario 3: Autoregressive correlation",
#     description = "Correlation decays with distance: Corr(x_i, x_j) = ρ^|i-j|",
#     p = 40,                        
#     beta = c(rep(2, 10), rep(0, 30)),
#     sigma = 8,                       
#     data_gen = function(n, p, beta, sigma) {
#       # AR(1) correlation structure with ρ = 0.7
#       rho = 0.7
#       
#       # Generate from multivariate normal with AR(1) covariance
#       Sigma = rho^abs(outer(1:p, 1:p, "-"))
#       X = mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
#       
#       X = scale(X)
#       y = as.numeric(X %*% beta + rnorm(n, 0, sigma))
#       
#       list(X = X, y = y)
#     }
#   )
#   
#   # -----------------------------------------------------------------
#   # SCENARIO 4: Mixture of strong & weak signals (Tibshirani style)
#   # -----------------------------------------------------------------
#   scenarios[[4]] = list(
#     name = "Scenario 4: Mixture of signal strengths",
#     description = "Few strong signals, many weak signals, rest zero",
#     p = 50,                        
#     beta = c(rep(4, 3), rep(1.5, 7), rep(0.5, 10), rep(0, 30)),
#     sigma = 12,                      
#     data_gen = function(n, p, beta, sigma) {
#       # Moderate correlation among all predictors (ρ = 0.3)
#       rho = 0.3
#       Sigma = matrix(rho, p, p)
#       diag(Sigma) = 1
#       
#       X = mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
#       X = scale(X)
#       y = as.numeric(X %*% beta + rnorm(n, 0, sigma))
#       
#       list(X = X, y = y)
#     }
#   )
#   
#   # -----------------------------------------------------------------
#   # SCENARIO 5: Very high correlation (challenge for LASSO)
#   # -----------------------------------------------------------------
#   scenarios[[5]] = list(
#     name = "Scenario 5: High correlation challenge",
#     description = "Pairs of highly correlated predictors (ρ=0.95) with different coefficients",
#     p = 20,                        
#     beta = c(3, 0, 0, 3, 0, 0, 3, 0, 0, rep(0, 11)),
#     sigma = 6,                      
#     data_gen = function(n, p, beta, sigma) {
#       # Create correlated pairs: (x1,x2), (x4,x5), (x7,x8) have ρ=0.95
#       X = matrix(rnorm(n * p), n, p)
#       
#       # Make pairs highly correlated
#       pairs = list(c(1,2), c(4,5), c(7,8))
#       for(pair in pairs) {
#         i = pair[1]
#         j = pair[2]
#         X[, j] = 0.95 * X[, i] + sqrt(1 - 0.95^2) * rnorm(n)
#       }
#       
#       X = scale(X)
#       y = as.numeric(X %*% beta + rnorm(n, 0, sigma))
#       
#       list(X = X, y = y)
#     }
#   )
#   
#   return(scenarios)
# }
# 
# 


# ------------------------------------------------------------------
# 3. CORE ANALYSIS FUNCTIONS
# ------------------------------------------------------------------


#  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# 3.1 | Elastic Net Model Code
#  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-


en_model_code <- "
model{

  # --------------------------------------------------
  # Likelihood
  # --------------------------------------------------
  for(i in 1:n){
    y[i] ~ dnorm(mu[i], tau_eps)
    mu[i] <- inprod(X[i,], beta[])
  }

  # --------------------------------------------------
  # Hyperparameter definitions
  # --------------------------------------------------
  lambda1_sq <- lambda1 * lambda1
  rate_tau   <- lambda1_sq / (8 * lambda2 * sigma2)

  # --------------------------------------------------
  # Elastic Net prior (Li & Lin)
  # --------------------------------------------------
  for(j in 1:p){
    beta[j] ~ dnorm(0, prec_beta[j])
    prec_beta[j] <- (lambda2 * tau_j[j]) / (sigma2 * (tau_j[j] - 1))

    tau_j_raw[j] ~ dgamma(0.5, rate_tau)
    tau_j[j] <- tau_j_raw[j] + 1
  }

  # --------------------------------------------------
  # Error variance
  # --------------------------------------------------
  tau_eps ~ dgamma(0.001, 0.001)
  sigma2 <- 1 / tau_eps
}
"




#  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# 3.2 | Monte Carlo Expectation Maximization (λ1,λ2) estimation
#  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

library(glmnet)
get_optimal_lambda <- function(X, y, 
                               max_iter,
                               mcmc_iter,
                               burn_in,
                               tol,
                               verbose) {
  
  # Convert inputs
  X <- as.matrix(X)
  y <- as.vector(y)
  n <- nrow(X)
  p <- ncol(X)
  
  if (p == 0) return(list(lambda1 = NA, lambda2 = NA))
  
  # ----------------------------------------------------------
  # Standardize data
  # ----------------------------------------------------------
  y_centered <- y - mean(y)
  
  # Standardize X
  col_means <- colMeans(X)
  col_sds <- apply(X, 2, sd)
  
  if (any(abs(col_means) > 1e-8) || any(abs(col_sds - 1) > 1e-8)) {
    if (verbose) cat("Standardizing X...\n")
    X_std <- scale(X, center = TRUE, scale = TRUE)
  } else {
    X_std <- X
  }
  
  # ----------------------------------------------------------
  # ROBUST INITIALIZATION (Handles p > n)
  # ----------------------------------------------------------
  
  if (p < n - 1) {
    # CASE 1: p < n (Low-dimensional) - Use OLS
    if (verbose) cat("Low-dimensional (p < n): Using OLS for initialization\n")
    fit_lm <- lm(y_centered ~ X_std - 1)
    beta_ls <- coef(fit_lm)
    sigma_ls <- summary(fit_lm)$sigma
    
    lambda1_old <- p * sigma_ls / (sum(abs(beta_ls)) + 1e-10)
    lambda2_old <- 0.1  # Small initial L2 penalty
    
  } else {
    # CASE 2: p >= n (High-dimensional) - Use smarter initialization
    if (verbose) {
      cat(sprintf("High-dimensional detected: p = %d >= n = %d\n", p, n))
      cat("Using Lasso-CV for initialization\n")
    }
    
    # Use glmnet with cross-validation to get initial lambda
    
    cv_fit = cv.glmnet(X_std, y_centered, alpha = 0.5)  # Elastic Net with alpha=0.5
    lambda1_old = cv_fit$lambda.min * 0.5  # Convert to Bayesian scale
    lambda2_old = cv_fit$lambda.min * 0.5
  }
  # else {
  #   # CASE 2: p >= n (High-dimensional) - Use fixed values
  #   if (verbose) {
  #     cat(sprintf("High-dimensional detected: p = %d >= n = %d\n", p, n))
  #     cat("Using fixed initialization\n")
  #   }
  #   
  #   # Fixed reasonable initial values
  #   lambda1_old <- 1.0   # Moderate L1 penalty
  #   lambda2_old <- 0.5   # Moderate L2 penalty
  #   
  #   # Optional: You could add dimension-based scaling
  #   # lambda1_old <- sqrt(p/n)  # Scales with dimension ratio
  #   # lambda2_old <- 0.5 * sqrt(p/n)
  # }
  
  # Apply sensible bounds to initial values
  lambda1_old <- max(min(lambda1_old, 10), 0.1)
  lambda2_old <- max(min(lambda2_old, 10), 0.01)
  
  if (verbose) {
    cat("=", 60, "\n")
    cat("MCEM for Bayesian Elastic Net\n")
    cat(sprintf("Dimensions: n = %d, p = %d\n", n, p))
    cat(sprintf("Initial λ₁: %.6f, λ₂: %.6f\n", lambda1_old, lambda2_old))
    cat("=", 60, "\n")
  }
  
  # ----------------------------------------------------------
  # Save model to temporary file
  # ----------------------------------------------------------
  temp_model_file <- tempfile(fileext = ".txt")
  writeLines(en_model_code, temp_model_file)
  on.exit(unlink(temp_model_file))
  
  # ----------------------------------------------------------
  # Monte Carlo EM Loop
  # ----------------------------------------------------------
  for (iter in 1:max_iter) {
    
    if (verbose) cat(sprintf("Iter %d: ", iter))
    
    # Prepare data for JAGS
    jags_data <- list(
      X = X_std,
      y = y_centered,
      n = n,
      p = p,
      lambda1 = lambda1_old,
      lambda2 = lambda2_old
    )
    
    tryCatch({
      model <- jags.model(
        file = temp_model_file,
        data = jags_data,
        n.chains = 1,
        quiet = TRUE
      )
      
      update(model, burn_in, progress.bar = "none")
      
      samples <- coda.samples(
        model,
        variable.names = c("tau_j", "beta"),
        n.iter = mcmc_iter,
        thin = 5,
        progress.bar = "none"
      )
      
      samples_mat <- as.matrix(samples)
      
      # Extract samples
      tau_j_cols <- grep("tau_j", colnames(samples_mat))
      beta_cols <- grep("beta", colnames(samples_mat))
      
      if (length(tau_j_cols) == 0 || length(beta_cols) == 0) {
        if (verbose) cat("Could not extract samples.\n")
        next
      }
      
      tau_j_mat <- samples_mat[, tau_j_cols, drop = FALSE]
      beta_mat <- samples_mat[, beta_cols, drop = FALSE]
      
      # Compute expectations
      E_sum_tau_j <- sum(colMeans(tau_j_mat))
      E_sum_beta2 <- sum(colMeans(beta_mat^2))
      
      # EM updates (same for all cases - MCEM will adjust)
      lambda1_new <- p / (E_sum_tau_j/2 + 1e-10)
      lambda2_new <- p / (E_sum_beta2 + 1e-10)
      
      # Prevent extreme values
      lambda1_new <- max(min(lambda1_new, 100), 0.001)
      lambda2_new <- max(min(lambda2_new, 100), 0.001)
      
      # Convergence check
      lambda1_diff <- abs(lambda1_new - lambda1_old)
      lambda2_diff <- abs(lambda2_new - lambda2_old)
      max_diff <- max(lambda1_diff, lambda2_diff)
      
      if (verbose) {
        cat(sprintf("λ₁ = %.6f, λ₂ = %.6f, |Δ| = %.6f", 
                    lambda1_new, lambda2_new, max_diff))
        if (max_diff < tol) cat(" ✓")
        cat("\n")
      }
      
      # Check convergence
      if (max_diff < tol && iter >= 2) {
        if (verbose) {
          cat("\n", strrep("-", 50), "\n")
          cat(sprintf("✓ CONVERGED after %d iterations!\n", iter))
          cat(sprintf("Final λ₁: %.6f, λ₂: %.6f (max change < %.6f)\n", 
                      lambda1_new, lambda2_new, tol))
          cat(strrep("-", 50), "\n")
        }
        return(list(lambda1 = lambda1_new, lambda2 = lambda2_new))
      }
      
      # Update for next iteration
      lambda1_old <- lambda1_new
      lambda2_old <- lambda2_new
      
    }, error = function(e) {
      if (verbose) cat(sprintf("Error: %s\n", e$message))
    })
  }
  
  # If we reach here, didn't converge
  if (verbose) {
    cat("\n", strrep("*", 50), "\n")
    cat(sprintf("Warning: MCEM did not converge in %d iterations\n", max_iter))
    cat(sprintf("Returning current values: λ₁ = %.6f, λ₂ = %.6f\n", 
                lambda1_old, lambda2_old))
    cat(strrep("*", 50), "\n")
  }
  
  return(list(lambda1 = lambda1_old, lambda2 = lambda2_old))
}





# 
# set.seed(123)
# n = 100
# p = 10
# 
# X_test1 = matrix(rnorm(n*p),n,p)
# beta_true = c(2,-1.5,1,rep(0,p-3))
# y_test1 = X_test1%*%beta_true + rnorm(n,0,1)
# 
# cat("Data dimensions:", n, "observations,", p, "predictors\n")
# cat("True signal: β =", round(beta_true, 2), "\n")
# 
# result = get_optimal_lambda(
#   X = X_test1,
#   y = y_test1,
#   max_iter = 150,
#   mcmc_iter = 2000,
#   burn_in = 1000,
#   tol = 1e-3,
#   verbose = TRUE
# )
# 
# 
# result
# 

#  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# 3.3 | Fit Bayesian EN Regression
#  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

fit_bayes_en = function(X, y) {
  X = as.matrix(X)
  p = ncol(X)
  n = nrow(X)
  
  if (p == 0) {
    return(list(
      beta_samples = matrix(0, nrow = 1, ncol = 0),
      beta_mean = numeric(0),
      lambda1 = NA,
      lambda2 = NA
    ))
  }
  
  # MCEM optimization for both lambda1 and lambda2
  
  
  if(p < n & p > 0){
  lambda_hat = get_optimal_lambda(
    X = X,
    y = y,
    max_iter = 450,                       # ===========================================================
    mcmc_iter = 500,                      # ===========================================================
    burn_in = 100,                        # ===========================================================
    tol = 0.001,                          # ===========================================================
    verbose = FALSE                       # ===========================================================
  )
  }else{
    lambda_hat = get_optimal_lambda(
      X = X,
      y = y,
      max_iter = 450,                       # ===========================================================
      mcmc_iter = 500,                      # ===========================================================
      burn_in = 100,                        # ===========================================================
      tol = 0.019,                          # ===========================================================
      verbose = FALSE)                       # ===========================================================
  }
  # Use elastic net model code
  model = jags.model(
    textConnection(en_model_code),
    data = list(
      X = X,
      y = y, 
      n = nrow(X), 
      p = ncol(X),
      lambda1 = lambda_hat$lambda1,
      lambda2 = lambda_hat$lambda2
    ),
    quiet = TRUE
  )
  
  update(model, 5000)
  samples = coda.samples(model, "beta", 5000)
  beta_samples = as.matrix(samples)
  
  list(
    beta_samples = beta_samples, 
    beta_mean = colMeans(beta_samples),
    lambda1 = lambda_hat$lambda1,  # Return estimated lambda1
    lambda2 = lambda_hat$lambda2   # Return estimated lambda2
  )
}


# 3.3.1 | Confidence interval selection rule
# ----------------------------------------------------

select_ci_alpha = function(beta_samples, alpha) {
  ci = apply(beta_samples, 2, quantile, probs = c(alpha/2, 1 - alpha/2))
  !(ci[1, ] <= 0 & ci[2, ] >= 0)
}


# 3.3.2 | Scaled neighborhood selection rule
# ----------------------------------------------------
select_scaled_neighbourhood = function(beta_samples, gamma) {
  post_var = apply(beta_samples, 2, var)
  prob_inside = sapply(seq_len(ncol(beta_samples)), function(j) {
    mean(beta_samples[, j] >= -sqrt(post_var[j]) & 
           beta_samples[, j] <= sqrt(post_var[j]))
  })
  prob_inside <= gamma
}


# 3.3.3 | Calculate true/false positive rates
# ----------------------------------------------------
tpr_fpr = function(selected, beta_true) {
  signal = beta_true != 0
  TP = sum(selected & signal)
  FP = sum(selected & !signal)
  TN = sum(!selected & !signal)
  FN = sum(!selected & signal)
  # Handle division by zero
  TPR = if (TP + FN == 0) 0 else TP/(TP + FN)  # If no signals, TPR = 0#===================================
  FPR = if (FP + TN == 0) 0 else FP/(FP + TN)  # If no noise, FPR = 0
  
  c(TPR = TPR, FPR = FPR)
  #c(TPR = TP/(TP + FN), FPR = FP/(FP + TN))
}

# 3.3.4 | Bootstrap SE Function
# ----------------------------------------------------
bootstrap_se = function(x, B = 1000) {
  # Calculate bootstrap standard error of the median
  # Args:
  #   x: numeric vector
  #   B: number of bootstrap samples
  
  if (length(x) == 0 || all(is.na(x))) return(NA)
  
  # Generate bootstrap samples of the median
  boot_medians = replicate(B, {
    x_boot = sample(x, replace = TRUE)
    median(x_boot)
  })
  
  # Calculate standard deviation of bootstrap medians
  sd(boot_medians)
}



# ------------------------------------------------------------------
# 4. Main Simulation Engine and storing all the necessary records
# ------------------------------------------------------------------

run_scenario_simulation = function(
    scenario,         # Scenario definition from define_scenarios()
    n_sim,            # Number of simulation repetitions
    n_train,          # Training sample size
    n_total,          # Total sample size
    alpha_grid,       # CI significance levels
    gamma_grid        # SN probability thresholds
) {
  
  p = scenario$p
  beta_true = scenario$beta
  sigma = scenario$sigma
  
  # CHANGE: Update method names from "Lasso" to "Elastic Net"
  # Arrays to store ROC results
  roc_ci = array(0, c(n_sim, length(alpha_grid), 2))
  roc_sn = array(0, c(n_sim, length(gamma_grid), 2))
  
  # Matrices to store selection results
  sel_matrix_ci = matrix(0, n_sim, p)
  sel_matrix_sn = matrix(0, n_sim, p)
  
  # Matrix to store mean squared error
  mse_matrix = matrix(0, n_sim, 2)
  
  # Identify signal and noise indices
  signal_idx = which(beta_true != 0)
  noise_idx = which(beta_true == 0)
  
  # Main simulation loop
  for (s in 1:n_sim) {
    cat(sprintf("  Simulation %d/%d\r", s, n_sim))
    
    # Generate data
    dat = scenario$data_gen(n_total, p, beta_true, sigma)
    
    # Randomly split into training and testing
    idx = sample(1:n_total, n_train)
    test_idx = setdiff(1:n_total, idx)
    
    # KEY CHANGE: Use fit_bayes_en() instead of fit_bayes_lasso()
    # Fit Bayesian Elastic Net on training data
    fit = fit_bayes_en(dat$X[idx, ], dat$y[idx])
    
    # Find median thresholds
    median_alpha = 0.5 # alpha_grid[which.min(abs(alpha_grid - 0.05))] ===========================================
    median_gamma = 0.5 # alpha_grid[which.min(abs(alpha_grid - 0.05))] ===========================================
    
    # CI method selection at median α
    sel_ci = select_ci_alpha(fit$beta_samples, median_alpha)
    sel_matrix_ci[s, ] = sel_ci
    
    # SN method selection at median γ
    sel_sn = select_scaled_neighbourhood(fit$beta_samples, median_gamma)
    sel_matrix_sn[s, ] = sel_sn
    
    # ===== CORRECTED MSE CALCULATION =====
    
    # For CI method
    if (sum(sel_ci) > 0) {
      # Get indices of selected variables
      selected_idx_ci = which(sel_ci)
      
      # Fit model with ONLY selected variables
      X_train_ci = dat$X[idx, selected_idx_ci, drop = FALSE]
      # CHANGE: Use fit_bayes_en() here too
      fit_ci = fit_bayes_en(X_train_ci, dat$y[idx])
      
      # Predict on test data
      if (length(fit_ci$beta_mean) > 0) {
        X_test_ci = dat$X[test_idx, selected_idx_ci, drop = FALSE]
        y_pred_ci = X_test_ci %*% fit_ci$beta_mean
        mse_matrix[s, 1] = mean((y_pred_ci - dat$y[test_idx])^2)
      } else {
        mse_matrix[s, 1] = mean((mean(dat$y[idx]) - dat$y[test_idx])^2)
      }
    } else {
      mse_matrix[s, 1] = mean((mean(dat$y[idx]) - dat$y[test_idx])^2)
    }
    
    # For SN method
    if (sum(sel_sn) > 0) {
      # Get indices of selected variables
      selected_idx_sn = which(sel_sn)
      
      # Fit model with ONLY selected variables
      X_train_sn = dat$X[idx, selected_idx_sn, drop = FALSE]
      # CHANGE: Use fit_bayes_en() here too
      fit_sn = fit_bayes_en(X_train_sn, dat$y[idx])
      
      # Predict on test data
      if (length(fit_sn$beta_mean) > 0) {
        X_test_sn = dat$X[test_idx, selected_idx_sn, drop = FALSE]
        y_pred_sn = X_test_sn %*% fit_sn$beta_mean
        mse_matrix[s, 2] = mean((y_pred_sn - dat$y[test_idx])^2)
      } else {
        mse_matrix[s, 2] = mean((mean(dat$y[idx]) - dat$y[test_idx])^2)
      }
    } else {
      mse_matrix[s, 2] = mean((mean(dat$y[idx]) - dat$y[test_idx])^2)
    }
    
    # ===== ROC CALCULATIONS =====
    
    # Calculate ROC for CI method across α grid
    for (k in seq_along(alpha_grid)) {
      sel = select_ci_alpha(fit$beta_samples, alpha_grid[k])
      roc_ci[s, k, ] = tpr_fpr(sel, beta_true)
    }
    
    # Calculate ROC for SN method across γ grid
    for (k in seq_along(gamma_grid)) {
      sel = select_scaled_neighbourhood(fit$beta_samples, gamma_grid[k])
      roc_sn[s, k, ] = tpr_fpr(sel, beta_true)
    }
  }
  
  cat("\n")  # New line after progress
  
  # ===== CREATE RESULTS =====
  
  # ROC curve data
  # CHANGE: Update method names to "Elastic Net"
  roc_df = rbind(
    data.frame(
      FPR = colMeans(roc_ci[, , 2]),
      TPR = colMeans(roc_ci[, , 1]),
      Method = "Elastic Net + CI",  # Changed from "Lasso + CI"
      Scenario = scenario$name
    ),
    data.frame(
      FPR = colMeans(roc_sn[, , 2]),
      TPR = colMeans(roc_sn[, , 1]),
      Method = "Elastic Net + SN",  # Changed from "Lasso + SN"
      Scenario = scenario$name
    )
  )
  
  # Power curve data
  # CHANGE: Update method names to "Elastic Net"
  power_df = rbind(
    data.frame(
      Threshold = alpha_grid,
      Power = colMeans(roc_ci[, , 1]),
      Method = "Elastic Net + CI",  # Changed from "Lasso + CI"
      Scenario = scenario$name
    ),
    data.frame(
      Threshold = gamma_grid,
      Power = colMeans(roc_sn[, , 1]),
      Method = "Elastic Net + SN",  # Changed from "Lasso + SN"
      Scenario = scenario$name
    )
  )
  
  # Exclusion table
  exclusion_table = data.frame(
    Variable = paste0("β", 1:p),
    True_Value = beta_true,
    Type = ifelse(beta_true != 0, "Signal", "Noise"),
    CI_Exclude_Count = colSums(sel_matrix_ci == 0),
    SN_Exclude_Count = colSums(sel_matrix_sn == 0)
  )
  
  # Performance stats
  # CHANGE: Update method names to "Elastic Net"
  performance_stats = data.frame(
    Method = c("Elastic Net + CI", "Elastic Net + SN"),  # Changed from "Lasso"
    TPR = c(
      mean(rowSums(sel_matrix_ci[, signal_idx, drop = FALSE]) / length(signal_idx)),
      mean(rowSums(sel_matrix_sn[, signal_idx, drop = FALSE]) / length(signal_idx))
    ),
    FPR = c(
      mean(rowSums(sel_matrix_ci[, noise_idx, drop = FALSE]) / length(noise_idx)),
      mean(rowSums(sel_matrix_sn[, noise_idx, drop = FALSE]) / length(noise_idx))
    )
  )
  
  # Calculate precision
  TP_ci = sum(sel_matrix_ci[, signal_idx])
  FP_ci = sum(sel_matrix_ci[, noise_idx])
  TP_sn = sum(sel_matrix_sn[, signal_idx])
  FP_sn = sum(sel_matrix_sn[, noise_idx])
  
  performance_stats$Precision = c(
    ifelse(TP_ci + FP_ci > 0, TP_ci / (TP_ci + FP_ci), 0),
    ifelse(TP_sn + FP_sn > 0, TP_sn / (TP_sn + FP_sn), 0)
  )
  
  # Results list
  results = list(
    scenario_name = scenario$name,
    roc = roc_df,
    power = power_df,
    exclusion_table = exclusion_table,
    mse_summary = data.frame(
      Method = c("Elastic Net + CI", "Elastic Net + SN"),  # Changed from "Lasso"
      Mean_MSE = colMeans(mse_matrix),
      Median_MSE = apply(mse_matrix, 2, median),
      SD_MSE = apply(mse_matrix, 2, sd),
      Bootstrap_SE = c(
        bootstrap_se(mse_matrix[, 1]),
        bootstrap_se(mse_matrix[, 2])
      )
    ),
    performance_stats = performance_stats,
    beta_true = beta_true,
    n_sim = n_sim
  )
  
  return(results)
}




# ------------------------------------------------------------------
# 5. Plotting
# ------------------------------------------------------------------

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# 5.1 | Plotting Properties
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-


# 5.1.1 | Themes
# ----------------------------------------------------
theme_islr <- function(base_size = 11, base_family = "sans") {
  theme_bw(base_size = base_size, base_family = base_family) %+replace%
    theme(
      # Text elements
      plot.title = element_text(
        size = base_size + 2,
        face = "bold",
        hjust = 0.5,
        margin = margin(b = 10)
      ),
      plot.subtitle = element_text(
        size = base_size,
        hjust = 0.5,
        margin = margin(b = 15)
      ),
      axis.title = element_text(
        size = base_size,
        face = "bold"
      ),
      axis.text = element_text(size = base_size - 1),
      
      # Legend
      legend.position = "bottom",
      legend.title = element_text(size = base_size, face = "bold"),
      legend.text = element_text(size = base_size - 1),
      legend.key.size = unit(0.8, "lines"),
      legend.key.width = unit(1.5, "lines"),
      legend.margin = margin(t = -5),
      
      # Panel and plot
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(15, 15, 10, 15),
      
      # Facets
      strip.background = element_rect(
        fill = "gray95", 
        color = "black", 
        linewidth = 0.5
      ),
      strip.text = element_text(size = base_size, face = "bold")
    )
}

# 5.1.2 | Providing colors and linetypes
# ----------------------------------------------------
# CHANGE: Update colors for Elastic Net
method_colors <- c(
  "Elastic Net + CI" = "#0072B2",      # Blue (different from Lasso's green)
  "Elastic Net + SN" = "#D55E00"       # Orange (different from Lasso's purple)
)

method_linetypes <- c(
  "Elastic Net + CI" = "solid",
  "Elastic Net + SN" = "dashed"
)




# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# 5.2 | ROC Plotting for each scenario
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

plot_roc_single_scenario <- function(results) {
  
  # Calculate AUC for legend labels
  auc_values <- results$roc %>%
    group_by(Method) %>%
    arrange(FPR) %>%
    summarise(
      AUC = sum(diff(FPR) * (head(TPR, -1) + tail(TPR, -1)) / 2),
      .groups = "drop"
    )
  
  # Add AUC to method names
  roc_data <- results$roc %>%
    left_join(auc_values, by = "Method") %>%
    mutate(
      Method_Label = sprintf("%s (AUC = %.3f)", Method, AUC),
      Method_Original = Method
    )
  
  # Use dynamic method names
  all_methods <- unique(roc_data$Method_Original)
  color_mapping <- method_colors[all_methods]
  linetype_mapping <- method_linetypes[all_methods]
  names(color_mapping) <- unique(roc_data$Method_Label)
  names(linetype_mapping) <- unique(roc_data$Method_Label)
  
  # Create plot with coord_cartesian ONLY (no scale limits)
  p <- ggplot(roc_data, aes(x = FPR, y = TPR, 
                            color = Method_Label, 
                            linetype = Method_Label,
                            group = Method_Label)) +
    geom_line(linewidth = 0.8) +
    geom_abline(slope = 1, intercept = 0, 
                color = "gray50", linetype = "dashed", 
                linewidth = 0.5, alpha = 0.6) +
    scale_color_manual(values = color_mapping) +
    scale_linetype_manual(values = linetype_mapping) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    scale_x_continuous(
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = 0.02)
    ) +
    scale_y_continuous(
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = 0.02)
    ) +
    labs(
      title = paste("ROC Curve -", results$scenario_name),
      x = "False Positive Rate",
      y = "True Positive Rate",
      color = "Method",
      linetype = "Method"
    ) +
    theme_islr() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  return(p)
}


# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# 5.3 | Power Plotting for each scenario
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

plot_power_single_scenario <- function(results) {
  
  # Clean the data first to ensure no issues
  power_data <- results$power %>%
    # Ensure values are within reasonable range
    mutate(
      Threshold = pmax(0, pmin(1, Threshold)),
      Power = pmax(0, pmin(1, Power))
    )
  
  # Create plot with coord_cartesian ONLY (no scale limits)
  p <- ggplot(power_data, aes(x = Threshold, y = Power,
                              color = Method,
                              linetype = Method,
                              group = Method)) +
    geom_line(linewidth = 0.8) +
    geom_vline(xintercept = 0.05, 
               color = "gray70", 
               linetype = "dashed",
               linewidth = 0.3) +
    geom_vline(xintercept = 0.5,
               color = "gray70",
               linetype = "dashed",
               linewidth = 0.3) +
    scale_color_manual(values = method_colors) +
    scale_linetype_manual(values = method_linetypes) +
    # USE ONLY coord_cartesian for zooming - NO SCALE LIMITS
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    scale_x_continuous(
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = 0.02)
    ) +
    scale_y_continuous(
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = 0.02)
    ) +
    labs(
      title = paste("Power Curve -", results$scenario_name),
      x = "Selection Threshold (α for CI, γ for SN)",
      y = "True Positive Rate (Power)",
      color = "Method",
      linetype = "Method"
    ) +
    theme_islr() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.box = "vertical",
      legend.spacing.y = unit(0.1, "cm")
    )
  
  return(p)
}


# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# 5.4 | Count plot of exclusion for each variable
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

plot_exclusion_counts <- function(results) {
  
  df_long <- results$exclusion_table %>%
    pivot_longer(
      cols = c(CI_Exclude_Count, SN_Exclude_Count),
      names_to = "Method",
      values_to = "Exclude_Count"
    ) %>%
    mutate(
      # CHANGE: Update to Elastic Net
      Method = ifelse(Method == "CI_Exclude_Count", "Elastic Net + CI", "Elastic Net + SN"),
      Variable = factor(Variable, 
                        levels = unique(results$exclusion_table$Variable[
                          order(results$exclusion_table$True_Value, decreasing = TRUE)
                        ]))
    )
  
  ggplot(df_long, aes(x = Variable, y = Exclude_Count, fill = Method)) +
    geom_bar(stat = "identity", 
             position = position_dodge(width = 0.8),
             width = 0.7,
             color = "black",
             linewidth = 0.2) +
    geom_hline(yintercept = results$n_sim/2,
               color = "gray50",
               linetype = "dashed",
               linewidth = 0.5,
               alpha = 0.7) +
    facet_wrap(~Type, scales = "free_x", nrow = 1) +
    scale_fill_manual(values = method_colors) +
    scale_y_continuous(
      limits = c(0, results$n_sim),
      breaks = scales::pretty_breaks(n = 5)(0:results$n_sim),
      expand = expansion(mult = 0.02)
    ) +
    labs(
      title = paste("Variable Exclusion Frequency -", results$scenario_name),
      subtitle = paste("Based on", results$n_sim, "simulations"),
      x = "Predictor Variable",
      y = "Number of Exclusions",
      fill = "Method"
    ) +
    theme_islr() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
      strip.text = element_text(size = 10),
      panel.spacing = unit(1, "lines"),
      plot.title = element_text(hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5)
    )
}

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# 5.5 | Plotting of MSE
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

plot_mse_comparison <- function(results) {
  
  mse_data <- results$mse_summary
  
  ggplot(mse_data, aes(x = Method, y = Median_MSE, fill = Method)) +
    geom_bar(stat = "identity", 
             width = 0.6,
             color = "black",
             linewidth = 0.2) +
    geom_errorbar(aes(ymin = Median_MSE - Bootstrap_SE,
                      ymax = Median_MSE + Bootstrap_SE),
                  width = 0.2,
                  linewidth = 0.8,
                  color = "black") +
    scale_fill_manual(values = method_colors) +
    labs(
      title = paste("Test MSE Comparison -", results$scenario_name),
      subtitle = "Median MSE ± Bootstrap Standard Error",
      x = "Method",
      y = "Median Test MSE"
    ) +
    theme_islr() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5),
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      legend.position = "none"
    )
}

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# 5.6 | Combined Performance Plot and Comparison Plot
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# -------------------------------------------------
plot_performance_summary <- function(results) {
  
  perf_data <- results$performance_stats %>%
    pivot_longer(cols = c(TPR, FPR, Precision),
                 names_to = "Metric",
                 values_to = "Value") %>%
    mutate(Metric = factor(Metric, levels = c("TPR", "FPR", "Precision")))
  
  ggplot(perf_data, aes(x = Metric, y = Value, fill = Method)) +
    geom_bar(stat = "identity",
             position = position_dodge(width = 0.8),
             width = 0.7,
             color = "black",
             linewidth = 0.2) +
    scale_fill_manual(values = method_colors) +
    scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = 0.02)
    ) +
    labs(
      title = paste("Performance Summary -", results$scenario_name),
      x = "Performance Metric",
      y = "Value",
      fill = "Method"
    ) +
    theme_islr() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.x = element_text(angle = 0, hjust = 0.5)
    )
}
# --------------------------------------------------
plot_roc_comparison <- function(results_list) {
  
  roc_combined <- do.call(rbind, lapply(results_list, function(res) {
    res$roc %>%
      mutate(Scenario = res$scenario_name)
  }))
  
  # Clean the data first
  roc_clean <- roc_combined %>%
    mutate(
      FPR = pmax(0, pmin(1, FPR)),
      TPR = pmax(0, pmin(1, TPR))
    )
  
  ggplot(roc_clean, aes(x = FPR, y = TPR, 
                        color = Method, 
                        linetype = Scenario,
                        group = interaction(Method, Scenario))) +
    geom_line(linewidth = 0.8) +
    geom_abline(slope = 1, intercept = 0, 
                color = "gray50", linetype = "dashed", 
                linewidth = 0.5, alpha = 0.5) +
    scale_color_manual(values = method_colors) +
    # USE coord_cartesian for comparison plots too
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    scale_x_continuous(
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = 0.02)
    ) +
    scale_y_continuous(
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = 0.02)
    ) +
    labs(
      title = "ROC Curves Comparison Across Scenarios",
      subtitle = "Elastic Net: CI vs Scaled Neighborhood Selection Methods",
      x = "False Positive Rate",
      y = "True Positive Rate",
      color = "Method",
      linetype = "Scenario"
    ) +
    theme_islr() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5),
      legend.position = "bottom",
      legend.box = "vertical"
    )
}
# ---------------------------------------------------
plot_power_comparison <- function(results_list) {
  
  power_combined <- do.call(rbind, lapply(results_list, function(res) {
    res$power %>%
      mutate(Scenario = res$scenario_name)
  }))
  
  # Clean the data first
  power_clean <- power_combined %>%
    mutate(
      Threshold = pmax(0, pmin(1, Threshold)),
      Power = pmax(0, pmin(1, Power))
    )
  
  ggplot(power_clean, aes(x = Threshold, y = Power, 
                          color = Method, 
                          linetype = Scenario,
                          group = interaction(Method, Scenario))) +
    geom_line(linewidth = 0.8) +
    scale_color_manual(values = method_colors) +
    # USE coord_cartesian for comparison plots too
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    scale_x_continuous(
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = 0.02)
    ) +
    scale_y_continuous(
      breaks = seq(0, 1, 0.2),
      expand = expansion(mult = 0.02)
    ) +
    labs(
      title = "Power Curves Comparison Across Scenarios",
      subtitle = "Elastic Net: Power vs Selection Threshold (α for CI, γ for SN)",
      x = "Threshold",
      y = "Power (True Positive Rate)",
      color = "Method",
      linetype = "Scenario"
    ) +
    theme_islr() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5),
      legend.position = "bottom",
      legend.box = "vertical"
    )
}
# ----------------------------------------------------



# ==============================================================================================================================
# ==============================================================================================================================
# ------------------------------------------------------------------
# 6. Execution
# ------------------------------------------------------------------


# Set the theme globally
theme_set(theme_islr())

# Run simulations 
set.seed(123)
scenarios = define_scenarios()
results = list()

# for (i in 1:length(scenarios)) {
for (i in 1:1) {
    
  start_time = Sys.time()
  
  cat(sprintf("\n=== Scenario %d/%d: %s ===\n", 
              i, length(scenarios), scenarios[[i]]$name))
  
  results[[i]] = run_scenario_simulation(
    scenario = scenarios[[i]],
    n_sim = 10,                                    # ========================================================
    n_train = 42, #105,                                # ========================================================
    n_total = 60, #150,                                # ========================================================
    alpha_grid = seq(0.01, 1, length.out = 300),   # ========================================================
    gamma_grid = seq(0.01, 1, length.out = 300)    # ========================================================
  )
  
  end_time = Sys.time()
  elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  cat(sprintf("  Completed: %s\n", scenarios[[i]]$name))
  cat(sprintf("  Time: %.2f seconds (%.2f minutes)\n", elapsed, elapsed/60))
}

# Generate professional plots for each scenario
cat("GENERATING PROFESSIONAL PLOTS FOR EACH SCENARIO\n")

for (i in 1:length(results)) {
  cat(sprintf("\n--- Plotting for %s ---\n", results[[i]]$scenario_name))
  
  # 1. ROC plot
  p1 <- plot_roc_single_scenario(results[[i]])
  print(p1)
  
  # 2. Power plot
  p2 <- plot_power_single_scenario(results[[i]])
  print(p2)
  
  # 3. Exclusion counts plot (optional)
  # p3 <- plot_exclusion_counts(results[[i]])
  # print(p3)
  
  # 4. MSE comparison plot (optional)
  # p4 <- plot_mse_comparison(results[[i]])
  # print(p4)
  
  # 5. Performance summary plot (optional)
  # p5 <- plot_performance_summary(results[[i]])
  # print(p5)
}

# Generate comparison plots across scenarios
cat("GENERATING COMPARISON PLOTS ACROSS ALL SCENARIOS\n")

p6 <- plot_roc_comparison(results)
print(p6)

p7 <- plot_power_comparison(results)
print(p7)

# Output tables 
for (i in 1:length(results)) {
  res = results[[i]]
  
  cat("\n")
  cat(strrep("=", 60), "\n")
  cat(res$scenario_name, "\n")
  cat(strrep("=", 60), "\n")
  
  cat("\nTABLE 1: Exclusion Counts (out of", res$n_sim, "simulations)\n")
  print(res$exclusion_table)
  
  cat("\nTABLE 2: MSE Comparison\n")
  print(res$mse_summary)
  
  cat("\nTABLE 3: Performance Statistics\n")
  print(res$performance_stats)
}







































