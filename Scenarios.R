
# ==============================================================================
# SCENARIO DEFINITIONS FOR BAYESIAN METHODS COMPARISON
# ==============================================================================
# This file defines 5 scenarios for comparing Bayesian Ridge, Lasso, and Elastic Net
# Each scenario is designed to showcase when each method excels
# ==============================================================================

# Load required package for multivariate normal generation
library(MASS)

define_scenarios = function() {
  scenarios = list()
  
  # -----------------------------------------------------------------
  # SCENARIO 1: Extreme Multicollinearity Challenge
  # -----------------------------------------------------------------
  # scenarios[[1]] = list(
  #   name = "Scenario 1: Dense Signals with extreme multicollinearity",
  #   description = "All predictors highly correlated (ρ=0.95), all coefficients equal",
  #   p = 10,                           
  #   beta = rep(0.90, 10),              # All coefficients = 0.90
  #   sigma = 0.5,                     
  #   data_gen = function(n, p, beta, sigma) {
  #     # Extreme correlation: ρ = 0.95 for all pairs
  #     rho = 0.95
  #     Sigma = matrix(rho, p, p)
  #     diag(Sigma) = 1
  #     
  #     # Generate from multivariate normal
  #     X = mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
  #     X = scale(X)
  #     y = as.numeric(X %*% beta + rnorm(n, 0, sigma))
  #     
  #     list(X = X, y = y)
  #   },
  #   expected_winner = "RIDGE",
  #   why = "Extreme correlation requires pure ℓ₂ penalty for stability. LASSO randomly selects one variable, EN compromised by ℓ₁ component."
  # )
  # -----------------------------------------------------------------
  # 
  # -----------------------------------------------------------------
  scenarios[[1]] = list(
    name = "Pollution Data Scenario",
    description = "Real Pollution dataset",
    p = ncol(pollution) - 1,
    beta = rep(0,15),
    sigma = NA,                  
    data_gen = function(n = NULL, p = NULL, beta = NULL, sigma = NULL) {
      X = scale(pollution)[, -16]
      y = scale(pollution)[, 16]
      list(X = X, y = y)
    },
    expected_winner = "To check",
    why = "To check"
  )
  
  
  # -----------------------------------------------------------------
  # SCENARIO 2: Independent Sparse Model 
  # -----------------------------------------------------------------
  scenarios[[2]] = list(
    name = "Scenario 2: Independent Sparse Model",
    description = "All predictors independent, sparse true model",
    p = 10,                          
    beta = c(0.8, 1.5, -3, 6, rep(0, 6)),
    sigma = 2,                       
    data_gen = function(n, p, beta, sigma) {
      # Independent predictors
      X = matrix(rnorm(n * p), n, p)
      X = scale(X)
      y = as.numeric(X %*% beta + rnorm(n, 0, sigma))
      
      list(X = X, y = y)
    },
    expected_winner = "LASSO",
    why = "Independent predictors and sparse model favor pure ℓ₁ penalty. Ridge includes all variables, EN adds unnecessary ℓ₂ penalty."
  )
  
  # -----------------------------------------------------------------
  # SCENARIO 3: Correlated Groups with Strong Signals
  # -----------------------------------------------------------------
  scenarios[[3]] = list(
    name = "Scenario 3: Correlated Groups with Strong Signals",
    description = "Three groups of correlated predictors, first two groups have signals",
    p = 30,                        
    beta = c(rep(3, 5), rep(3,5), rep(3, 5), rep(0, 15)),
    sigma = 3,                       
    data_gen = function(n, p, beta, sigma) {
      # Create 3 groups of 10 with within-group correlation
      X = matrix(0, n, p)
      
      # Group 1: variables 1-5, correlated via Z1
      Z1 = rnorm(n)
      for (i in 1:5) {
        X[, i] = Z1 + rnorm(n, 0, 0.01)
      }
      
      # Group 2: variables 6-10, correlated via Z2
      Z2 = rnorm(n)
      for (i in 6:10) {
        X[, i] = Z2 + rnorm(n, 0, 0.01)
      }
      
      # Group 3: variables 11-15, correlated via Z3
      Z3 = rnorm(n)
      for (i in 11:15) {
        X[, i] = Z3 + rnorm(n, 0, 0.01)
      }
      # Group 4 : variables 16-30, uncorrelated
      for(i in 16:30){
        X[,i] = rnorm(n,0,1)
      }
      
      X = scale(X)
      y = as.numeric(X %*% beta + rnorm(n, 0, sigma))
      
      list(X = X, y = y)
    },
    expected_winner = "ELASTIC NET",
    why = "Correlated groups favor EN's grouping effect. LASSO picks one from each group arbitrarily, Ridge includes all noise variables."
  )
  
  # -----------------------------------------------------------------
  # SCENARIO 4: Uniform Correlation with Block Sparsity
  # -----------------------------------------------------------------
  scenarios[[4]] = list(
    name = "Scenario 4: Uniform Correlation with Block Structure",
    description = "Moderate correlation (ρ=0.3) among all predictors, two blocks of signals",
    p = 50,                        
    beta = c(rep(0, 10), rep(2, 10), rep(0, 10), rep(2, 10), rep(0, 10)),
    sigma = 5,                      
    data_gen = function(n, p, beta, sigma) {
      # Moderate correlation among all predictors (ρ = 0.3)
      rho = 0.3
      Sigma = matrix(rho, p, p)
      diag(Sigma) = 1
      
      X = mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
      X = scale(X)
      y = as.numeric(X %*% beta + rnorm(n, 0, sigma))
      
      list(X = X, y = y)
    },
    expected_winner = "ELASTIC NET",
    why = "Uniform correlation with block sparsity challenges LASSO's random selection. EN's ℓ₂ component helps handle correlation while ℓ₁ gives sparsity."
  )
  
  # -----------------------------------------------------------------
  # SCENARIO 5: High-Dimensional with Mixed Signal Strengths
  # -----------------------------------------------------------------
  scenarios[[5]] = list(
    name = "Scenario 5: High-Dimensional data with Mixed Signals",
    description = "p=200 >> n=150, mixed signal strengths (strong/medium/weak)",
    p = 200,                        
    beta = c(rep(5, 5), rep(4, 5), rep(3, 5), rep(0, 185)),
    sigma = 1.5,                      
    data_gen = function(n, p, beta, sigma) {
      # Moderate correlation (ρ = 0.3) among all predictors
      rho = 0.2
      Sigma = matrix(rho, p, p)
      diag(Sigma) = 1
      
      X = mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
      X = scale(X)
      y = as.numeric(X %*% beta + rnorm(n, 0, sigma))
      
      list(X = X, y = y)
    },
    expected_winner = "ELASTIC NET or LASSO",
    why = "High-dimensional sparse setting needs sparsity (LASSO/EN). EN may have edge if correlation affects selection. Ridge fails with p >> n."
  )
  
  return(scenarios)
}

