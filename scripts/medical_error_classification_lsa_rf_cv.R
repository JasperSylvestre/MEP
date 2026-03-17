# Clear the environment
rm(list = ls())

# Loading in packages
library(readr)
library(dplyr)
library(tm)
library(lsa)
library(randomForest)

# Set working directory
setwd("../data")

# Read the medical error data
ds <- read.csv("medical_error_data.csv")

# Preprocess the data
# 1. Create 'TimeSeconds' variable:
ds <- ds %>%
  mutate(TimeSeconds = as.numeric(as.POSIXct(DateTime, format = "%Y-%m-%d %H:%M:%S")))

# 2. Convert remaining variables to factors:
ds <- ds %>%
  mutate(across(c(BroadCat, IDbyRole, Quarter, Month, TXIntent, TXMethod, Status, DevType), as.factor))

# 3. Remove unnecessary variables:
ds <- ds %>%
  select(-c(ID, DateTime))

# Define training data proportion
train.prop <- 0.8

# Calculate training set size based on data rows
train.size <- ceiling(nrow(ds) * train.prop)

# Set the number of cross-validation iterations (k-folds)
k <- 10

# Sparsity limit for potential model tuning
sprse <- 0.8

# Initialize empty vectors to store cross-validation results
rf_lsa_cv  <- rep(NA, k)
rf_cv  <- rep(NA, k)

# Set a random seed for reproducibility
set.seed(123)

# Perform cross-validation (CV) with 10 folds
for (i in 1:k) {
  # Randomly split data into training (80%) and testing (20%) sets
  smpl <- sample(nrow(ds), train.size)
  train_ds <- ds[smpl, ]
  test_ds <- ds[-smpl, ]
  
  # Text preprocessing
  
  # Create text corpora for training and testing sets
  train_corp <- VCorpus(VectorSource(train_ds$Desc))
  test_corp <- VCorpus(VectorSource(test_ds$Desc))
  
  # Create Term Document Matrices (TDMs) with TF-IDF weighting, 
  # including words with 2 or more characters, and removing stopwords
  train_tdm <- TermDocumentMatrix(
    train_corp,
    control = list(
      weighting = function(x)
        weightTfIdf(x, normalize = TRUE),
      wordLengths = c(2, Inf),
      stopwords = TRUE
    )
  )
  
  test_tdm <- TermDocumentMatrix(
    test_corp,
    control = list(
      weighting = function(x)
        weightTfIdf(x, normalize = TRUE),
      wordLengths = c(2, Inf),
      stopwords = TRUE
    )
  )
  
  # Remove sparse terms
  train_tdm_sprse = removeSparseTerms(train_tdm, sprse)
  test_tdm_sprse = removeSparseTerms(test_tdm, sprse)
  
  # Latent Semantic Analysis (LSA)
  
  # Create LSA spaces
  train_lsaSpace <- lsa(train_tdm_sprse)
  test_lsaSpace <- lsa(test_tdm_sprse)
  
  # Adjust dimensions to match for consistency
  vars <- min(length(train_lsaSpace$sk), length(test_lsaSpace$sk))
  train_lsaSpace <- lsa(train_tdm_sprse, dim = vars)
  test_lsaSpace <- lsa(test_tdm_sprse, dim = vars)
  
  # Feature extraction
  train_lsak <- train_lsaSpace$dk
  test_lsak <- test_lsaSpace$dk
  
  # Extract features from LSA spaces
  train_extract <-
    data.frame(train_ds %>%
                 select(-"Desc"),
               scale(train_lsak))
  test_extract <-
    data.frame(test_ds %>%
                 select(-"Desc"),
               scale(test_lsak))

  # Scale 'TimeSeconds' feature using mean and standard deviation from training set
  ts_sd <- sd(train_extract$TimeSeconds)
  ts_mean <- mean(train_extract$TimeSeconds)
  train_extract$TimeSeconds <- (train_extract$TimeSeconds - ts_mean) / ts_sd
  test_extract$TimeSeconds <- (test_extract$TimeSeconds - ts_mean) / ts_sd
  
  # Impute missing values in 'IDbyRole' with the most frequent category
  train_extract$IDbyRole[is.na(train_extract$IDbyRole)] <- levels(train_extract$IDbyRole)[which.max(table(train_extract$IDbyRole))]
  test_extract$IDbyRole[is.na(test_extract$IDbyRole)] <- levels(train_extract$IDbyRole)[which.max(table(train_extract$IDbyRole))]
  
  # Model building and prediction

  # Model with LSA features
  rf_lsa_model <-
    randomForest(BroadCat ~ .,
                 data = train_extract,
                 mtry = 4,
                 ntree = 500)
  rf_lsa_pred <- predict(rf_lsa_model, test_extract)
  
  # Model without LSA features
  rf_model <-
    randomForest(BroadCat ~ .,
                 data = train_extract[, 1:9],
                 mtry = 3,
                 ntree = 500)
  rf_pred <- predict(rf_model, test_extract[, 1:9])
  
  # Test Error (using classification error rate)
  rf_lsa_cv[i] <- mean(rf_lsa_pred != test_ds$BroadCat)
  rf_cv[i] <- mean(rf_pred != test_ds$BroadCat)
}

# Calculate confidence intervals for model results with and without LSA
result_lsa_cv <- t.test(rf_lsa_cv)
result_cv <- t.test(rf_cv)

# Test if model without LSA has lower error (one-tailed)
one_tailed_test <- t.test(rf_lsa_cv, rf_cv, paired = TRUE, alternative = "greater")

# Save combined results
result_data <- data.frame(
  method = c("lsa_cv", "cv", "one_tailed_test"),
  lower_bound = c(result_lsa_cv$conf.int[1], result_cv$conf.int[1], one_tailed_test$conf.int[1]),
  upper_bound = c(result_lsa_cv$conf.int[2], result_cv$conf.int[2], one_tailed_test$conf.int[2]),
  p_value = c(result_lsa_cv$p.value, result_cv$p.value, one_tailed_test$p.value)
)

file_path <- "../results/result_data.csv"
write.csv(result_data, file = file_path, row.names = FALSE)