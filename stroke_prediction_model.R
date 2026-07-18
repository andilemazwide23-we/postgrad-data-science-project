# 0. Load Libraries --
library(dplyr) # for data manipulation and preprocessing
library(caTools) # for splitting into train and test sets
library(caret) # used for performance metric functions
library(pROC) # used for obtaining AUC
library(h2o)

# for fitting  DT in rprart
library(rpart)
library(rpart.plot)
# the packages used for balancing
library(ROSE)
library(mltools)
# Load library used for SMOTE
library(UBL)
options(scipen = 999) 

# 1. Setup & User Parameters ---------------------------------------------

# User-specified parameters
seed = 456     # Seed for reproducibility (students can change)
train_frac <- 0.7  # Proportion of data in training set
metric <- "F1" # "auc", "aucpr" (Area Under Precision–Recall Curve), "logloss", "Accuracy", "Specificity", "Precision", "Recall", "F1" 
folds <- 5 # for 5-fold CV, or change to 10

  
library(readr)
Stroke_predictions <- read_csv("Stroke_predictions.csv",show_col_types = FALSE)
View(Stroke_predictions)

Stroke_predictions <- data.frame(Stroke_predictions)
summary(Stroke_predictions)


Stroke_predictions$Gender_encoded <- factor(Stroke_predictions$Gender_encoded)
Stroke_predictions$hypertension<- factor(Stroke_predictions$hypertension)
Stroke_predictions$heart_disease <- factor(Stroke_predictions$heart_disease )
Stroke_predictions$Evermarried_encoded <- factor(Stroke_predictions$Evermarried_encoded)
Stroke_predictions$Residence_encoded <- factor(Stroke_predictions$Residence_encoded )
Stroke_predictions$formerly.smoked <- factor(Stroke_predictions$formerly.smoked)
Stroke_predictions$smokes <- factor(Stroke_predictions$smokes)
Stroke_predictions$Unknown <- factor(Stroke_predictions$Unknown) # this is actually the target
Stroke_predictions$stroke<- factor(Stroke_predictions$stroke)

summary(Stroke_predictions)

sapply(Filter(is.factor, Stroke_predictions), nlevels)
round(prop.table(table(Stroke_predictions$stroke))*100,2)

df <- Stroke_predictions
target <- "stroke"
summary(df[[target]])

 Train/Test Split --------------------------------------------------------- 
 
  
set.seed(seed)  
split=sample.split(df[[target]],SplitRatio = train_frac) # train_frac was specified under the setup above 

training_set=subset(df,split==TRUE) 
test_set=subset(df,split==FALSE)

summary(training_set[[target]]) # check the class imbalance in the training set

##################### Under-sampling ###############################

total_under <- nrow(training_set[training_set[[target]] == "1", ])

train_under <- ovun.sample(
  as.formula(paste(target, "~ .")), # this formula specifies what the target is and the predictors
  data = training_set,
  method = "under",
  N = 2 * total_under, # new total = multiply by 2 for the two classes
  seed = seed
)

# Extract and save the resulting under-sampled data:
train_under_data <- train_under$data

summary(train_under_data[[target]])

############################## Over-sampling ##############################################

total_over <- nrow(training_set[training_set[[target]] == "0", ])

train_over <- ovun.sample(
  as.formula(paste(target, "~ .")),
  data = training_set,
  method = "over",
  N = 2 * total_over, # multiply by 2 for the twp classes
  seed = seed
)

train_over_data <- train_over$data
summary(train_over_data[[target]])

###################### Combination of over and under #################################

total_both <- nrow(training_set) # specify the total sample size after the procedure, this can be changed to any value
fraction_new <- 0.50 # specify the approx proportion of minority cases to be produced

train_both <- ovun.sample(
  as.formula(paste(target, "~ .")),
  data = training_set,
  method = "both",
  N = total_both,
  p = fraction_new,
  seed = seed
)

# Extract and save the resulting data (list):
train_both_data <- train_both$data
summary(train_both_data[[target]])

####################################### SMOTE ##################################

set.seed(seed)
train_smote_data <- SmoteClassif(
  as.formula(paste(target, "~ .")),
  training_set,
  C.perc = "balance", # minority and majority classes 
  k = 5, # number of nearest neighbours,
  dist = "HVDM"
)

summary(train_smote_data[[target]])

h2o.init()

predictors <- setdiff(names(training_set), target)

# ----------------------------------------------------------------------------- #
# 9.Repeatedly fit the LR model ----------------------------------------------
# ----------------------------------------------------------------------------- #

Performance_comparison <- data.frame()

train_set_name <- "train_smote_data"

training_set_final <- get(train_set_name)


balanced_train_set_h2o <- as.h2o(training_set_final)
test_h2o <- as.h2o(test_set) 

#Fit the logistic regression model
#LR <- h2o.glm(
 # x = predictors,
 # y = target,
 # training_frame = balanced_train_set_h2o,
 #family = "binomial", # logistic regression
  #lambda = 0, # no regularization (like classical GLM)
 # compute_p_values = TRUE # optional: get p-values
#)

########## --> Extract predicted probabilities ----

# Save predicted probabilities
#preds_LR_train <- h2o.predict(LR, balanced_train_set_h2o)
#preds_LR_test <- h2o.predict(LR, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
#preds_LR_train <- as.data.frame(preds_LR_train)
#preds_LR_test <- as.data.frame(preds_LR_test)

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets:

#train_LR_pred <- cbind(training_set_final,
                     #  setNames(preds_LR_train[, 3, drop = FALSE], "pred_prob"))  

#test_LR_pred <- cbind(test_set,
                     # setNames(preds_LR_test[, 3, drop = FALSE], "pred_prob")) 


######### Extract and append results to a data frame ############

######### REPEAT THESE STEPS FOR ALL BALANCED DATA SETS ######################### 

# Step 1: Extract metrics from confusion matrices

# Training AUC
#roc_LR_train <- roc(train_LR_pred[[target]], train_LR_pred$pred_prob)
#auc_train <- auc(roc_LR_train)

# Test AUC
#roc_LR_test <- roc(test_LR_pred[[target]], test_LR_pred$pred_prob)
#auc_test <- auc(roc_LR_test)

# Step 2: Create individual rows for train and test
#performance_train <- data.frame(
 # technique = train_set_name,
 # dataset = "train",
 # auc = auc_train
#)

#performance_test <- data.frame(
  #technique = train_set_name,
 # dataset = "test",
 #auc = auc_test
#)

# Step 3: Append into one data frame (joins results from previous models each time)
#Performance_comparison <- rbind(Performance_comparison,performance_train, performance_test, make.row.names = FALSE)


# Step 4: View results
#View(Performance_comparison)
#print(Performance_comparison)




#7. Fit Naive Bayes Classifier -----------------------------------------------
  
  ########## --> Fit the model ----

nb <- h2o.naiveBayes(
  x = predictors,
  y = target,
  training_frame = balanced_train_set_h2o,
  laplace = 0, # a smoothing parameter for categories with 0 observations to avoid zero probabilities
  nfolds = folds, # this is based on 5 or 10 folds as specified in the setup (section 1)
  seed = seed # based on the seed set
)

h2o.performance(nb)

# we will extract the predicted probabilities of class label = 1, append it to the original training and test sets to determine the predicted class for each based on the threshold and then create our confusion matrix and obtain the performance measures:

########## --> Extract predicted probabilities ----

# Save predicted probabilities
preds_nb_train <- h2o.predict(nb, balanced_train_set_h2o)
preds_nb_test <- h2o.predict(nb, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_nb_train <- as.data.frame(preds_nb_train)
preds_nb_test <- as.data.frame(preds_nb_test)

# view the extracted predictions to see what we actually obtained:

View(preds_nb_train)

# Column 3 contained the predicted probabilities for out positive class (class 1).

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets:

train_nb_pred <- cbind(training_set_final,
                       setNames(preds_nb_train[, 3, drop = FALSE], "pred_prob")) # extract the predicted probabilities in column 3 of preds_nb_train, combine it with the original training set and call the column "pred_prob", this all is saved in a new dataframe called train_nb_pred

test_nb_pred <- cbind(test_set, 
                      setNames(preds_nb_test[, 3, drop = FALSE], "pred_prob")) # the same is done with the test set

# view the above:

View(train_nb_pred)

########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# This following converts predicted probabilities into class labels by applying a threshold: observations with predicted probability above the threshold are classified as “1” (the positive class), and those below as “0”, with the result stored as a factor.

# training
train_nb_pred$pred_class <- factor(ifelse(train_nb_pred$pred_prob > threshold,"1","0"))

# test
test_nb_pred$pred_class <- factor(ifelse(test_nb_pred$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set

# predicted classes first then actual classes for the training set
metrics_train <-confusionMatrix(
  train_nb_pred$pred_class,
  train_nb_pred[[target]],
  positive = "1",
  mode = "everything"
)
mcc_train <- mcc(train_nb_pred$pred_class,  train_nb_pred[[target]])

# actual classes first then predicted probabilities for the training set
roc_nb_train <- pROC::roc(train_nb_pred[[target]], train_nb_pred$pred_prob)
pROC::auc(roc_nb_train)
plot(roc_nb_train)


# test set

# predicted classes first then actual classes for the test set
metrics_test <- confusionMatrix(
  test_nb_pred$pred_class,
  test_nb_pred[[target]],
  positive = "1",
  mode = "everything"
)
mcc_test <- mcc(test_nb_pred$pred_class,  test_nb_pred[[target]])

metrics_combinednb <- rbind(train = c(as.list(metrics_train$byClass),MCC = mcc_train),  
                          test = c(as.list(metrics_test$byClass),MCC = mcc_test))
#print(metric_vombined)
# or save as a CSV file

#write.csv(metrics_combined, "model_performance_metrics.csv", row.names = TRUE)
# actual classes first then predicted probabilities for the test set
roc_nb_test <- pROC::roc(test_nb_pred[[target]], test_nb_pred$pred_prob)
pROC::auc(roc_nb_test)
plot(roc_nb_test)

#8. Fit a decision tree using h2o --------------------------------------------
  
  ########## --> Hyperparameter tuning ----

# we will start by tuning the hyperparameter. The possible hyperparameters are:

# max_depth: Maximum depth of the tree
# min_rows: Minimum number of observations in a leaf
# min_split_improvement: Minimum reduction in error required to make a split


# Set up the hyperparameter search space (we will do an example for max_depth and min_rows)
hyper_params <- list(
  max_depth = seq(3, 21, by = 2), # from 3 to 21 in increments of 2
  min_rows = c(1, 5, 10, 20, 50)
)

# Define search criteria:
search_criteria <- list(
  strategy = "Cartesian" # Try "RandomDiscrete" for random search
)


h2o.rm("dtree_grid")

# Run the grid search using a single decision tree, GBM (gradient boosting method) with ntrees = 1)
grid <- h2o.grid(
  algorithm = "gbm",
  grid_id = "dtree_grid", # this is just the ID we are giving the grid search
  x = predictors,
  y = target,
  training_frame = balanced_train_set_h2o,
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  ntrees = 1,
  learn_rate = 1.0, # Full weight per tree (since it's only one)
  sample_rate = 1.0,  # use 100% of the training data
  col_sample_rate = 1.0, # use 100% of the attributes
  stopping_rounds = 0, # no stopping rounds required as only 1 tree is being grown
  seed = seed,
  nfolds =folds
)


model_results_dt <- h2o.getGrid("dtree_grid", sort_by = metric, decreasing = TRUE) # arrange metric in descending order so that the first model in this object has the best CV performance 



print(model_results_dt)

# Extract the best model ID _which is in the first row of model_results_dt
best_model_id <- model_results_dt@model_ids[[1]]

# Retrieve hyperparameter values associated with the best model
best_model <- h2o.getModel(best_model_id)

# Step 1: Automatically identify which hyperparameters were tuned
# (this pulls the names directly from the hyper_params list — no hardcoding)
tuned_param_names <- names(hyper_params)

# Step 2: Extract the actual tuned values that were used in the best model
# (best_model@parameters stores everything that was actually applied)
best_tuned_values <- lapply(tuned_param_names, function(param_name) { 
  best_model@allparameters[[param_name]]
})

# append the hyperparameter names to the tuned values:
names(best_tuned_values) <- tuned_param_names

########## --> Fit the model ----

final_dt_model <- do.call(h2o.decision_tree, c(
  list(
    x = predictors,
    y = target,
    training_frame = balanced_train_set_h2o,
    seed = seed                       # the seed
  ),
  best_tuned_values     # ← automatically inserts max_depth, min_rows, etc.
))

########## --> Extract predicted probabilities ----

# Save predicted probabilities
preds_dt_train <- h2o.predict(final_dt_model, balanced_train_set_h2o)
preds_dt_test <- h2o.predict(final_dt_model, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_dt_train <- as.data.frame(preds_dt_train)
preds_dt_test <- as.data.frame(preds_dt_test)

#the structure of the predicted output is the same as that from the Naive Bayes:
View(preds_dt_train)

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets in the exact same manner as that for the Naive bayes:

train_dt_pred <- cbind(training_set_final,
                       setNames(preds_dt_train[, 3, drop = FALSE], "pred_prob")) 

test_dt_pred <- cbind(test_set, 
                      setNames(preds_dt_test[, 3, drop = FALSE], "pred_prob"))  

# view the above:

View(train_dt_pred)

########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# training
train_dt_pred$pred_class <- factor(ifelse(train_dt_pred$pred_prob > threshold,"1","0"))

# test
test_dt_pred$pred_class <- factor(ifelse(test_dt_pred$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set

# predicted classes first then actual classes for the training set
metrics_train <-confusionMatrix(
  train_dt_pred$pred_class,
  train_dt_pred[[target]],
  positive = "1",
  mode = "everything"
)
mcc_train <- mcc(train_dt_pred$pred_class,  train_dt_pred[[target]])
# actual classes first then predicted probabilities for the training set
roc_dt_train <- pROC::roc(train_dt_pred[[target]], train_dt_pred$pred_prob)
pROC::auc(roc_dt_train)
plot(roc_dt_train)


# test set

# predicted classes first then actual classes for the test set
metrics_test <- confusionMatrix(
  test_dt_pred$pred_class,
  test_dt_pred[[target]],
  positive = "1",
  mode = "everything"
)

mcc_test <- mcc(test_dt_pred$pred_class, test_dt_pred[[target]])
metrics_combineddt <- rbind(train = c(as.list(metrics_train$byClass),MCC = mcc_train),  
                          test = c(as.list(metrics_test$byClass),MCC = mcc_test))

#write.csv(metrics_combined, "model_performance_metrics.csv", row.names = TRUE)

# actual classes first then predicted probabilities for the test set
roc_dt_test <- pROC::roc(test_dt_pred[[target]], test_dt_pred$pred_prob)
pROC::auc(roc_dt_test)
plot(roc_dt_test)



# ----------------------------------------------------------------------------- #
# 9. Fit a decision tree using rpart --------------------------------------------

set.seed(seed)


DT_rpart <- rpart(
  as.formula(paste(target, "~ .")),
  data = training_set_final,
  method = "class", # for classification
  xval = folds, # CV 
  control = rpart.control(
    #cp = 0.02,             # complexity parameter for pruning
    #minsplit = 20,         # minimum observations to attempt a split
    maxdepth = 4           # maximum depth
  )
) # default attribute selection measure is Gini Index

DT_rpart # run this to see information about the fitted tree

# rpart automatically searches over values for cp to be tuned.

# rpart grows a large tree first (with a very small cp), then considers a whole sequence of smaller sub-trees by increasing the effective cp.

# --------------------------------------------  #
# While the tree is being grown, a samll cp acts as a stopping rule:

# A split is only made if it improves the model fit by at least cp amount.
# More precisely: the reduction in impurity must exceed cp.

# So if cp is large → fewer splits → smaller tree
# If cp is very small → tree grows much deeper

# This is pre-pruning (early stopping).
# -------------------------------------------- #
# After growing a large tree, rpart also uses cp in a cost-complexity pruning framework:

# It computes a sequence of nested subtrees
# Each subtree corresponds to a different cp value
# These are stored in the complexity parameter table (cptable)
# -------------------------------------------- #

printcp(DT_rpart)
plotcp(DT_rpart)


########## --> Extract predicted probabilities ----


pred_prob_DT_train <- predict(DT_rpart, newdata = training_set_final, type = "prob")

train_DT_rpart <- cbind(training_set_final, 
                        setNames(data.frame(pred_prob_DT_train[, 2]), "pred_prob")) # We only want the probs in column 2 (for "1")

# View the results:

View(train_DT_rpart)


pred_prob_DT_test <- predict(DT_rpart, newdata = test_set, type = "prob")

test_DT_rpart <- cbind(test_set, 
                       setNames(data.frame(pred_prob_DT_test[, 2]), "pred_prob")) # We only want the probs in column 2 (for "1")

########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# training set
train_DT_rpart$pred_class <- factor(ifelse(train_DT_rpart$pred_prob > threshold, "1","0"))

# test
test_DT_rpart$pred_class <- factor(ifelse(test_DT_rpart$pred_prob > threshold,"1","0"))

########## --> Obtain confusion matrix and model performance ----

# training set 

# predicted classes first then actual classes for the training set
metrics_train<-confusionMatrix(
  train_DT_rpart$pred_class,
  train_DT_rpart[[target]],
  positive = "1",
  mode = "everything"
)

mcc_train <- mcc( train_DT_rpart$pred_class, train_DT_rpart[[target]])

# actual classes first then predicted probabilities for the training set
roc_DT_train_rpart <- pROC::roc(train_DT_rpart[[target]], train_DT_rpart$pred_prob)
pROC::auc(roc_DT_train_rpart)
plot(roc_DT_train_rpart)

# test set

# predicted classes first then actual classes for the test set
metrics_test<-confusionMatrix(
  test_DT_rpart$pred_class,
  test_DT_rpart[[target]],
  positive = "1",
  mode = "everything"
)

mcc_test <- mcc(test_DT_rpart$pred_class,  test_DT_rpart[[target]])
metrics_combinedDT_rpart <- rbind(train = c(as.list(metrics_train$byClass),MCC = mcc_train),  
                          test = c(as.list(metrics_test$byClass),MCC = mcc_test))

#write.csv(metrics_combined, "model_performance_metrics.csv", row.names = TRUE)
# actual classes first then predicted probabilities for the test set
roc_DT_test_rpart <- pROC::roc(test_DT_rpart[[target]], test_DT_rpart$pred_prob)
pROC::auc(roc_DT_test_rpart)
plot(roc_DT_test_rpart)


########## --> visualize the DT ----

dev.new(width = 15, height = 20) # This just allows the plot to be shown in a separate window (useful for small screens)

rpart.plot(DT_rpart)
rpart.plot(DT_rpart, yesno = 1, type = 2, fallen.leaves = FALSE) # add additional options to change the appearance.
# see http://www.milbo.org/rpart-plot/prp.pdf for more options to customize the plot



# ----------------------------------------------------------------------------- #
# 10. Fit a logistic regression model --------------------------------------------
# ----------------------------------------------------------------------------- #

########## --> Fit the LR model in H2O ----

# Fit the logistic regression model
LR <- h2o.glm(
  x = predictors,
  y = target,
  training_frame = balanced_train_set_h2o,
  family = "binomial", # logistic regression
  lambda = 0, # no regularization (like classical GLM)
  compute_p_values = TRUE # optional: get p-values
)

########## --> Perform inference using the LR model ----

# extract p-values for inference and save into a df called LR_results

LR_results <- LR@model[["coefficients_table"]]

# create odds ratios from the regression coefficient estimates
LR_results$OR <- exp(LR_results[,2])

# round the p-values off to 4 decimal places
LR_results$p_value <- round(LR_results$p_value,4)
LR_results$OR <- round(LR_results$OR,4)

View(LR_results)

########## --> Extract predicted probabilities ----

# Save predicted probabilities
preds_LR_train <- h2o.predict(LR, balanced_train_set_h2o)
preds_LR_test <- h2o.predict(LR, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_LR_train <- as.data.frame(preds_LR_train)
preds_LR_test <- as.data.frame(preds_LR_test)

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets:

train_LR_pred <- cbind(training_set_final,
                       setNames(preds_LR_train[, 3, drop = FALSE], "pred_prob"))  

test_LR_pred <- cbind(test_set,
                      setNames(preds_LR_test[, 3, drop = FALSE], "pred_prob")) 



########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# training
train_LR_pred$pred_class <- factor(ifelse(train_LR_pred$pred_prob > threshold,"1","0"))

# test
test_LR_pred$pred_class <- factor(ifelse(test_LR_pred$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set 

# predicted classes first then actual classes for the training set
metrics_train <-confusionMatrix(
  train_LR_pred$pred_class,
  train_LR_pred[[target]],
  positive = "1",
  mode = "everything"
)
mcc_train <- mcc( train_LR_pred$pred_class,   train_LR_pred[[target]])

# actual classes first then predicted probabilities
roc_LR_train <- pROC::roc(train_LR_pred[[target]], train_LR_pred$pred_prob)
pROC::auc(roc_LR_train)
plot(roc_LR_train)

# test

# predicted classes first then actual classes
metrics_test <- confusionMatrix(
  test_LR_pred$pred_class,
  test_LR_pred[[target]],
  positive = "1",
  mode = "everything"
)

mcc_test <- mcc(test_LR_pred$pred_class,  test_LR_pred[[target]])
metrics_combinedLR <- rbind(train = c(as.list(metrics_train$byClass),MCC = mcc_train),  
                          test = c(as.list(metrics_test$byClass),MCC = mcc_test))

all_models <- rbind(metrics_combinednb,metrics_combineddt,metrics_combinedDT_rpart,metrics_combinedLR)
#print (all_models)
write.csv(all_models, "model_performance_metrics.csv", row.names = TRUE)
# actual classes first then predicted probabilities
roc_LR_test <- pROC::roc(test_LR_pred[[target]], test_LR_pred$pred_prob)
pROC::auc(roc_LR_test)
plot(roc_LR_test)


#################### Combine ROC curves of test set for all models ############

# Plot (see https://r-charts.com/colors/ for more colours)
plot(
  roc_nb_test,
  col = "#458B74",
  lwd = 2,
  main = "ROC Curve Comparison of test set for NB, DT and LR"
)
lines(roc_DT_test_rpart, col = "#0000FF", lwd = 2)
lines(roc_dt_test, col = "#CD3333", lwd = 2)
lines(roc_LR_test, col = "#009ACD", lwd = 2)

# Add legend
legend(
  "bottomright",
  legend = c("Naive Bayes", "Decision tree", "Logistic regression","Decision tree rpart" ),
  col = c("#458B74", "#CD3333", "#009ACD", "#0000FF"),
  lwd = 2
)


h2o.shutdown(prompt = FALSE)
