# titanic_analysis_model.R

rm(list=ls())
setwd('C:/Users/NSingh/Nikhil/Personal/Kaggle/titanic/')

# remotes::install_github("mlr-org/mlr3")
pacman::p_load(mlr3)
pacman::p_load(data.table)
pacman::p_load(dplyr)
pacman::p_load(dataPreparation)
pacman::p_load(mice)
pacman::p_load(VIM)
pacman::p_load(mlr3learners)
pacman::p_load(precrec)

getDTthreads()

titanic_data <- fread(input = 'data/train.csv')
titanic_data

titanic_data %>% glimpse

titanic_data %>% is.na %>% colSums

description(titanic_data, level=1)

titanic_data$PassengerId %>% unique %>% length
titanic_data$Survived %>% table
titanic_data$Pclass %>% table
titanic_data$Name %>% unique %>% length
titanic_data$Sex %>% table
titanic_data$Age %>% table %>% as.numeric %>% sum            # 177 NAs
titanic_data$SibSp %>% table
titanic_data$Parch %>% table
titanic_data$Ticket %>% unique %>% length
titanic_data$Fare %>% summary
titanic_data$Embarked %>% table

# Fixing response variable ----
titanic_data$Survived <- as.factor(titanic_data$Survived)   

titanic_data %>% glimpse

# NA imputation ----

## handling NAs of column Embarked first
table(titanic_data$Sex, titanic_data$Embarked)
table(titanic_data$Survived, titanic_data$Embarked)
table(titanic_data$Pclass, titanic_data$Embarked)

## simply removing the 2 NA rows of Embarked columns
titanic_data_cln <- titanic_data[Embarked!=""]

## understanding NAs of Age column now
plot(titanic_data$Age)
md.pattern(titanic_data)

aggr(titanic_data
     , col=mdc(1:2)
     , numbers=TRUE
     , sortVars=TRUE
     , labels=names(titanic_data)
     , cex.axis=.7
     , gap=3
     , ylab=c("Proportion of missingness","Missingness Pattern"))

# mice_plot <- aggr(titanic_data
#                   , col=c('navyblue','yellow')
#                   , numbers  = TRUE
#                   , sortVars = TRUE
#                   , labels   = names(titanic_data)
#                   , cex.axis = 0.7
#                   , gap      = 3
#                   , ylab     = c('Missing data', 'Pattern'))

# marginplot(titanic_data[, c("Sex", "Age")], col = mdc(1:2)
#            , cex.numbers = 1.2, pch = 19)

lattice::densityplot(titanic_data$Age)

## performing predictive mean modeling using mice package
imputed_data <- mice(titanic_data_cln
                     , m      = 5
                     , maxit  = 50
                     , method = "pmm"
                     , seed   = 500)

imputed_data %>% str
imputed_data %>% summary
imputed_data %>% head

# imputed_data$blocks %>% str
semi_cleaned_data_imputed <- complete(imputed_data, 1)
semi_cleaned_data_imputed %>% head


# Cleaning the data further by removing a few columns
scd <- semi_cleaned_data_imputed %>% 
  select(-c(Name, Ticket, Cabin, PassengerId)) %>% 
  mutate_at(c("Pclass","Sex", "SibSp", "Parch", "Embarked"), factor)


scd %>% head
scd %>% glimpse

scd$Survived %>% table
scd$Fare %>% 
  hist

# Constructing Learners and Tasks
set.seed(12)

## creating learning task
task_scd = TaskClassif$new(id='scd'
                           , backend = scd
                           , target = "Survived")

task_scd
task_scd$feature_names

## train, test split
train_scd <- sample(task_scd$nrow, 0.8*task_scd$nrow)
test_scd <- sample(seq_len(task_scd$nrow), train_scd)

## load learner and set hyperparameter    - rpart
learner = lrn("classif.rpart", cp=0.01)

## train the model
learner$train(task_scd, row_ids = train_scd)

## predict data
prediction = learner$predict(task_scd, row_ids=test_scd)

## performance
prediction$confusion

measure=msr("classif.acc")
prediction$score(measure)

# Resample
## automatic resampling
resampling = rsmp("cv", folds=3L)
rr = resample(task_scd, learner, resampling)

rr$score(measure)

rr$aggregate(measure)

### ### ### ### ### ### ### ### ### ### ### ###
                # RANGER ----------------
### ### ### ### ### ### ### ### ### ### ### ###

## load learner and set hyperparameter    - ranger
learner_ranger = lrn("classif.ranger", predict_type="prob")

## training the model using ranger
learner_ranger$train(task_scd, row_ids = train_scd)

## predicting data using the ranger object we created above
pred_ranger = learner_ranger$predict(task_scd, row_ids=test_scd)

## performance from ranger
pred_ranger$confusion
pred_ranger$score(measure)
rr_ranger = resample(task_scd, learner_ranger, resampling)
rr_ranger$score(measure)
rr_ranger$aggregate(measure)

evaluated_ranger = evalmod(scores = pred_ranger$prob
                           , label = pred_ranger$truth
                           , posclass = task_scd$positive)

## tpr vs FPR / Sensitivity vs (1-Specificity)
ggplot2::autoplot(evaluated_ranger, curvetype="ROC")

## Precision vs Recall
ggplot2::autoplot(evaluated_ranger, curvetype="PRC")

### ### ### ### ### ### ### ### ### ### ### ###
# XGBOOST ----------------
### ### ### ### ### ### ### ### ### ### ### ###
semi_cleaned_data_imputed %>% glimpse
xg_scd <- semi_cleaned_data_imputed %>% 
  select(-c(Name, Ticket, Cabin, PassengerId)) %>% 
  mutate_at(c("Pclass","Sex", "SibSp", "Parch", "Embarked"), numeric)

learner_xgboost = lrn("classif.xgboost", predict_type="prob")
learner_xgboost$param_set$ids()
learner_xgboost$train(task_scd, row_ids = train_scd)