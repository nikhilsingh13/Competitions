# titanic_data_analysis.R
rm(list=ls())

setwd('C:/Users/NSingh/Nikhil/Personal/Kaggle/titanic/')

## Initializing Libraries ----
pacman::p_load(dataPreparation)
pacman::p_load(tidyverse)
pacman::p_load(mice)
pacman::p_load(VIM)
pacman::p_load(ranger)
pacman::p_load(xgboost)
pacman::p_load(caret)

## Loading and understanding data ----
train_data <- read_csv('data/train.csv')

train_data %>% head
train_data %>% dim                                     # 891 X 12
train_data$PassengerId %>% unique %>% length           # 891

description(train_data)

## cleaning

### filtering dataset
train_data %>% whichAreConstant
train_data %>% whichAreBijection  # Name and PassengerId are bijections, will drop 1
train_data %>% whichAreInDouble

### cleaning data manually
semi_cleaned_data <- train_data %>% 
  mutate_at(c("Survived", "Pclass", "Sex", "Embarked", "SibSp", "Parch")
            , factor) %>% 
  mutate(count=1) %>% 
  spread(Sex, count, fill=0, sep="_") %>% 
  drop_na(Embarked) %>% 
  mutate(count=1) %>% 
  spread(Embarked, count, fill=0, sep="_") %>% 
  # mutate(scaled_Fare = scale(Fare)) %>% 
  select(-c(Name, Sex_male, Cabin, Ticket))
  # %>% map_df(~sum(is.na(.)))
  # is.na %>% colSums
  # ?fastHandleNa %>% 
  # mutate(count=1) %>% 
  # spread(Embarked, count, fill=0, sep="_") %>% 
  # head

k %>% head
table(k$Sex_female, k$Parch)
table(k$Embarked) %>% as.integer %>% sum

table(k$Age)%>% as.integer %>% sum

### analysing column with NA values in age
md.pattern(k)

mice_plot <- aggr(semi_cleaned_data, col=c('navyblue','yellow')
                  , numbers  = TRUE
                  , sortVars = TRUE
                  , labels   = names(k)
                  , cex.axis = 0.7
                  , gap      = 3
                  , ylab     = c('Missing data', 'Pattern'))

imputed_data <- mice(semi_cleaned_data
                     , m      = 5
                     , maxit  = 50
                     , method = "pmm"
                     , seed   = 500)

imputed_data %>% summary

imputed_data %>% head
imputed_data$imp$Age

semi_cleaned_data_imputed <- complete(imputed_data,2)

completeData %>% head
k %>% head

completeData %>% is.na %>% colSums

completeData$Fare %>% summary

semi_cleaned_data_imputed %>% head
semi_cleaned_data_imputed %>% is.na %>% colSums


cleaned_data_imputed <- semi_cleaned_data_imputed %>% 
  mutate(scaled_fare = scale(Fare)[,1]
         , scaled_age = scale(Age)[,1]) %>% 
  select(-c(Fare, Age))

cleaned_data_imputed %>% head
cleaned_data_imputed %>% glimpse

# Splitting training and validation dataset

training_data_index <- sample(1:nrow(cleaned_data_imputed)
                              , 0.8*nrow(cleaned_data_imputed))
testing_data_index <- setdiff(1:nrow(cleaned_data_imputed)
                              , training_data_index)

training_data_index %>% length # 711
testing_data_index %>% length  # 178

training_data <- cleaned_data_imputed[training_data_index,]
testing_data <- cleaned_data_imputed[testing_data_index,]

training_data %>% dim
testing_data %>% dim

## Ranger
ranger_model <- ranger(Survived~., data=training_data)
prediction <- predict(ranger_model, data = testing_data)
confusionMatrix(testing_data$Survived, prediction$predictions)

## XGboost

training_data_df <- data.table(training_data, keep.rownames = FALSE)
testing_data_df <- data.table(testing_data, keep.rownames = FALSE)

sparse_matrix <- sparse.model.matrix(Survived~., data = training_data_df)[,-1]
sparse_matrix %>% head
sparse_matrix %>% dim

output_vector = training_data_df[,Survived] == "1"
output_vector %>% length

xgb_model <- xgboost(data=sparse_matrix
                     , label = output_vector
                     , eta = 1#0.1
                     , max_depth = 4#15
                     , nround = 10
                     # , subsample = 0.5
                     # , colsample_bytree = 0.5
                     , seed = 123
                     , eval_metric = "auc"
                     , objective = "binary:logistic"
                     # , num_class = 2
                     , nthread = 2
                     , verbose = 1)

importance <- xgb.importance(feature_names = colnames(sparse_matrix)
                             , model = xgb_model)

sparse_matrix_test <- sparse.model.matrix(Survived~., data = testing_data_df)[,-1]
sparse_matrix_test %>% dim

# output_test_vector = testing_data_df[, Survived] == "1"

test_prob <- predict(xgb_model, sparse_matrix_test)
test_pred = as.numeric(test_prob>=0.5)

confusionMatrix(testing_data_df$Survived, as.factor(test_pred))

### Reference:
#### https://cran.r-project.org/web/packages/xgboost/vignettes/discoverYourData.html
