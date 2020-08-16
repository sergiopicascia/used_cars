# Setup
library(tidyverse)
library(caret)
library(car)
library(glmnet)
library(pls)
library(MASS)
library(tree)
library(mltools)
library(data.table)
library(randomForest)
library(ipred)
set.seed(22)

# Import data
df <- read.csv('/Users/sergiopicascia/Desktop/vehicles.csv', na.strings = c('', 'NA', 'missing'))

str(df)
summary(df)
sapply(df, function(x) sum(is.na(x)))

# Removing columns in excess
vehicles <- subset(df, select = -c(id, url, region, region_url, model, image_url, 
                                   description, county, lat, long, vin, size))

# Eliminating 'price' outliers
hist(vehicles$price[vehicles$price < 100000])
vehicles <- vehicles[(vehicles$price >= 1000) & (vehicles$price < 40000), ]

# Eliminating 'year' outliers, dropping NA
hist(vehicles$year)
vehicles <- vehicles[(vehicles$year >= 1980) & (vehicles$year < 2021), ]
vehicles <- vehicles %>% drop_na(year)

# Dropping 'manufacturer' NA, recode as categorical
unique(vehicles$manufacturer)
vehicles <- vehicles %>% drop_na(manufacturer)
vehicles$manufacturer <- as.factor(vehicles$manufacturer)
vehicles <- vehicles[!(vehicles$manufacturer == 'ferrari' | vehicles$manufacturer == 'morgan'), ]

# Dropping 'condition' NA, recode as categorical
unique(vehicles$condition)
vehicles <- vehicles %>% drop_na(condition)
vehicles$condition <- as.factor(vehicles$condition)

# Dropping 'fuel' NA, recode as categorical
unique(vehicles$fuel)
vehicles <- vehicles %>% drop_na(fuel)
vehicles$fuel <- as.factor(vehicles$fuel)

# Eliminating 'odometer' outliers, dropping NA
hist(vehicles$odometer[vehicles$odometer < 500000])
vehicles <- vehicles[(vehicles$odometer < 300000), ]
vehicles <- vehicles %>% drop_na(odometer)

# Recode 'state' as categorical
unique(vehicles$state)
vehicles$state <- as.factor(vehicles$state)

# Dropping 'title_status' NA, recode as categorical
unique(vehicles$title_status)
vehicles <- vehicles %>% drop_na(title_status)
vehicles$title_status <- as.factor(vehicles$title_status)

# Recode 'cylinders' as categorical
unique(vehicles$cylinders)
vehicles <- vehicles %>% drop_na(cylinders)
vehicles$cylinders <- as.factor(vehicles$cylinders)

# Recode 'transmission' as categorical, dropping NA
unique(vehicles$transmission)
vehicles <- vehicles %>% drop_na(transmission)
vehicles$transmission <- as.factor(vehicles$transmission)

# Recode 'drive' as categorical
unique(vehicles$drive)
vehicles <- vehicles %>% drop_na(drive)
vehicles$drive <- as.factor(vehicles$drive)

# Recode 'type' as categorical
unique(vehicles$type)
vehicles <- vehicles %>% drop_na(type)
vehicles$type <- as.factor(vehicles$type)

# Recode 'paint_color' as categorical
unique(vehicles$paint_color)
vehicles <- vehicles %>% drop_na(paint_color)
vehicles$paint_color <- as.factor(vehicles$paint_color)

summary(vehicles)
sapply(vehicles, function(x) sum(is.na(x)))

# Plot vars against price
for (i in 1:length(vehicles[, -1])) {
 plot(vehicles[, i+1], vehicles$price, xlab = colnames(vehicles)[i+1], ylab = 'price', cex.axis = 0.7, las = 2)
}

### Linear Regression
train_samples <- vehicles$price %>% createDataPartition(p = 0.8, list = F)
train_vehicles <- vehicles[train_samples, ]
test_vehicles <- vehicles[-train_samples, ]

lin_model <- lm(data = train_vehicles, price ~.)
summary(lin_model)
car::vif(lin_model)

lin_model_pred <- predict(lin_model, test_vehicles)
caret::R2(lin_model_pred, test_vehicles$price)
RMSE(lin_model_pred, test_vehicles$price)
RMSE(lin_model_pred, test_vehicles$price)/mean(test_vehicles$price)

compare <- cbind (actual = test_vehicles$price, lin_model_pred)
mean(apply(compare, 1, min)/apply(compare, 1, max))

# Scaled variables
train_scaled <- train_vehicles
test_scaled <- test_vehicles
train_scaled[, c(2, 7)] <- scale(train_vehicles[, c(2, 7)], center = T, scale = T)
test_scaled[, c(2, 7)] <- scale(test_vehicles[, c(2, 7)], center = T, scale = T)

scaled_model <- lm(data = train_scaled, price ~.)
summary(scaled_model)

# Log model
train_log <- train_vehicles
train_log[, 1] <- log(train_log[, 1])
test_log <- test_vehicles
test_log[, 1] <- log(test_log[, 1])

train_log2 <- train_vehicles
train_log2[, c(1, 2)] <- log(train_log2[, c(1, 2)])
train_log2[, 7] <- log(train_log2[, 7] + 1)

train_log3 <- train_vehicles
train_log3[, 2] <- log(train_log3[, 2])
train_log3[, 7] <- log(train_log3[, 7] + 1)

log_model <- lm(data = train_log, price ~.) # Log-level
summary(log_model)
car::vif(log_model)

log_model2 <- lm(data = train_log2, price ~.) # Log-log
summary(log_model2)

log_model3 <- lm(data = train_log3, price ~.) # Level-log
summary(log_model3)

log_model_pred <- predict(log_model, test_log)
caret::R2(log_model_pred, test_log$price)
RMSE(log_model_pred, test_log$price)
RMSE(log_model_pred, test_log$price)/mean(test_log$price)

compare_log <- cbind (actual = test_log$price, log_model_pred)
mean(apply(compare_log, 1, min)/apply(compare_log, 1, max))

# Regression Diagnostics
outlierTest(log_model)

qqPlot(log_model)

hist(studres(log_model), freq = F)
xfit <- seq(min(studres(log_model)), max(studres(log_model)), length=40)
yfit <- dnorm(xfit)
lines(xfit, yfit)

# K-fold Cross Validation
cv10_control <- trainControl(method = 'cv', number = 10)
log_model_cv <- train(data = train_log, price ~., method = 'lm', trControl = cv10_control)
summary(log_model_cv)

log_model_cv_pred <- predict(log_model_cv, test_log)
RMSE(log_model_cv_pred, test_log$price)/mean(test_log$price)

# Backward Regression
back_model <- train(data = train_log, price ~., method = 'leapBackward',
                    tuneGrid = data.frame(nvmax = 1:139), trControl = cv10_control)
back_model$results
back_model$bestTune

# Forward Regression
forw_model <- train(data = train_log, price ~., method = 'leapForward',
                    tuneGrid = data.frame(nvmax = 1:139), trControl = cv10_control)
forw_model$results
forw_model$bestTune

# Stepwise Regression
step_model <- train(data = train_log, price ~., method = 'leapSeq',
                    tuneGrid = data.frame(nvmax = 1:139), trControl = cv10_control)
step_model$results
step_model$bestTune

step_model_pred <- predict(step_model, test_log)
RMSE(step_model_pred, test_log$price)/mean(test_log$price)

# Ridge Regression
x <- model.matrix(data = train_log, price ~.)
test_x <- model.matrix(data = test_log, price ~.)
ridge_cv <- cv.glmnet(x, train_log$price, alpha = 0)
lambda <- ridge_cv$lambda.min
plot(ridge_cv)

ridge_model <- glmnet(x, train_log$price, alpha = 0, lambda = lambda)
ridge_pred <- predict(ridge_model, s = lambda, newx = test_x)
RMSE(ridge_pred, test_log$price)/mean(test_log$price)
compare_ridge <- cbind (actual = test_log$price, ridge_pred)
mean(apply(compare_ridge, 1, min)/apply(compare_ridge, 1, max))

# Lasso Regression
lasso_cv <- cv.glmnet(x, train_log$price, alpha = 0)
lambda2 <- lasso_cv$lambda.min
plot(lasso_cv)

lasso_model <- glmnet(x, train_log$price, alpha = 1, lambda = lambda2)
lasso_pred <- predict(lasso_model, s = lambda2, newx = test_x)
RMSE(lasso_pred, test_log$price)/mean(test_log$price)
compare_lasso <- cbind (actual = test_log$price, lasso_pred)
mean(apply(compare_lasso, 1, min)/apply(compare_lasso, 1, max))

# Principal Component Regression
pcr_model <- pcr(log(price) ~., data = train_scaled, validation = 'CV')
summary(pcr_model)
validationplot(pcr_model, val.type = 'MSEP')
validationplot(pcr_model, val.type = 'R2')
predplot(pcr_model)

test_scaled_log <- test_scaled
test_scaled_log[, 1] <- log(test_scaled_log[, 1])

pcr_pred <- predict(pcr_model, test_scaled_log, ncomp = 3)
RMSE(pcr_pred, test_scaled_log$price)
RMSE(pcr_pred, test_scaled_log$price)/mean(test_scaled_log$price)

compare_pcr <- cbind(actual = test_scaled_log$price, pcr_pred)
mean(apply(compare_pcr, 1, min)/apply(compare_pcr, 1, max))

coefplot(pcr_model, ncomp = c(1, 2, 3))

# Partial Least Squares
pls_model <- plsr(log(price) ~., data = train_scaled, validation = 'CV')
summary(pls_model)
validationplot(pls_model, val.type = 'MSEP')
validationplot(pls_model, val.type = 'R2')
predplot(pls_model)

pls_pred <- predict(pls_model, test_scaled_log, ncomp = 3)
RMSE(pls_pred, test_scaled_log$price)
RMSE(pls_pred, test_scaled_log$price)/mean(test_scaled_log$price)

compare_pls <- cbind(actual = test_scaled_log$price, pls_pred)
mean(apply(compare_pls, 1, min)/apply(compare_pls, 1, max))

# Tree Based Methods
train_vehicles <- as.data.table(train_vehicles)
test_vehicles <- as.data.table(test_vehicles)
oh_train <- data.frame(one_hot(train_vehicles))
oh_test <- data.frame(one_hot(test_vehicles))

tree_model <- tree(data = oh_train, price ~.)
summary(tree_model)
plot(tree_model)
text(tree_model, pretty = 0)
tree_model

tree_pred <- predict(tree_model, oh_test)
compare_tree <- cbind (actual = oh_test$price, tree_pred)
mean(apply(compare_tree, 1, min)/apply(compare_tree, 1, max))

# Tree with log(price)
train_log <- as.data.table(train_log)
test_log <- as.data.table(test_log)
oh_train_log <- data.frame(one_hot(train_log))
oh_test_log <- data.frame(one_hot(test_log))

tree_log_model <- tree(data = oh_train_log, price ~.)
summary(tree_log_model)
plot(tree_log_model)
text(tree_log_model, pretty = 0)
tree_log_model

tree_log_pred <- predict(tree_log_model, oh_test_log)
compare_tree_log <- cbind (actual = oh_test_log$price, tree_log_pred)
mean(apply(compare_tree_log, 1, min)/apply(compare_tree_log, 1, max))

# Bagging
bag_model <- bagging(formula = price ~., data = oh_train_log, nbagg = 100, coob = T)
bag_model
bag_pred <- predict(bag_model, oh_test_log)
compare_bag <- cbind (actual = oh_test_log$price, bag_pred)
mean(apply(compare_bag, 1, min)/apply(compare_bag, 1, max))

# Random Forest (it takes a lot of time to run!)
rf_model <- randomForest(data = oh_train_log, price ~.)
rf_model
rf_pred <- predict(rf_model, oh_test_log)
compare_rf <- cbind (actual = oh_test_log$price, rf_pred)
mean(apply(compare_rf, 1, min)/apply(compare_rf, 1, max))