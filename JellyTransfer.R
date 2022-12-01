library(tensorflow)
library(keras)
library(stringr)
library(readr)
library(purrr)
library(caret)
library(e1071)
library(imager)
library(magrittr)
library(abind)
set.seed(1)

Aurelia_labiata <- list.files('Documents/UofA/MATH574M/FinalProject/Jelly/Aurelia_labiata/')

file_path <- 'Documents/UofA/MATH574M/FinalProject/Jelly/Aurelia_labiata/00000000018.jpg'
#print(file_path)
test_im <- as.array(load.image(file_path))
dim(test_im) <- c(1, 75, 75, 3)
ua_data <- test_im
for (file in Aurelia_labiata) {
  file_path <- paste('Documents/UofA/MATH574M/FinalProject/Jelly/Aurelia_labiata/', sep="", file)
  #print(file_path)
  im <- as.array(load.image(file_path))
  dim(im) <- c(1, 75, 75, 3)
  ua_data <- abind(ua_data, im, along=1)
}
ua_data <- ua_data[2:301, , , ]
sample <- sample(c(TRUE, FALSE), nrow(ua_data), replace=TRUE, prob=c(0.8, 0.2))
ua_data_xtrain <- ua_data[sample, , , ]
ua_data_xtest <- ua_data[!sample, , , ]
ua_data_ytrain <- matrix(0, nrow(ua_data_xtrain), 1)
ua_data_ytest <- matrix(0, nrow(ua_data_xtest), 1)

Chrysaora_fuscescens <- list.files('Documents/UofA/MATH574M/FinalProject/Jelly/Chrysaora_fuscescens/')
to_data <- test_im
for (file in Chrysaora_fuscescens) {
  file_path <- paste('Documents/UofA/MATH574M/FinalProject/Jelly/Chrysaora_fuscescens/', sep="", file)
  #print(file_path)
  im <- as.array(load.image(file_path))
  dim(im) <- c(1, 75, 75, 3)
  to_data <- abind(to_data, im, along=1)
}
to_data <- to_data[2:301, , , ]
sample <- sample(c(TRUE, FALSE), nrow(to_data), replace=TRUE, prob=c(0.8, 0.2))
to_data_xtrain <- to_data[sample, , , ]
to_data_xtest <- to_data[!sample, , , ]
to_data_ytrain <- matrix(1, nrow(to_data_xtrain), 1)
to_data_ytest <- matrix(1, nrow(to_data_xtest), 1)


Stomolophus_meleagris <- list.files('Documents/UofA/MATH574M/FinalProject/Jelly/Stomolophus_meleagris/')
uar_data <- test_im
for (file in Stomolophus_meleagris) {
  file_path <- paste('Documents/UofA/MATH574M/FinalProject/Jelly/Stomolophus_meleagris/', sep="", file)
  #print(file_path)
  im <- as.array(load.image(file_path))
  dim(im) <- c(1, 75, 75, 3)
  uar_data <- abind(uar_data, im, along=1)
}
uar_data <- uar_data[2:301, , , ]
sample <- sample(c(TRUE, FALSE), nrow(uar_data), replace=TRUE, prob=c(0.8, 0.2))
uar_data_xtrain <- uar_data[sample, , , ]
uar_data_xtest <- uar_data[!sample, , , ]
uar_data_ytrain <- matrix(2, nrow(uar_data_xtrain), 1)
uar_data_ytest <- matrix(2, nrow(uar_data_xtest), 1)


# Aurelia_aurita <- list.files('Documents/UofA/MATH574M/FinalProject/Jelly/Aurelia_aurita/')
# um_data <- test_im
# for (file in Aurelia_aurita) {
#   file_path <- paste('Documents/UofA/MATH574M/FinalProject/Jelly/Aurelia_aurita/', sep="", file)
#   #print(file_path)
#   im <- as.array(load.image(file_path))
#   dim(im) <- c(1, 75, 75, 3)
#   um_data <- abind(um_data, im, along=1)
# }
# um_data <- um_data[2:301, , , ]
# sample <- sample(c(TRUE, FALSE), nrow(um_data), replace=TRUE, prob=c(0.8, 0.2))
# um_data_xtrain <- um_data[sample, , , ]
# um_data_xtest <- um_data[!sample, , , ]
# um_data_ytrain <- matrix(3, nrow(um_data_xtrain), 1)
# um_data_ytest <- matrix(3, nrow(um_data_xtest), 1)

#############################################################################################

train_data <- abind(ua_data_xtrain, to_data_xtrain, uar_data_xtrain, along=1)
otrain_label <- abind(ua_data_ytrain, to_data_ytrain, uar_data_ytrain, along=1)
test_data <- abind(ua_data_xtest, to_data_xtest, uar_data_xtest, along=1)
otest_label <- abind(ua_data_ytest, to_data_ytest, uar_data_ytest, along=1)

train_label <- to_categorical(otrain_label, num_classes = 3)
test_label <- to_categorical(otest_label, num_classes = 3)

##############################################################################################


############################################################################

pretrained <- application_xception(weights = 'imagenet',
                                   include_top = FALSE,
                                   input_shape = c(75, 75, 3))

model <- keras_model_sequential() %>%
  pretrained %>%
  layer_flatten() %>%
  layer_dense(units = 3, activation = "softmax")

model %>% compile(
  optimizer = optimizer_adam(0.0001),
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

history <- model %>% 
  fit(
    x = train_data, y = train_label,
    epochs = 50,
    validation_split=0.2,
    use_multiprocessing=TRUE,
    callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE))
  )

plot(history)


############################################################################
Prediction_data_train<- model %>% predict(train_data) %>% k_argmax()
u <- union(Prediction_data_train, otrain_label)
t <- table(factor(Prediction_data_train, u), factor(otrain_label, u))
confusionMatrix(t)

Prediction_data_test <- model %>% predict(test_data) %>% k_argmax()
u <- union(Prediction_data_test, otest_label)
t <- table(factor(as.matrix(Prediction_data_test), u), factor(otest_label, u))
confusionMatrix(t)