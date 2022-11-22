# importing packages

#install.packages("keras")
#install.packages("magicK")
#install.packages("OpenImageR")

library(tensorflow)
library(keras)
library(OpenImageR)
library(pbapply)
library(magick)
library(stringr)
library(readr)
library(purrr)
library(caret)

# preprocessing images for CNN
library(e1071)
cifar <- dataset_cifar10()

preprocess_image <- function(image){
    width <- 100
    height <- 100
    # resize 
    image <- resizeImage(image, w = width, h = height)
    # padding
    image <- padding(image, 150, 150, fill_value=0)
    image <- image$data
    # normalizing between 0 to 1
    im <- im/255
    
    return(image)
}

# read csv file having images_path and labels
train_data <- scale(cifar$train$x)
dim(train_data) <- c(50000,32,32,3)

test_data <- scale(cifar$test$x)
dim(test_data) <- c(10000,32,32,3)

train_label <- as.numeric(cifar$train$y)
dim(train_label) <- c(50000)

test_label <- as.numeric(cifar$test$y)
dim(test_label) <- c(10000)

#######################################################################

class_names <- c('airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck')

index <- 1:30

par(mfcol = c(5,6), mar = rep(1, 4), oma = rep(0.2, 4))

cifar$train$x[index,,,] %>% 
  purrr::array_tree(1) %>%
  purrr::set_names(class_names[cifar$train$y[index] + 1]) %>% 
  purrr::map(as.raster, max = 255) %>%
  purrr::iwalk(~{plot(.x); title(.y)})

######################################################
################## CNN MODEL #########################
######################################################

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", input_shape = c(32,32,3)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_batch_normalization() %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_batch_normalization() %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) 

summary(model)

model %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

summary(model)

model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

history <- model %>% 
  fit(
    x = train_data, y = train_label,
    epochs = 2,
    validation_split=0.2,
    use_multiprocessing=TRUE
  )

plot(history)

############################################################################
Prediction_train_data <- predict_classes(model, train_data)
#model %>% predict(x) %>% k_argmax()%>%as.vector()
confusionMatrix(table(Prediction_train_data,train_label))

Prediction_data_test <-predict_classes(model, test_data)
confusionMatrix(table(Prediction_data_test,test_label))

