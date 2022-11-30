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


# Download the dataset first from the links provided in the csv file
# we will be using 1000 images for now. One can increase the dataset 
# size as per the requirement.

set.seed(42)

num_samples = 50
jelly = read.csv("/Users/gaurav/Desktop/Education/M-574/semesterproject/jellyfish_t14.csv")
# shuffling the dataset and take n_samples out of it
jelly = jelly[sample(1:nrow(jelly)),][0:num_samples,]
links= jelly[,4]
dst_dir = "/Users/gaurav/Desktop/Education/M-574/semesterproject/M574_project/images/"
for (n in 1:num_samples){
  link = links[n]
  print(paste(dst_dir,n,".jpg"))
  download.file(link, destfile = paste(dst_dir,n,".jpg", sep=""),
                method = "wget", extra = "-r -p --random-wait")
}

# split the dataset into train, test
taxon_name = jelly[,5]
images = paste0(dst_dir, seq(1:num_samples), ".jpg")
df <- do.call(rbind.data.frame, Map('c', images, taxon_name))
colnames(df)[1] <- "image_path"
colnames(df)[2] <- "taxon_name"
# labeling: text labels to categorical 
numeric_labels = unclass(as.factor(df$taxon_name))
df = cbind(df, numeric_labels)
# check first 5 samples 
head(df, n=5)

# Let's split the dataset into train, and test
# 80% train 20% test
sample = sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.8,0.2))
train.df = df[sample, ]
test.df = df[!sample, ]

train.df.image_paths = train.df$image_path
train.df.labels = train.df$numeric_labels
test.df.image_paths = test.df$image_path
test.df.labels = test.df$numeric_labels

# preprocess the images
preprocess_image <- function(image){
  width <- 50
  height <- 50
  # resize 
  image <- resizeImage(image, w = width, h = height)
  # padding
  image <- padding(image, 150, 150, fill_value=0)
  image <- image$data
  # normalizing between 0 to 1
  image <- image/255
  
  return(image)
}

calling_fn <- function(img){
  tryCatch(

    {
      image=readImage(img)
      image=preprocess_image(image)
      return(image)
    },
    #if an error occurs, tell me the error
    error=function(e) {
      message('An Error Occurred')
      print(e)
    },
    #if a warning occurs, tell me the warning
    warning=function(w) {
      message('A Warning Occurred')
      print(w)
      return(NA)
    }
  )
}

image.array = c()
count=0
for (img in train.df.image_paths){
  print(img)
  image = calling_fn(img)
  print(image[1])
  ifelse(is.na(image), print("Discarding Image!"), {count=count+1})
  
  #image = readImage(img)
  #image = my_function(image)
  image.array = c(image.array, image)
}

image.array = array(data = image.array, dim= c(count, 150, 150, 3))

dim(train_data) <- c(num_samples,32,32,3)
dim(test_data) <- c(10000,32,32,3)
dim(train_label) <- c(50000)
dim(test_label) <- c(10000)

#######################################################################
# preprocessing images for CNN
#library(e1071)
#cifar <- dataset_cifar10()

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
  l1ayer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>% 
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

