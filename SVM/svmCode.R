##************************************************************************************************************************
## Developer : Deepansh Parab
## Start Date : 21th April, 2017
## Project : SVM_FinalProject
##************************************************************************************************************************

rm(list= ls())
library(caret)
library(doParallel)

cl <- makeCluster(detectCores())
registerDoParallel(cl)



load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train <<- load_image_file('/Users/deepanshparab/Desktop/cs-513Project/dataset/train-images.idx3-ubyte')
  test <<- load_image_file('/Users/deepanshparab/Desktop/cs-513Project/dataset/t10k-images.idx3-ubyte')
  
  train$y <<- load_label_file('/Users/deepanshparab/Desktop/cs-513Project/dataset/train-labels.idx1-ubyte')
  test$y <<- load_label_file('/Users/deepanshparab/Desktop/cs-513Project/dataset/t10k-labels.idx1-ubyte')  
}

train <- data.frame()
test <- data.frame()

load_mnist()

show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

# Normalize: X = (X - min) / (max - min) => X = (X - 0) / (255 - 0) => X = X / 255.
train$x <- train$x / 255

# Setup training data with digit and pixel values with 60/40 split for train/cv.
inTrain = data.frame(y=train$y, train$x)
inTrain$y <- as.factor(inTrain$y)
?createDataPartition
trainIndex = createDataPartition(inTrain$y, p = 0.60,list=FALSE)
training = inTrain[trainIndex,]
cv = inTrain[-trainIndex,]

# SVM. 95/94.
fit <- train(y ~ ., data = head(training, 6000), method = 'svmRadial', tuneGrid = data.frame(sigma=0.0107249, C=1))
results <- predict(fit, newdata = head(cv, 6000))
confusionMatrix(results, head(cv$y, 6000))



# Draw the digit.
show_digit(as.matrix(training[1000,2:785]))

# Predict the digit.
prediction <- predict(fit, newdata = training[0:6000,])

# Check the actual answer for the digit.
trained_model<- training[1:6000,1]

table(prediction,trained_model)

sum_acc1<-sum(diag(table(prediction,trained_model)))/6000