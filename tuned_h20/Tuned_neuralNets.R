library(h2o)

rm(list=ls())
train <- read.csv ("/Users/deepanshparab/Desktop/cs-513Project/h20/data/train.csv")


# Create a 28*28 matrix with pixel color values
m = matrix(unlist(train[10,-1]), nrow = 28, byrow = TRUE)

# Plot that matrix
image(m,col=grey.colors(255))

# reverses (rotates the matrix)
rotate <- function(x) t(apply(x, 2, rev)) 

# Plot some of images
par(mfrow=c(2,3))
lapply(1:6, 
       function(x) image(
         rotate(matrix(unlist(train[x,-1]),nrow = 28, byrow = TRUE)),
         col=grey.colors(255),
         xlab=train[x,1]
       )
)

par(mfrow=c(1,1)) # set plot options back to default

# partitioning the datasets
library (caret)
inTrain<- createDataPartition(train$label, p=0.8, list=FALSE)
training<-train[inTrain,]
testing<-train[-inTrain,]



library(h2o)

#start a local h2o cluster
local.h2o <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, nthreads=-1)

training <- train[0:6000,] 
testing  <- testing[0:1500,]

# convert digit labels to factor for classification
training[,1]<-as.factor(training[,1])

# pass dataframe from inside of the R environment to the H2O instance
trData<-as.h2o(training)
tsData<-as.h2o(testing)

res.dl <- h2o.deeplearning(x = 2:785, y = 1, trData, activation = "Tanh",hidden=c(500,500,1000), epochs = 20)
# Accuracy =  0.9833333
pred.dl<-h2o.predict(object=res.dl,newdata=tsData[,-1])
pred.dl.df<-as.data.frame(pred.dl)
sum(diag(table(testing$label,pred.dl.df[,1])))
#table(testing$label,pred.dl.df[,1])
accuracy_sum <- sum(diag(table(testing$label,pred.dl.df[,1])))/nrow(testing)

accuracy_sum