##************************************************************************************************************************
## Developer : Deepansh Parab
## Start Date : 21th April, 2017
## Project : KNN_FinalProject
##************************************************************************************************************************


rm(list=ls())
kNN_dist <- function(features, p, predFeatures=NULL, cv=FALSE)
{
  # Reference the required libraries.
  library(assertthat, quietly=TRUE)
  
  if (cv)
  {
    assert_that(is.null(predFeatures))
    predFeatures = features
  }
  not_empty(features)
  assert_that(p > 0)
  
  N <- ncol(features)
  
  # Create a distances matrix, initially containing all NA values.
  dist <- matrix(NA, nrow(predFeatures), nrow(features))
  
  # Fill the matrix in different ways based on the value of p.
  if (p == Inf)
  {
    # Loop through all test features.  The loop is inside the conditional on p for performance.
    for (i in 1:nrow(predFeatures))
    {
      # Store the Chebyshev distances for each training feature to the test feature.
      dist[i,] <- 0
      for (j in 1:N)
      {
        dist[i,] <- pmax.int(dist[i,], abs(features[,j]-predFeatures[i,j]))
      }
    }
  }
  else
  {
    # Precalculate the reciprocal of p to reduce computation inside the loop below.
    p_recip <- 1/p
    
    # Loop through all test features.  The loop is inside the conditional on p for performance.
    for (i in 1:nrow(predFeatures))
    {
      # Store the Euclidean distances (or other-powered distances if p!=2) for each training
      # feature to the test feature.
      dist[i,] <- 0
      for (j in 1:N)
      {
        dist[i,] <- dist[i,] + abs(features[,j]-predFeatures[i,j])^p
      }
      dist[i,] <- dist[i,] ^ p_recip
    }
  }
  
  return(dist)
}


kNN_eval <- function(dist, features, labels, uniqueLabels, k, r, predFeatures=NULL, cv=FALSE)
{
  if (cv)
  {
    assert_that(is.null(predFeatures))
    predFeatures = features
  }
  
  # Reference the required libraries.
  library(assertthat, quietly=TRUE)
  
  # Assert that the training features, training labels, and prediction features
  # are consistent with each other and have the necessary dimensionality.
  not_empty(features)
  not_empty(labels)
  not_empty(predFeatures)
  assert_that(ncol(features) == ncol(predFeatures))
  assert_that(nrow(features) == length(labels))
  
  kNeighbors <- apply(dist, 1, order)[(1+cv):(k+cv),]
  
  # Create a matrix of labels with appropriate dimension.
  kLabels <- matrix(NA, k, nrow(predFeatures))
  kDistances <- matrix(NA, k, nrow(predFeatures))
  
  if (k == 1)
  {
    kLabels[1,] <- labels[kNeighbors]
    kDistances[1,] <- dist[1,kNeighbors]
  }
  else
  {
    kLabels[,] <- labels[kNeighbors]
    
    for (j in 1:k)
    {
      kDistances[j,] <- dist[,kNeighbors[j]]
    }
  }
  
  # Create slots for each new prediction.
  predLabels <- rep(0,nrow(predFeatures))

  for (m in 1:nrow(predFeatures))
  {
    buckets <- rep(0, length(uniqueLabels))
    for (j in 1:k)
    {
      position <- which(uniqueLabels == kLabels[j,m])
      buckets[position] <- buckets[position] + kDistances[j,m]^(1/r)
    }
    predLabels[m] <- uniqueLabels[which.max(buckets)]
  }
  
  # If we are doing cross-validation, generate our accuracy and return it.
  if (cv)
  {
    return(mean(predLabels==labels))
  }
  

  matches <- apply(kLabels, 1, `==`, as.matrix(predLabels))
  # ...and then count them.
  counts <- apply(matches, 1, sum)
  # Use the counts to determine our probability scores.
  prob <- counts/k
  
  # Return predicted labels and corresponding probabilities in a list.
  return(list(predLabels=predLabels, prob=prob))
}


kNN_cv <- function(trainFeatures, trainLabels, uniqueLabels, pChoices, kChoices, rChoices, outputCsv)
{
  df <- data.frame(k=0, p=0, r=0, result=NA)
  
  for (p in pChoices)
  {
    # The distance matrix only varies based on values of p.  It can be reused among
    # all validation on k and r.
    dist <- kNN_dist(trainFeatures, p, NULL, cv=TRUE)
    for (k in kChoices)
    {
      for (r in rChoices)
      {
        # Reuse the distance matrix and finish the validation with a call to kNN_eval.
        result <- kNN_eval(dist, trainFeatures, trainLabels, uniqueLabels, k, r, NULL, cv=TRUE)
        df <- rbind(df, c(k, p, r, result))
        print(df)
        write.csv(df, outputCsv)
      }
    }
  }
}



library(caret)
library(doParallel)

cl <- makeCluster(detectCores())
registerDoParallel(cl)

kNN_predict <- function(trainFeatures, trainLabels, uniqueLabels, p, k, r, predFeatures)
{
  dist <- kNN_dist(trainFeatures, p, predFeatures)
  kNN_eval(dist, trainFeatures, trainLabels, uniqueLabels, k, r, predFeatures)
}

# Load the training and test data.
trainDigits <- read.csv("/Users/deepanshparab/Desktop/cs-513Project/knn/data/mnist_train.csv")
testDigits <- read.csv("/Users/deepanshparab/Desktop/cs-513Project/knn/data/mnist_test.csv")
trainFeatures <- trainDigits[1:6000,2:785]
trainLabels <- trainDigits[1:6000,1]
predFeatures <- read.csv("/Users/deepanshparab/Desktop/cs-513Project/knn/data/mnist_test.csv")
testFeature <- predFeatures[1:1500,2:785]
testVarify <- predFeatures[1:1500,1]
# Run the kNN prediction using my tuned parameters for p, k, and r.
results <- kNN_predict(trainFeatures, trainLabels, 0:9, 4.2, 1, -0.25, testFeature)

# Save the predictions to file.
df <- data.frame(predLabels=results$predLabels)




accuracy <-sum(diag(table(results$predLabels,testVarify)))/nrow(testVarify)
View(df)
write.csv(df, "MNIST_predictions.csv", row.names=FALSE)






