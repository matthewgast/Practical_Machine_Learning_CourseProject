# Prediction of Activity by Performance Measurement
Matthew Gast  
October 2015  

## Background

Quantified self devices collect data on personal activity.  One limitation of these devices is that users regularly measure the amount of an activity, but not its quality.  This project uses accelerometer data from a series of activities done both correctly and incorrectly.

Six subjects in the project were asked to lift barbells in one of five different ways:

+ Class A: exactly according to the specification
+ Class B: throwing the elbows to the front
+ Class C: lifting the dumbbell only halfway
+ Class D: lowering the dumbbell only halfway
+ Class E: throwing the hips to the front

More information is available from the website here:
http://groupware.les.inf.puc-rio.br/har (see the section on the Weight
Lifting Exercise Dataset).

## Goal

The purpose of this project is to use data to predict the manner in which the weight lift was performed.  The outcome is the activity type, stored in the "classe" variable in the training set.

Contents
+ How you built your model
+ How you used cross validation
+ What you think the expected out of sample error is
+ Why you made teh choices you did

Constraints
+ Write up < 2000 words
+ < 5 figures
+ It will make it easier for the graders if you submit a repo with a
gh-pages branch so the HTML page can be viewed online (and you always
want to make it easy on graders :-).

## Load the data


```r
source("prediction.R")
loadPackages()
setConstants()
```

Begin by downloading the data and reading it into data structures.
The read functions defined for this project automatically convert
Excel division-by-zero (#DIV/O!") and unavailable values into R's NA
value.


```r
download.pmlfiles(proj.dir)
training.raw <- read.pmlfile("pml-training.csv")
testing.raw <- read.pmlfile("pml-testing.csv")
```

## Clean Data

To clean the data, remove columns with NA values and the metadata columns.


```r
training.data <- remove.na.cols(training.raw)
training.data <- remove.metadata.cols(training.data)
testing.data <- remove.na.cols(testing.raw)
testing.data <- remove.metadata.cols(testing.data)
```

## Model Building

Remove any columns that have low variance.


```r
nsv.cols <- nearZeroVar (training.data)
if (length(nsv.cols) > 0) {
    training.data <- training.data[-nsv.cols]
}
```

### Training and validation sets

For cross-validation, split the data into a training component and a
test component.  Note that the split is controlled by a global
variable.


```r
in.train <- createDataPartition(training.data$classe, p=train.pct, list=FALSE)
train.set <- training.data[in.train,]
test.set <- training.data[-in.train,]
```

### Model building

The dependent variable in the model is a factor, so re-cast the output
of the prediction function as a factor.


```r
y <- as.factor(train.set$classe)
x <- train.set[-ncol(train.set)]
```

### Random Forest model

First, try a random forest model.


```r
model.rf <- randomForest(y ~ ., data=x)
```

Print accuracy of predictions against the training set.


```r
rf.train.pred <- predict (model.rf, train.set)
confusionMatrix(rf.train.pred, train.set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

For validation of the model, test the accuracy against data not used to train the model.  Still looks good!


```r
rf.test.pred <- predict (model.rf, test.set)
confusionMatrix(rf.test.pred,test.set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2226    6    0    0    0
##          B    5 1507   12    0    0
##          C    0    5 1353   23    1
##          D    0    0    3 1261    7
##          E    1    0    0    2 1434
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9917          
##                  95% CI : (0.9895, 0.9936)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9895          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9973   0.9928   0.9890   0.9806   0.9945
## Specificity            0.9989   0.9973   0.9955   0.9985   0.9995
## Pos Pred Value         0.9973   0.9888   0.9790   0.9921   0.9979
## Neg Pred Value         0.9989   0.9983   0.9977   0.9962   0.9988
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2837   0.1921   0.1724   0.1607   0.1828
## Detection Prevalence   0.2845   0.1942   0.1761   0.1620   0.1832
## Balanced Accuracy      0.9981   0.9950   0.9923   0.9895   0.9970
```

### Decision tree model

Set up a classification tree to see if it works better.


```r
model.dt <- rpart(y ~ ., data=x, method="class")
dt.test.pred <- predict(model.dt, test.set, type="class")
confusionMatrix(dt.test.pred, test.set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2024  315   27  149   38
##          B   54  782   68   59  107
##          C   54  295 1162  146  190
##          D   82  105  110  859  110
##          E   18   21    1   73  997
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7423          
##                  95% CI : (0.7325, 0.7519)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6728          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9068  0.51515   0.8494   0.6680   0.6914
## Specificity            0.9058  0.95449   0.8943   0.9380   0.9824
## Pos Pred Value         0.7928  0.73084   0.6291   0.6785   0.8982
## Neg Pred Value         0.9607  0.89138   0.9657   0.9351   0.9339
## Prevalence             0.2845  0.19347   0.1744   0.1639   0.1838
## Detection Rate         0.2580  0.09967   0.1481   0.1095   0.1271
## Detection Prevalence   0.3254  0.13638   0.2354   0.1614   0.1415
## Balanced Accuracy      0.9063  0.73482   0.8718   0.8030   0.8369
```

Accuracy is only 71%, compared to 99% in the random forest model.

### Model choice

The random forest model has much better predictive value.

## Predict on test data and write it out


```r
test.pred <- predict(model.rf, testing.raw)
pml_write_files(test.pred)
```
