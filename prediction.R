## prediction.R
##
## Matthew Gast, October 2015
##
## Course: "Practical Machine Learning" at the JHU Bloomberg School of
## Health (Coursera Data Science Specialization).  This file builds a
## prediction model of fitness data, predicting the type of activity
## from measurements of that activity.

loadPackages <- function () {
# Load required packages for the analysis of fitness data.
#
# Input:  None.
# Output: None.

    # Caret library supports regression and data training functions
    library(caret)
    # Random forest classification library
    library(randomForest)
    
}

setConstants <- function () {
# Set global constants used in analysis of fitness data
#
# Input:  None.
# Output: No function output, but global constants are available in
# scope.

    # Set fraction of data used for validation and training
    validation.pct <<- 0.4
    train.pct <<- 1 - validation.pct

    # Project directory
    proj.dir <<- "/Users/mgast/Dropbox/data-science-specialization/8-practical-machine-learning/Practical_Machine_Learning_CourseProject"
    
    # Location of training and testing data for download
    train.url <<- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    test.url <<- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    
    # Metadata columns to drop in the data
    metadata.cols <<- c("X","user_name","raw_timestamp_part_1",
                   "raw_timestamp_part_2","cvtd_timestamp",
                   "new_window","num_window")

}


download.pmlfile <- function(dir=getwd() ) {
# This function downloads fitness data into the specified directory.
# If no directory is specified, the current working directory is
# used. The file is not downloaded if it already exists.
#
# Input: A directory used to store fitness data.  If none is
# specified, the current working directory is used.   
# Output: Testing and training files are in the specified directory.
    
    orig.wd <- getwd()
    setwd(dir)
    if (!file.exists("pml-training.csv")) {
        download.file(train.url, "pml-training.csv")
    }
    if (!file.exists("pml-testing.csv")) {
        download.file(test.url, "pml-testing.csv")
    }
    setwd(orig.wd)
    
}

read.pmlfile <- function(file, dir=getwd() ) {
# This function reads a specified fitness file name from the specified
# directory, returning its contents as a data frame.  The read process
# replaces any blank spaces or division-by-zero errors with
# not-a-number (NA) values in R.
#
# Input: A file name to be read, and a directory in which that file is
# held.  If no directory is specified, the current working directory
# is used.
# Output: The contents of the file are returned as a data frame.

    filename = paste(dir,file,sep="/")
    df <- read.csv(filename, na.strings=c("#DIV/0!", "NA", ""))
    return (df)
    
}

remove.na.cols <- function (df) {

# This function removes columns that contain any NA values to sanitize
# the data.  By inspection, many of the columns in fitness data files
# that have NA values have significant numbers of NA values and cannot
# be used.
#
# Input: A data frame    
# Output: The data frame, with columns containing NA values removed.
    
    na.columns <- apply(df,2, function(x) any(is.na(x)))
    df <- df[,!na.columns]
    return(df)

}

remove.metadata.cols <- function (df) {

# This function removes "metadata" columns in the input data frame.
# Metadata separates data into time series, which has little
# predictive power.  Therefore, we remove metadata.
#    
# Input: A data frame.  This function also uses a global variable with
# the metadata columns to remove.
# Output: The data frame, with the specified columns removed.
    
    for (col in metadata.cols) {
        df[,col] <- NULL
    }
    return(df)
    
}

## Step 1: Read & preprocess data
loadPackages()
setConstants()
download.pmlfile(proj.dir)
training.raw <- read.pmlfile("pml-training.csv")
testing.raw <- read.pmlfile("pml-testing.csv")

training.data <- remove.na.cols(training.raw)
training.data <- remove.metadata.cols(training.data)

testing.data <- remove.na.cols(testing.raw)
testing.data <- remove.metadata.cols(testing.data)

## Step 2: Build prediction model

# remove zero variance variables, if any

nsv.cols <- nearZeroVar (training.data)
if (length(nsv.cols) > 0) {
    training.data <- training.data[-nsv.cols]
}

# split into training & validation sets

in.train <- createDataPartition(training.data$classe, p=train.pct, list=FALSE)
train.set <- training.data[in.train,]
validate.set <- training.data[-in.train,]

# build prediction model using random forests, with dependent variable
# "classe" and using all others as predictors

y <- as.factor(train.set$classe)
x <- train.set[-ncol(train.set)] # Remove last column (classe)
rf <- randomForest(y ~ ., data=x)

# Print accuracy of predictions
train.pred <- predict (rf, train.set)
confusionMatrix(train.pred, train.set$classe)

# Validate model with validation set and print accuracy
valid.pred <- predict (rf, validate.set)
confusionMatrix(valid.pred,validate.set$classe)

## Step 3: Write output files with predictions

# This function was supplied by class instructors
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

test.pred <- predict(rf, testing.raw)
pml_write_files(test.pred)

