library(highcharter)
library(neuralnet)
library(caret)
library(dplyr)
library(Hmisc)
library(smotefamily)
library(readr)
library(rio)
library(kdensity)
library(ROCit)
library(ggplot2)
library(doParallel)

seed <- set.seed(42)
data <- import("data/uscecchini28.csv")
data <- data[,-match(c("p_aaer", "new_p_aaer"), names(data))]
data <- data[complete.cases(data),]

all_names <- c("fyear", "misstate", "act", "ap", "at", "ceq", "che", 
               "cogs", "csho", "dlc", "dltis", "dltt", "dp", "ib", 
               "invt", "ivao", "ivst", "lct", "lt", "ni", "ppegt", 
               "pstk", "re", "rect", "sale", "sstk", "txp", "txt", 
               "xint", "prcc_f", "dch_wc", "ch_rsst", "dch_rec", 
               "dch_inv", "soft_assets", "dpi", "ch_cs", "ch_cm", 
               "ch_roa", "ch_fcf", "reoa", "EBIT", "issue", "bm")
all_data <- data[, match(all_names, names(data))]

normalize <- function(x){
  return((x - min(x)) / (max(x) - min(x)))
}

for (year in unique(all_data$fyear)){
  for (col in names(all_data)){
    all_data[data$fyear == year, col] <- normalize(
      all_data[data$fyear == year, col]
    ) 
  }
}
all_data <- all_data[, -1] 

raw_names <- c("misstate", "act", "ap", "at", "ceq", "che", "cogs", 
               "csho", "dlc", "dltis", "dltt", "dp", "ib", "invt", 
               "ivao", "ivst", "lct", "lt", "ni", "ppegt", "pstk", 
               "re", "rect", "sale", "sstk", "txp", "txt", "xint", 
               "prcc_f")

ratio_names <- c("misstate", "dch_wc", "ch_rsst", "dch_rec", 
                 "dch_inv", "soft_assets", "dpi", "ch_cs", "ch_cm", 
                 "ch_roa", "ch_fcf", "reoa", "EBIT", "issue", "bm")

raw_data <- all_data[, match(raw_names, names(all_data))]
ratio_data <- all_data[, match(ratio_names, names(all_data))]


### Ab hier beginnt der interressante Teil ###

train_test_split_smote<- function(data, y_col, frac = 0.7, seed = 42, 
                                  k = 5){
  set.seed(seed)
  
  smp_size <- floor(frac * nrow(data))
  train_ind <- sample(seq_len(nrow(data)), size = smp_size)
  
  X_smote <- data[train_ind, -match(y_col, names(data))]
  y_smote <- data[train_ind, match(y_col, names(data))]
  
  true_frac <- sum(y_smote) / length(y_smote)
  
  train_smote_object1 <- SMOTE(X_smote, y_smote, K = 5, 
                               dup_size = 1 / true_frac)$data
  
  X_train <- train_smote_object1[,-match('class', 
                                         names(train_smote_object1))]
  y_train <- as.numeric(train_smote_object1[,
                                            match('class', names(train_smote_object1))])
  
  X_test <- data[-train_ind, -match(y_col, names(data))]
  y_test <- data[-train_ind, match(y_col, names(data))]
  
  return(list(X_train = X_train, X_test = X_test, y_train = y_train, 
              y_test = y_test))
}

splitted_data <- train_test_split_smote(data = raw_data, 
                                        y_col = 'misstate', frac = 0.7)

X_train <- splitted_data$X_train
X_test <- splitted_data$X_test
y_train <- splitted_data$y_train
y_test <- splitted_data$y_test

train <- X_train
train$misstate <- y_train

### Ende interessanter Teil ###

cl <- makePSOCKcluster(5)
registerDoParallel(cl)

nn <- neuralnet(
  misstate ~.,
  data = train,
  hidden = 10,
  err.fct = 'sse',
  linear.output = FALSE,
  stepmax = 100000,
  lifesign = 'full',
  threshold = 15
)

stopCluster(cl)

output <- neuralnet::compute(nn, X_test)
y_pred_NN_all_data <- output$net.result
y_pred_NN_all_data <- as.vector(y_pred_NN_all_data)

evaluate <- function(test, pred, border = 0.5, k= 0.01){
  
  pred_round <- ifelse(pred >= border, 1, 0)
  confusion <- table(test, pred_round)
  TN <- confusion[1,1]
  TP <- confusion[2,2]
  FP <- confusion[1,2]
  FN <- confusion[2,1]
  
  total_acc <- numeric(2)
  total_acc[1] <- NaN
  total_acc[2] <- round((TN + TP) / sum(confusion),4)
  
  prec <- numeric(2)
  prec[1] <- NaN
  prec[2] <- round(TP / (TP + FP),4)
  
  sens <- numeric(2)
  sens[1] <- NaN
  sens[2] <- round(TP / (TP + FN),4)
  
  F1 <- numeric(2)
  F1[1] <- NaN
  F1[2] <- round(2*(prec[2]*sens[2])/(prec[2] + sens[2]), 4)
  
  F.score <- function(beta, p = prec[2], s = sens[2]){
    round((1 + beta^2)*(p*s)/(beta^2*p + s),4)
  }
  
  F2 <- numeric(2)
  F2[1] <- NaN
  F2[2] <- F.score(2)
  
  F.5 <- numeric(2)
  F.5[1] <- NaN
  F.5[2] <- F.score(0.5)
  
  ROCit_obj <- rocit(score=pred,class=test)
  AUC <- numeric(2)
  AUC[1] <- NaN
  AUC[2] <- round(ROCit_obj$AUC, 4)
  plot(ROCit_obj)
  
  NDCG_at_k <- numeric(2)
  NDCG_at_k[1] <- NaN
  
  NDCG_df <- cbind(test, pred)
  NDCG_df <- NDCG_df[order(-pred),]
  k_frac <- round(length(test)*k)
  
  actual_tpr <- sum(NDCG_df[0:k_frac, 1]) / k_frac
  best_tpr <- sum(sort(test, decreasing=T)) / k_frac
  
  NDCG_at_k[2] <- actual_tpr / best_tpr
  
  return(cbind(confusion, total_acc, prec, sens, F1, F2, F.5, AUC, NDCG_at_k))
}

evaluate(y_test, y_pred_NN_all_data)