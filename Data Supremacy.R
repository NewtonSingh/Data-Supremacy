



train = read.csv(file = "C:/Users/HP/Documents/train_tar.csv",header = TRUE)
test  = read.csv(file = "C:/Users/HP/Documents/test_tar.csv",header = TRUE)

train_1 = train[,-c(1,2)]
test_1  = test[,-c(1,2)]



na_train = sapply(X = train,FUN = function(x) sum(is.na(train)))

na_test  = sapply(X = test,FUN = function(x) sum(is.na(test)))

summary(train)

# Gradient method

y_tar = "target"

y = train_1[y_tar]


#all = rbind(tr,test)

train_1$training_hours = scale(x = train$training_hours,center = T)

test_1$training_hours = scale(x = test$training_hours,center = T)

one_hot = dummyVars(~ ., train_1, fullRank = FALSE)

library(dplyr)

ames_train_hot = predict(one_hot, train_1) %>% as.data.frame()

var_names = colnames(ames_train_hot)

predictors = setdiff(var_names,y_tar)

n_trn = nrow(train_1)

tr = ames_train_hot[predictors]

library(gbm)


ntrees = 500

model = gbm.fit(x = tr,
  y = train$target,distribution = "bernoulli",n.trees = ntrees,
  shrinkage = 0.005,interaction.depth =25,
  n.minobsinnode = 15,nTrain = round(n_trn*0.85),
  verbose = TRUE,bag.fraction = 0.8)

gbm.perf(model)

summary(model)

one_hot_1 = dummyVars(~ ., test_1, fullRank = FALSE)

ames_test_hot = predict(one_hot_1, test_1) %>% as.data.frame()

# Predictions
#te = test[,-c(1,2)]

test_pred = predict(object = model,newdata = ames_test_hot,n.trees = 
    gbm.perf(model,plot.it = FALSE),type = "response")

train_pred =predict(object = model,newdata = tr,n.trees = 
    gbm.perf(model,plot.it = FALSE),type = "response") 


head(train_pred,n = 100)

head(test_pred,n = 100)


sub_m = data.frame("enrollee_id" = test$enrollee_id,"target" = test_pred)

write.csv(x = sub_m,file = "sub_m5.csv")



# Xgboost Method

set.seed(2001)
ind = sample(nrow(train),nrow(train)* 0.80,replace = T)


training = train[ind,-c(1)]
testing =  train[-ind,-c(1)]

library(xgboost)
library(dplyr)
library(matrix)
# Create matrix one-hot encoding
tr_label = training[,"target"]

options(na.action = na.pass)
train_m = sparse.model.matrix(target~.-target,data = training)

train_matrix = xgb.DMatrix(data = as.matrix(train_m),label = tr_label)

test_m = sparse.model.matrix(target~.-target,data = testing)

test_label = testing[,"target"]

test_matrix = xgb.DMatrix(data = as.matrix(test_m),label = test_label)

#nc = length(unique(tr_label))

#rdesc = makeResampleDesc("CV",stratify = T,iters=5L)

xgb_params = list("objective" = "binary:logistic",
  "eval_metric" = "auc")


watch_list = list(train = train_matrix,test = test_matrix)

# Xtreme gradient boosting

bst_mod = xgb.train(params = xgb_params,
  data = train_matrix,
  nrounds =500,
  watchlist = watch_list,
  eta = 0.05,
  max.depth = 20,subsample = 0.6
  ,gamma = 12
  ,alpha = 1)

#xgbcv= xgb.cv( params = xgb_params, data = train_matrix,
#  nrounds = 100, nfold = 5, showsd = T,
 # stratified = T, print.every.n = 10,
 # early.stop.round = 20, maximize = F)
#xgbcv$best_ntreelimit
#Training and test plot

e = data.frame(bst_mod$evaluation_log)

#plot(e$iter,e$train_mlogloss,col = "blue")
#lines(e$iter,e$test_mlogloss,col = "red")

a = max(e$test_auc)

e[e$test_auc == a,]

p = predict(bst_mod,newdata = test_matrix,type = "response")
head(p)


main_mat = sparse.model.matrix(~.,data = test[,-c(1)])

main_matrix = xgb.DMatrix(data = as.matrix(main_mat))

p_1 = predict(bst_mod,newdata = main_matrix)
head(p_1)

sub_m = data.frame("enrollee_id"=test_data$enrollee_id,"target" = p_1)

write.csv(x = sub_m,file = "sub_m3.csv")


# Logit method

library(plyr) 
#install.packages("caret")
library(caret) # Confusion matrix

# false pos rate 

fp_rate = NULL

# false negative rate

fn_rate = NULL

# Number of Iterations 

k = 200

#Now we will initialize our progress bar

p_bar =create_progress_bar("text")

p_bar$init(k)

# Accuracy

acc = NULL

set.seed(120)

for(i in 1:k)
{
  
  # Train-test Split
  
  smp_size = floor(0.80*nrow(train))
  
  index = sample(seq_len(nrow(train),size = smp_size)
  
  tr = train[index,-c(1)]
  
  te  = train[-index,-c(1)]
  
  train[] = lapply(tr, function(x) {
    if(is.factor(x)) as.numeric(as.character(x)) else x
})

  test[] = lapply(te, function(x) {
    if(is.factor(x)) as.numeric(as.character(x)) else x
})
  
  # Fitting the logistic eqn
  
  model = glm(tr$target~.,family = binomial,data = tr)
  
  # Predicting results
  
  results_pr = predict(model,te,type = "response")
  
  results = ifelse(te = results_pr > 0.5,yes = 1,no = 0)
  
  answers = te$target
  
  mis_classification = mean(answers!=results)
  
  acc[i] = 1 - mis_classification
  
  #Confusion Matrix
  
  con_mat = table(data = results,reference = answers)
  
  fp_rate[i] = con_mat[2]/(nrow(train)-smp_size)
  
  fn_rate[i] = con_mat[3]/(nrow(train)-smp_size)
  
  p_bar$step()
  
}

#test_data[13] = lapply(test_data[13], normal)

mean(acc)

pred_test = predict(object = model,test[,-c(1)],type = "prob")

subm_2 = data.frame(test$enrollee_id,pred_test)

names(subm_2)[] = c("enrollee_id","target")

write.csv(x = subm_2,file = "sub6.csv")
