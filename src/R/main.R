library(tidyverse)
library(caret)

message("[INFO] Loading training data...")
train = read.csv("../data/train.csv")
message(paste("[INFO] Train shape:", nrow(train), "rows,", ncol(train), "columns"))

# select features
df = train %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare) %>%
  mutate(
    Sex = ifelse(Sex == "male", 1, 0),
    Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age),
    Fare = ifelse(is.na(Fare), median(Fare, na.rm = TRUE), Fare)
  )

# split data
set.seed(42)
train_index = createDataPartition(df$Survived, p = 0.8, list = FALSE)
train_data = df[train_index, ]
valid_data = df[-train_index, ]

# logistic regression model
model = glm(Survived ~ ., data = train_data, family = binomial)

# predictions on validation split
pred_prob = predict(model, valid_data, type = "response")
pred_class = ifelse(pred_prob > 0.5, 1, 0)

# accuracy
acc = mean(pred_class == valid_data$Survived)
message(sprintf("[RESULT] Train validation accuracy: %.3f", acc))

# count of predicted survivors vs died (validation set)
val_summary <- table(Predicted = pred_class)
message("[RESULT] Train prediction summary:")
print(val_summary)

# predict on test set
message("[INFO] Loading test data...")
test = read.csv("../data/test.csv")
message(paste("[INFO] Test shape:", nrow(test), "rows,", ncol(test), "columns"))

# prepare features
test_prep = test %>%
  select(Pclass, Sex, Age, SibSp, Parch, Fare) %>%
  mutate(
    Sex = ifelse(Sex == "male", 1, 0),
    Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age),
    Fare = ifelse(is.na(Fare), median(Fare, na.rm = TRUE), Fare)
  )

# predictions on test set
test_pred = predict(model, test_prep, type = "response")
pred_labels = ifelse(test_pred > 0.5, 1, 0)

# count of predicted survivors and died
test_summary <- table(Predicted = pred_labels)
message(paste("[RESULT] Test predictions generated:", length(pred_labels)))
message("[RESULT] Test prediction summary:")
print(test_summary)

# save predictions
submission = data.frame(PassengerId = test$PassengerId, Survived = pred_labels)
write.csv(submission, "../data/prediction_R.csv", row.names = FALSE)
message("[INFO] Saved predictions to ../data/prediction_R.csv")
