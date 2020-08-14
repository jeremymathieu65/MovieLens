#Loading Libraries
library(tidyverse)
library(dslabs)
library(caret)
library(lubridate)
library(stringr)



#Start of Edx distribution code
##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#End of Edx distribution code



#Beginning of code used to build model

set.seed(1, sample.kind ="Rounding")

#Creating a test index to subset 10% of the edx data set
index <- createDataPartition(y = edx$rating, times = 1, p = 0.10, list = FALSE)

#Defining training set as 90% of the edx data set
train <- edx %>% slice(-index)

#Defining test set as 10% of edx data set
temp_test <- edx %>% slice(index)

#Ensuring movies and users in test set are present in training set
test <- temp_test %>% semi_join(train, by="movieId") %>% semi_join(train, by="userId")

#Adding users and movies filtered from the test set back to the training set
extra <- anti_join(test, temp_test)
train <- rbind(train, extra)
rm(extra, temp_test)

#Defining a vector containing the average ratings for each individual movie
movie_avgs <- train %>% group_by(movieId) %>% summarize(avg = mean(rating), movie_n = n(), genres = genres[1])

#Defining a vector containing the average ratings across all movies in each individual genre
genre_avgs <- train %>% group_by(genres) %>% summarize(genre_avg = mean(rating), genre_n = n())

#Defining a vector containing average movie ratings for movies classified as 'scarce' (having less than 15 ratings)
#Redefining the average movie rating of each 'scarce' movie as the average rating of all movies of its genre
movie_avgs_scarce <- movie_avgs %>% filter(movie_n <= 15) %>% left_join(genre_avgs, by="genres") %>% mutate(avg = genre_avg) %>% select(movieId, avg)

#Defining a vector containing average movie ratings for movies classified as 'non-scarce' (having more than 15 ratings)
movie_avgs_final <- movie_avgs %>% filter(movie_n > 15) %>% select(movieId, avg)

#Defining a vector containing average ratings for both 'non-scarce' and 'scarce' movies
#'scarce' movies have average genre ratings set as average rating whereas 'non-scarce' have average movie ratings set as average ratings
movie_avgs_final <- movie_avgs_final %>% rbind(movie_avgs_scarce)

#Defining a vector containing the average bias for each user regularized with lambda = 5
user_effects <- train %>% left_join(movie_avgs_final, by="movieId") %>% group_by(userId) %>% summarize(B_u_r = sum(rating - avg)/(n() + 5))

#Defining a vector containing the average bias for each genre regularized with lambda = 5
genre_effects <- train %>% left_join(movie_avgs_final, by="movieId") %>% left_join(user_effects, by="userId") %>% group_by(genres) %>% summarize(B_g_r = sum(rating - avg - B_u_r)/(n() + 5))

#Making predictions on the test set using average ratings and regularized user and genre biases
#Genre biases for movies classified as 'scarce' set to 0 as their average ratings are already derived from their average genre ratings
preds <- test %>% left_join(movie_avgs_final, by="movieId") %>% left_join(user_effects, by="userId") %>% left_join(genre_effects, by="genres") %>% mutate(B_g_r = ifelse(movieId %in% movie_avgs_scarce$movieId, 0, B_g_r), pred = avg + B_u_r + B_g_r) %>% pull(pred)

#Reporting the final RMSE obtained on the test set
RMSE(preds, test$rating)

#End of code used to build model



#Beginning of code used to build final model using the entire edx dataset without partitioning

#Defining a vector containing the average ratings for each individual movie
movie_avgs <- edx %>% group_by(movieId) %>% summarize(avg = mean(rating), movie_n = n(), genres = genres[1])

#Defining a vector containing the average ratings across all movies in each individual genre
genre_avgs <- edx %>% group_by(genres) %>% summarize(genre_avg = mean(rating), genre_n = n())

#Defining a vector containing average movie ratings for movies classified as 'scarce' (having less than 15 ratings)
#Redefining the average movie rating of each 'scarce' movie as the average rating of all movies of its genre
movie_avgs_scarce <- movie_avgs %>% filter(movie_n <= 15) %>% left_join(genre_avgs, by="genres") %>% mutate(avg = genre_avg) %>% select(movieId, avg)

#Defining a vector containing average movie ratings for movies classified as 'non-scarce' (having more than 15 ratings)
movie_avgs_final <- movie_avgs %>% filter(movie_n > 15) %>% select(movieId, avg)

#Defining a vector containing average ratings for both 'non-scarce' and 'scarce' movies
#'scarce' movies have average genre ratings set as average rating whereas 'non-scarce' have average movie ratings set as average ratings
movie_avgs_final <- movie_avgs_final %>% rbind(movie_avgs_scarce)

#Defining a vector containing the average bias for each user regularized with lambda = 5
user_effects <- edx %>% left_join(movie_avgs_final, by="movieId") %>% group_by(userId) %>% summarize(B_u_r = sum(rating - avg)/(n() + 5))

#Defining a vector containing the average bias for each genre regularized with lambda = 5
genre_effects <- edx %>% left_join(movie_avgs_final, by="movieId") %>% left_join(user_effects, by="userId") %>% group_by(genres) %>% summarize(B_g_r = sum(rating - avg - B_u_r)/(n() + 5))

#Making predictions on the test set using average ratings and regularized user and genre biases
#Genre Biases for movies classified as 'scarce' set to 0 as their average ratings are already derived from their average genre ratings
preds <- validation %>% left_join(movie_avgs_final, by="movieId") %>% left_join(user_effects, by="userId") %>% left_join(genre_effects, by="genres") %>% mutate(pred = avg + B_u_r + B_g_r) %>% pull(pred)

#Defining a function to standardize predictions i.e. ensure they are in the range 0.5 <= Y <= 5
standardize_preds <- function(pred){
  #if prediction is lower than lowest possible rating, then redefine it to be the lowest possible, 0.5.
  if (pred < 0.5){
    pred <- 0.5
  }
  #if prediction is higher than the highest possible rating, then redefine it to be the highest possible, 5.
  else if (pred > 5){
    pred <- 5
  }
  #if prediction is within the range of actual predictions, leave it as it is.
  else {
    pred <- pred
  }
  return (pred)
}

#Applying the above defined function on the predicted ratings
preds <- sapply(preds, FUN = "standardize_preds")

#Reporting the final RMSE obtained on the validation set
RMSE(preds, validation$rating)

#End of code used to build final model

