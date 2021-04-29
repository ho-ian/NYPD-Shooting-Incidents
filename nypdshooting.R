### MOTIVATING MACHINE LEARNING PREDICTION ###
## BASED ON THE GIVEN INFORMATION, CAN WE PREDICT THE VICTIM'S RACE? ##


## DATA SET SOURCES BELOW ##

#https://datagious.com/datasets/
#https://catalog.data.gov/dataset/nypd-shooting-incident-data-historic


## NECESSARY LIBRARIES TO RUN THE PROJECT ##

# note: if it doesn't run the first time, try rerunning it again once the packages are installed
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(stringr)) install.packages("stringr")
if(!require(scales)) install.packages("scales")
if(!require(caret)) install.packages("caret")
if(!require(rpart)) install.packages("rpart")
if(!require(randomForest)) install.packages("randomForest")
if(!require(dplyr)) install.packages("dplyr")
if(!require(ggridges)) install.packages("ggridges")
if(!require(lubridate)) install.packages("lubridate")
if(!require(leaflet)) install.packages("leaflet")
if(!require(mapview)) install.packages("mapview"); webshot::install_phantomjs()
if(!require(kableExtra)) install.packages("kableExtra")

options(digits = 5)

## DOWNLOADING AND IMPORTING THE DATASET ##

dl <- tempfile()
download.file("https://data.cityofnewyork.us/api/views/833y-fsy8/rows.csv?accessType=DOWNLOAD", dl)
dat <- read_csv(dl)
file.remove(dl)
rm(dl)

## DATA CLEANING SECTION ##

# seeing and counting which columns have NA's
sum(is.na(dat$BORO))
sum(is.na(dat$PRECINCT))
sum(is.na(dat$JURISDICTION_CODE))
sum(is.na(dat$LOCATION_DESC))
sum(is.na(dat$STATISTICAL_MURDER_FLAG))
sum(is.na(dat$PERP_AGE_GROUP))
sum(is.na(dat$PERP_SEX))
sum(is.na(dat$PERP_RACE))
sum(is.na(dat$VIC_AGE_GROUP))
sum(is.na(dat$VIC_SEX))
sum(is.na(dat$VIC_RACE))

# proportion of na's to the number of rows in the data set
sum(is.na(dat$JURISDICTION_CODE))/nrow(dat)
sum(is.na(dat$LOCATION_DESC))/nrow(dat)
sum(is.na(dat$PERP_AGE_GROUP))/nrow(dat)
sum(is.na(dat$PERP_SEX))/nrow(dat)
sum(is.na(dat$PERP_RACE))/nrow(dat)

# we can remove the rows in jurisdiction code with na because it has a small enough
# proportion to the data set, however, the columns location_desc, perp_age_group,
# perp_sex and perp_race have too high of a proportion to the dataset. I think
# it is most advisable to NOT INCLUDE them in the machine learning portion because
# they would not be reliable predictors.

# removing na rows in jurisdiction code
dat <- dat[!is.na(dat$JURISDICTION_CODE),]

# make date into correct type
dat$OCCUR_DATE <- date(mdy(dat$OCCUR_DATE))

# remove a unnecessary columns
dat <- dat %>% select(-INCIDENT_KEY,
                      -Lon_Lat,
                      -X_COORD_CD,
                      -Y_COORD_CD,
                      -LOCATION_DESC,
                      -PERP_AGE_GROUP,
                      -PERP_SEX,
                      -PERP_RACE)

# make vic_race factor
dat$VIC_RACE <- factor(dat$VIC_RACE)


## CREATING WORKING AND VALIDATION SETS ##
nrow(dat)

y <- dat$VIC_RACE
set.seed(212, sample.kind = "Rounding")
validation_index <- createDataPartition(y, times = 1, p = 0.15, list = FALSE)
validation <- dat %>% slice(validation_index)
dat <- dat %>% slice(-validation_index)


## DATA EXPLORATION SECTION ##

# structure of the data
str(dat)
# head of the data frame
head(dat)
# number of rows in the data frame
nrow(dat)
# column names in the data frame
colnames(dat)

# unique values of various columns 
unique(dat$BORO)
unique(dat$PRECINCT)
unique(dat$JURISDICTION_CODE)
# victim column unique values
unique(dat$VIC_AGE_GROUP)
unique(dat$VIC_SEX)
unique(dat$VIC_RACE)

# earliest and latest shooting incidents
min(dat$OCCUR_DATE)
max(dat$OCCUR_DATE)

# proportion of shooting incidents resulting in death
mean(dat$STATISTICAL_MURDER_FLAG)

# observing counts of incidents via grouping ...

# by borough
boro_incidents <- dat %>%
  group_by(BORO) %>%
  summarize(count = n(), 
            prop = count/nrow(dat),
            prop_death = mean(STATISTICAL_MURDER_FLAG))

boro_incidents %>% arrange(desc(count))

# by precinct
precinct_incidents <- dat %>%
  group_by(PRECINCT) %>%
  summarize(count = n(), 
            prop = count/nrow(dat),
            prop_death = mean(STATISTICAL_MURDER_FLAG))

precinct_incidents %>% arrange(desc(count))

# by jurisdiction code
jurisdiction_incidents <- dat %>%
  group_by(JURISDICTION_CODE) %>%
  summarize(count = n(), 
            prop = count/nrow(dat),
            prop_death = mean(STATISTICAL_MURDER_FLAG))

jurisdiction_incidents %>% arrange(desc(count))

# by vic_age_group
victim_age_incidents <- dat %>%
  group_by(VIC_AGE_GROUP) %>%
  summarize(count = n(), 
            prop = count/nrow(dat),
            prop_death = mean(STATISTICAL_MURDER_FLAG))

victim_age_incidents %>% arrange(desc(count))

# by vic_sex
victim_sex_incidents <- dat %>%
  group_by(VIC_SEX) %>%
  summarize(count = n(), 
            prop = count/nrow(dat),
            prop_death = mean(STATISTICAL_MURDER_FLAG))

victim_sex_incidents %>% arrange(desc(count))

# by vic_race
victim_race_incidents <- dat %>%
  group_by(VIC_RACE) %>%
  summarize(count = n(), 
            prop = count/nrow(dat),
            prop_death = mean(STATISTICAL_MURDER_FLAG))

victim_race_incidents %>% arrange(desc(count))


## DATA VISUALIZATION ##

# the following are plots related to the data exploration insights we gained earlier

# plot of shooting incidents over occur date
dat %>%
  ggplot(aes(x = OCCUR_DATE)) + 
  geom_histogram(bins = 48) +
  xlab("Occur Date") +
  ggtitle("Distribution of Shooting Incidents by Date of Occurence")

# plot of shooting incidents over occur time
dat %>% 
  ggplot(aes(OCCUR_TIME)) + 
  geom_histogram(bins = 48) + 
  xlab("Occur Time") +
  ggtitle("Distribution of Shooting Incidents by Time of Occurrence")
# as it stands, the histogram does not appear to be normally distributed
# but if we take a closer look at the x axis, it would seem that most shooting
# incidents took place around 12 am. let's transform the data slightly to visualize
# it better with another plot

# here is a function to split the times across two dates in order to visualize the 
# occurrence time better. i found this function from ...
# https://stackoverflow.com/questions/31039228/creating-ggplot2-histogram-with-times-around-midnight
justtime <- function(x, split=12) {
  h <- as.numeric(strftime(x, "%H"))
  y <- as.POSIXct(paste(ifelse(h<split, "2015-01-02","2015-01-01"),strftime(x, "%H:%M:%S")))
}

# distribution of shooting incidents centered around 12 am
dat %>% 
  mutate(time = justtime(OCCUR_TIME)) %>%
  ggplot(aes(time)) + 
  geom_histogram(bins = 48) + 
  scale_x_datetime(labels = function(x) format(x, format = "%H:%M")) + 
  xlab("Time") +
  ylab("Count of Shooting Incidents") +
  ggtitle("Distribution of Shooting Incidents by Time of Occurrence")

# plot of shooting incidents by borough
boro_incidents %>%
  ggplot(aes(count, y = reorder(BORO, -count), fill = BORO)) + 
  geom_bar(stat = "identity") +
  ylab("Borough") +
  xlab("Number of Shooting Incidents") +
  ggtitle("Number of Shooting Incidents Grouped By Borough") +
  theme(legend.position = "none")

# plot of shooting incidents by precinct
precinct_incidents %>%
  ggplot(aes(PRECINCT, count, fill = PRECINCT)) +
  geom_bar(stat = "identity") +
  ylab("Number of Shooting Incidents") +
  xlab("Precinct") +
  ggtitle("Number of Shooting Incidents Grouped By Precinct") +
  theme(axis.text.x = element_blank()) +
  guides(fill = guide_legend(title = "Precinct"))

# plot of shooting incidents by jurisdiction code
jurisdiction_incidents %>%
  ggplot(aes(count, y = reorder(JURISDICTION_CODE, -count), fill = JURISDICTION_CODE)) +
  geom_bar(stat = "identity") +
  ylab("Jurisdiction Code") +
  xlab("Number of Shooting Incidents") +
  ggtitle("Number of Shooting Incidents Grouped By Jurisdiction Code") +
  theme(legend.position = "none")

# plot of shooting incidents by vic_age_group
victim_age_incidents %>%
  ggplot(aes(VIC_AGE_GROUP, y = count, fill = VIC_AGE_GROUP)) +
  geom_bar(stat = "identity") +
  ylab("Number of Shooting Incidents") +
  xlab("Victim Age Group") +
  ggtitle("Number of Shooting Incidents Grouped By Victim Age Group") +
  theme(legend.position = "none")

# plot of shooting incidents by vic_sex
victim_sex_incidents %>%
  ggplot(aes(reorder(VIC_SEX, -count), y = count, fill = VIC_SEX)) +
  geom_bar(stat = "identity") +
  ylab("Number of Shooting Incidents") +
  xlab("Victim Sex") +
  ggtitle("Number of Shooting Incidents Grouped By Victim Sex") +
  theme(legend.position = "none")

# plot of shooting incidents by vic_race
victim_race_incidents %>% 
  ggplot(aes(count, y = reorder(VIC_RACE, - count), fill = VIC_RACE)) +
  geom_bar(stat = "identity") +
  ylab("Victim Race") +
  xlab("Number of Shooting Incidents") +
  ggtitle("Number of Shooting Incidents Grouped By Victim Race") +
  theme(legend.position = "none")

# notes: from these plots we see that there are multiple key variables
# that have a higher shooting incident rate. What we are interested in now 
# is gaining insight into various conditional probabilities as well as
# taking a look at columns involving longitude, latitude,  we will first
# plot these out to see what we get

# plot of shooting incidents using longitude and latitude
dat %>% ggplot(aes(x = Longitude, y = Latitude)) + geom_point()

# the following are plots of shooting incidents on a map of nyc

# plotting nyc shooting incidents on a map of nyc by cluster
cluster <- leaflet(dat) %>%
  addTiles() %>%
  addMarkers(clusterOptions = markerClusterOptions()) %>%
  setView(-74.00, 40.71, zoom = 10) %>%
  addProviderTiles("CartoDB.VoyagerLabelsUnder")
cluster
# include the following piece of code in the rmarkdown file to save the image temporarily
mapshot(cluster, file = paste0(getwd(), "/nycshooting.png"))

# plotting nyc shooting incidents coloured by borough
borocol <- colorFactor(palette = "Set1", dat$BORO)

borough <- leaflet(dat) %>%
  addTiles() %>%
  addCircleMarkers(lng = ~Longitude, lat = ~Latitude,
                   color = ~borocol(dat$BORO), weight = 1,
                   opacity = 1, radius = 0.1) %>%
  setView(-74.00, 40.71, zoom = 10) %>%
  addProviderTiles("CartoDB.VoyagerLabelsUnder") %>%
  addLegend(pal = borocol, values = dat$BORO, position = "topleft")

borough

# plotting nyc shooting incidents coloured by precinct
preccol <- colorFactor(palette = hue_pal()(length(unique(dat$PRECINCT))), dat$PRECINCT)

precinct <- leaflet(dat) %>%
  addTiles() %>%
  addCircleMarkers(lng = ~Longitude, lat = ~Latitude,
                   color = ~preccol(dat$PRECINCT), weight = 1,
                   opacity = 1, radius = 0.1) %>%
  setView(-74.00, 40.71, zoom = 10) %>%
  addProviderTiles("CartoDB.VoyagerLabelsUnder")

precinct
# compare this plot with the earlier plot of number of shooting incidents grouped
# by precinct and now we can visually see which locations have more shooting incidents
# and which precinct they clearly belong to

# lets take a deeper dive into the victim's age, sex, and race by plotting the 
# shooting incidents in a single borough and making the age sex and race distinct
# in the plot

# plot of shooting incidents in staten island by age
agecol <- colorFactor(palette = "Set1", dat$VIC_AGE_GROUP)
age <- leaflet(dat[dat$BORO == "STATEN ISLAND",]) %>%
  addTiles() %>%
  addCircles(lng = ~Longitude, lat = ~Latitude,
                   radius = 5, opacity = 1,
                   color = ~agecol(dat$VIC_AGE_GROUP)) %>%
  setView(-74.15, 40.58, zoom = 12) %>%
  addProviderTiles("CartoDB.VoyagerLabelsUnder") %>%
  addLegend(pal = agecol, values = dat$VIC_AGE_GROUP, position = "topleft")

age

# plot of shooting incidents in staten island by sex
sexcol <- colorFactor(palette = "Set1", dat$VIC_SEX)
sex <- leaflet(dat[dat$BORO == "STATEN ISLAND",]) %>%
  addTiles() %>%
  addCircles(lng = ~Longitude, lat = ~Latitude,
             radius = 5, opacity = 1,
             color = ~sexcol(dat$VIC_SEX)) %>%
  setView(-74.15, 40.58, zoom = 12) %>%
  addProviderTiles("CartoDB.VoyagerLabelsUnder") %>%
  addLegend(pal = sexcol, values = dat$VIC_SEX, position = "topleft")

sex

# plot of shooting incidents in staten island by race
racecol <- colorFactor(palette = "Set1", dat$VIC_RACE)
race <- leaflet(dat[dat$BORO == "STATEN ISLAND",]) %>%
  addTiles() %>%
  addCircles(lng = ~Longitude, lat = ~Latitude,
             radius = 5, opacity = 1,
             color = ~racecol(dat$VIC_RACE)) %>%
  setView(-74.15, 40.58, zoom = 12) %>%
  addProviderTiles("CartoDB.VoyagerLabelsUnder") %>%
  addLegend(pal = racecol, values = dat$VIC_RACE, position = "topleft")

race

# these three plots visually demonstrate that the shooting incidents largely
# involve the 18-24 and 25-44 age groups, male, and of the black race.

## PROBABILITY EXPLORATION ##

# since we're interesting in predicting victim race from the various columns that 
# exist in the dataset, we formulate our conditional probabilities with that idea
# in mind

# probability density plot of shooting incident over occur hour
dat %>%
  mutate(hour = ifelse(hour(OCCUR_TIME) > 12, hour(OCCUR_TIME) - 24, hour(OCCUR_TIME))) %>%
  group_by(hour) %>%
  ggplot(aes(x = hour)) + 
  geom_density() +
  xlab("Hour") +
  ylab("Probability") +
  ggtitle("Density Plot of Shooting Incidents over Occurrence Hour")

# probability density plot of shooting incidents per victim race over occur hour
dat %>%
  mutate(hour = ifelse(hour(OCCUR_TIME) > 12, hour(OCCUR_TIME) - 24, hour(OCCUR_TIME))) %>%
  group_by(hour, VIC_RACE) %>%
  ggplot(aes(x = hour, y = VIC_RACE)) +
  geom_density_ridges(aes(fill = VIC_RACE), alpha = 0.55) +
  xlab("Hour") +
  ylab("Probability") +
  ggtitle("Density Plot of Shooting Incidents by Victim Race over Occurence Hour") +
  theme(legend.position = "none")

# probabilities of victim race grouped by victim age group
race_by_age <- dat %>%
  group_by(VIC_AGE_GROUP, VIC_RACE) %>%
  summarize(count = n()) %>%
  mutate(prob = count / sum(count))

# top victim age group and victim race
race_by_age %>%
  arrange(desc(count)) %>%
  top_n(10)

# counts of shooting incidents grouped by victim age group and victim race
race_by_age %>% 
  ggplot(aes(VIC_AGE_GROUP, count, fill = VIC_RACE)) +
  geom_bar(stat = "identity") +
  xlab("Victim Age Group") +
  ylab("Shooting Incidents") +
  ggtitle("Shooting Incidents Grouped By Victim Age Group and Victim Race") +
  guides(fill = guide_legend(title = "Victim Race"))

# probabilities of victim race grouped by borough
race_by_boro <- dat %>%
  group_by(BORO, VIC_RACE) %>%
  summarize(count = n()) %>%
  mutate(prob = count / sum(count))

# in all 5 boroughs, black people are most likely involved in shooting incidents
race_by_boro %>% 
  arrange(desc(prob)) %>%
  top_n(10)

# here we can see the probabilities visually through a stacked bar plot
race_by_boro %>%
  ggplot(aes(BORO, prob, fill = VIC_RACE)) +
  geom_bar(stat = "identity", position = "stack") +
  xlab("Borough") +
  ylab("Probability of Victim Race") +
  ggtitle("Probabilities of Victim Races in each Borough") +
  guides(fill = guide_legend(title = "Victim Race"))

# probabilities of murder given the victim's race
race_by_murder <- dat %>%
  group_by(VIC_RACE, STATISTICAL_MURDER_FLAG) %>%
  summarize(count = n()) %>%
  mutate(prob = count / sum(count))

# as we learned earlier, murder happens about 20% of the time, and it appears that
# no american indian/alaskan native has ever been murdered in a shooting incident
race_by_murder %>%
  arrange(desc(prob))

# plot of murder likelihood given victim's race
race_by_murder %>%
  ggplot(aes(prob, VIC_RACE, fill = STATISTICAL_MURDER_FLAG)) +
  geom_bar(stat = "identity") +
  xlab("Probability of Murder") +
  ylab("Victim Race") +
  ggtitle("Likelihood of Murder given the Victim's Race") +
  guides(fill = guide_legend(title = "Murder"))


# notes: from what we gather in this section, we see that various variables
# present potential predictors for finding the victim's race. Borough, 
# murder statistic, perpetrator age, and perpetrator race gave promising
# probabilities. Example, if the perp age is 65+, there is 0% chance that the victim
# race is asian / pacific islander. If the perp race is white, the victim race
# is most likely white as well.


## CREATING TRAINING AND TEST SETS ##
y <- dat$VIC_RACE
set.seed(718, sample.kind = "Rounding")
test_index <- createDataPartition(y, times = 1, p = 0.2, list = FALSE)
train_set <- dat %>% slice(-test_index)
test_set <- dat %>% slice(test_index)

## MACHINE LEARNING MODELLING ##

# naive model, predict the highest probability victim race
naive_guess <- train_set %>% 
  group_by(VIC_RACE) %>%
  summarize(count = n()) %>%
  filter(count == max(count)) %>%
  pull(VIC_RACE)

y_naive <- test_set %>%
  mutate(y_hat = naive_guess) %>%
  pull(y_hat)

naive_acc <- confusionMatrix(y_naive, reference = test_set$VIC_RACE)$overall["Accuracy"]
naive_acc

# training the data using a regression tree and observing the accuracy of the model
start.time <- Sys.time()
fit_rt <- train(VIC_RACE ~ ., data = train_set, method = "rpart")
end.time <- Sys.time()
end.time - start.time

start.time <- Sys.time()
y_rt <- predict(fit_rt, newdata = test_set)
end.time <- Sys.time()
end.time - start.time

plot(fit_rt$finalModel, margin = 0.1)
text(fit_rt$finalModel, cex = 0.75)

rt_acc <- confusionMatrix(y_rt, reference = test_set$VIC_RACE)$overall["Accuracy"]
rt_acc

# training the using the random forest classifier
start.time <- Sys.time()
fit_rf <- train(VIC_RACE ~ ., data = train_set, method = "rf", allowParallel = TRUE)
end.time <- Sys.time()
end.time - start.time

start.time <- Sys.time()
y_rf <- predict(fit_rf, newdata = test_set)
end.time <- Sys.time()
end.time - start.time

rf_acc <- confusionMatrix(y_rf, reference = test_set$VIC_RACE)$overall["Accuracy"]
rf_acc

plot(fit_rf)

# training the data using k-nearest neighbours
start.time <- Sys.time()
fit_knn <- train(VIC_RACE ~ Latitude + Longitude, data = train_set, method = "knn")
end.time <- Sys.time()
end.time - start.time

start.time <- Sys.time()
y_knn <- predict(fit_knn, newdata = test_set) %>% as.factor()
end.time <- Sys.time()
end.time - start.time

knn_acc <- confusionMatrix(y_knn, reference = test_set$VIC_RACE)$overall["Accuracy"]
knn_acc

plot(fit_knn)

# it appears that there can be improvements if we increase the # of neighbours

# fitting a naive_bayes model
start.time <- Sys.time()
fit_nb <- train(VIC_RACE ~ Longitude + Latitude, data = train_set, method = "naive_bayes")
end.time <- Sys.time()
end.time - start.time

start.time <- Sys.time()
y_nb <- predict(fit_nb, newdata = test_set)
end.time <- Sys.time()
end.time - start.time

nb_acc <- confusionMatrix(y_nb, reference = test_set$VIC_RACE)$overall["Accuracy"]
nb_acc

plot(fit_nb)

# fitting a multinomial regression model
start.time <- Sys.time()
fit_mln <- train(VIC_RACE ~ ., data = train_set, method = "multinom", MaxNWts = 1000000)
end.time <- Sys.time()
end.time - start.time

start.time <- Sys.time()
y_mln <- predict(fit_mln, newdata = test_set) %>% as.factor()
end.time <- Sys.time()
end.time - start.time

mln_acc <- confusionMatrix(y_mln, reference = test_set$VIC_RACE)$overall["Accuracy"]
mln_acc

plot(fit_mln)

acc_res <- data.frame(Models = c("Naive Model", 
                                 "Decision Tree",
                                 "Random Forest",
                                 "K-Nearest Neighbours",
                                 "Naive Bayes",
                                 "Multinomial Regression"),
                      Accuracy = c(naive_acc,
                                   rt_acc,
                                   rf_acc,
                                   knn_acc,
                                   nb_acc,
                                   mln_acc))

acc_res %>% knitr::kable()

# as we can see all three of our models perform relatively similar with random forest
# being the most accurate so far with 75% accuracy. let's use cross validation
# to try and fine tune each of these algorithms to see which one performs best

## CROSS VALIDATION ##
control <- trainControl(method = "cv", number = 10, p = .9)

# decision tree
start.time <- Sys.time()
fit_rt <- train(VIC_RACE ~ .,
                data = train_set,
                method = "rpart",
                tuneGrid = data.frame(cp = seq(0.0, 0.2, len = 50)),
                trControl = control)
end.time <- Sys.time()
end.time - start.time

start.time <- Sys.time()
y_rt <- predict(fit_rt, newdata = test_set)
end.time <- Sys.time()
end.time - start.time

plot(fit_rt)

rt_acc <- confusionMatrix(y_rt, reference = test_set$VIC_RACE)$overall["Accuracy"]
rt_acc

# randomforest 
start.time <- Sys.time()
fit_rf <- train(VIC_RACE ~ .,
                data = train_set,
                method = "rf",
                tuneGrid = data.frame(mtry = seq(2,24,2)),
                trControl = control,
                allowParallel = TRUE)
end.time <- Sys.time()
end.time - start.time

start.time <- Sys.time()
y_rf <- predict(fit_rf, newdata = test_set)
end.time <- Sys.time()
end.time - start.time

rf_acc <- confusionMatrix(y_rf, reference = test_set$VIC_RACE)$overall["Accuracy"]
rf_acc

plot(fit_rf)

# knn
start.time <- Sys.time()
fit_knn <- train(VIC_RACE ~ Latitude + Longitude,
                 data = train_set,
                 method = "knn",
                 tuneGrid = data.frame(k = seq(3,101,3)),
                 trControl = control)
end.time <- Sys.time()
end.time - start.time

start.time <- Sys.time()
y_knn <- predict(fit_knn, newdata = test_set) 
end.time <- Sys.time()
end.time - start.time

knn_acc <- confusionMatrix(y_knn, reference = test_set$VIC_RACE)$overall["Accuracy"]
knn_acc

plot(fit_knn)

# naive bayes
start.time <- Sys.time()
fit_nb <- train(VIC_RACE ~ Latitude + Longitude,
                data = train_set,
                method = "naive_bayes",
                tuneGrid = expand.grid(laplace = seq(0.1, 10, 0.1),
                                       usekernel = c(TRUE, FALSE),
                                       adjust = c(TRUE, FALSE)),
                trControl = control)
end.time <- Sys.time()
end.time - start.time

start.time <- Sys.time()
y_nb <- predict(fit_nb, newdata = test_set)
end.time <- Sys.time()
end.time - start.time

nb_acc <- confusionMatrix(y_nb, reference = test_set$VIC_RACE)$overall["Accuracy"]
nb_acc

plot(fit_nb)

# multinomial
start.time <- Sys.time()
fit_mln <- train(VIC_RACE ~ .,
                 data = train_set,
                 method = "multinom",
                 tuneGrid = data.frame(decay = seq(0.2, 2, 0.2)),
                 trControl = control,
                 MaxNWts = 1000000)
end.time <- Sys.time()
end.time - start.time

start.time <- Sys.time()
y_mln <- predict(fit_mln, newdata = test_set) %>% as.factor()
end.time <- Sys.time()
end.time - start.time

mln_acc <- confusionMatrix(y_mln, reference = test_set$VIC_RACE)$overall["Accuracy"]
mln_acc

plot(fit_mln)

acc_res <- bind_rows(acc_res,
                     data.frame(Models = c("K-Fold Cross Validated Decision Tree",
                                 "K-Fold Cross Validated Random Forest",
                                 "K-Fold Cross Validated K-Nearest Neighbours",
                                 "K-Fold Cross Validated Naive Bayes",
                                 "K-Fold Cross Validated Multinomial Regression"),
                      Accuracy = c(rt_acc,
                                   rf_acc,
                                   knn_acc,
                                   nb_acc,
                                   mln_acc)))

acc_res %>% knitr::kable()

races <- levels(dat$VIC_RACE)
y_ensemble <- data.frame(rt = y_rt,
                         rf = y_rf,
                         knn = y_knn,
                         nb = y_nb,
                         mln = y_mln)

ensemble <- apply(y_ensemble, 1, function(pred) {
  prob_race <- sapply(races, function(race) {
    mean(pred == race)
  })
  races[which.max(prob_race)]
})

ensemble <- factor(ensemble, levels = levels(test_set$VIC_RACE))
ens_acc <- confusionMatrix(ensemble, reference = factor(test_set$VIC_RACE))$overall["Accuracy"]
ens_acc

acc_res <- bind_rows(acc_res,
                     data_frame(Models = "Ensemble Model",
                                Accuracy = ens_acc))

acc_res %>% knitr::kable()

# ensemble model isnt that great but random forest performs the best so we will
# just use that model to predict the validation set

## FINAL MODEL TRAINING AND TESTING ##
start.time <- Sys.time()
fit_final <- randomForest(VIC_RACE ~ .,
                          data = dat,
                          allowParallel = TRUE,
                          mtry = fit_rf$bestTune[,"mtry"])
end.time <- Sys.time()
end.time - start.time

start.time <- Sys.time()
y_final <- predict(fit_final, newdata = validation)
end.time <- Sys.time()
end.time - start.time

final_acc <- confusionMatrix(y_final, reference = factor(validation$VIC_RACE))$overall["Accuracy"]
final_acc

acc_res <- bind_rows(acc_res,
                     data_frame(Models = "Final Model",
                                Accuracy = final_acc))

acc_res %>% knitr::kable()



