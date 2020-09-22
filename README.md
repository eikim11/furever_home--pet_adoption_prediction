![Project Image](https://github.com/eikim11/furever_home--pet_adoption_prediction/blob/master/img/title.png)

---

### Table of Contents

- [Description](#description)
- [How To Use](#how-to-use)
- [References](#references)
- [Author Info](#author-info)

---

## Description

![Background Info](https://github.com/eikim11/furever_home--pet_adoption_prediction/blob/master/img/annual_stats.png)

The original dataset contains about 20,000 rows describing each animal that enters an animal shelter in Sonoma County, CA from August, 2013 to September, 2020. It also contains 25 columns that include animal name, type, breed color, sex, size, approx. date of birth, outcome date, days in shelter, outcome type, intake/outcome condition, jurisdiction, location, date, etc.

Animal shelters across the nation are working hard to care for these companion friends to find their fur-ever homes. In order for them to be able to operate smoothly within capacity and to minimize animal euthanasia (unless absolutely necessary), I decided to examine the data to accomplish the following:

> Part 1. Build and optimize a classification model to predict a given animal's outcome type - adoption, return to owner, transfer, and euthanize.

> Part 2. Build and optimize a regression model to predict the number of days a given animal will spend at the shelter.

### Clean Data

First, I created an "Age" column using the approx. date of birth and intake date. About 24% of the animals did not have a date of birth value, for which I replaced with the median age. Whether an animal was fixed (spayed or neutered) was indicated in the "Sex" column, which I extracted to a separate boolean column, "Fixed", in order to have the Sex column only indicate either male or female.  Some of the categorial values were redundant but different (e.g. "DOMESTIC SH" vs. "SH DOMESTIC", "Return to owner" vs. "RTOS"). These were consolidated to group the same type together. The "Breed" and "Color" columns often had more than a single value, as there are a lot of mix breeds and colors. These values were seperated into 2 columns - Breed_1 & Breed_2, and Color_1 & Color_2. 

Then dummies were created, leaving the final dataset with about 550 columns. Running `clean_data.py` will load the raw dataset, clean, and save the final csv files into the data folder, which will be separated into "reg" for regression and "clf" for classifier.

### EDA

First, I looked at the adoption distribution among different days of the week. As expected, Saturday (but not Sunday, perhaps they are closed) had the highest number of adoptions, followed by Tuesday and then Friday. 
![adoptions by dayofweek](https://github.com/eikim11/furever_home--pet_adoption_prediction/blob/master/img/num_adoptions_by_dow.png?raw=true)

Next, animals that are younger seem to be more likely to be adopted. Below is the density distribution of the animals' age by each of the 4 outcome types. 

![age density disribution by outcome type](https://github.com/eikim11/furever_home--pet_adoption_prediction/blob/master/img/Age_distro_per_outcome_type.png?raw=true)

And here is the average age distribution among the different outcome types. Animals that get adopted have the lowest average age at just under 2 years, while animals that get returned to owners have the highest. Perhaps because people in general look for their lost companion animals no matter how old they are.

![average age by outcome type](https://github.com/eikim11/furever_home--pet_adoption_prediction/blob/master/img/avg_age_by_outcome_type.png?raw=true)

In terms of predicting the outcome type, for the purpose of this project, the two most important classes will be "Adoption" followed by "Euthanized". The reason why I consider "Adoption" to be more important is that optimizing a better adoption rate will consequently help reduce the number of euthanasia caused by capacity issues. 

Before jumping into how I built a classification model to predict outcome types, here are some additional statistics regarding the days spent in shelter and outcome by animal type. There were 3 unique values for the animal type column - Dogs, Cats, and Other (includes chickens, rabbits, goats, donkeys, etc.)

![avg number of days in shelter](https://github.com/eikim11/furever_home--pet_adoption_prediction/blob/master/img/avg_days_spent.png?raw=true)
Animals that are returned to owners spend on average 3.4 days in shelter, while animals that are adopted spend on average 31.4 days in shelter. Assuming that "Transfer" is mostly based on the flexibility and capacity of neighboring shelters, it will be tough to determine what characteristics of an animal makes it most likely to be transferred.

![outcome type by intake condition](https://github.com/eikim11/furever_home--pet_adoption_prediction/blob/master/img/outcome_type_by_intake_condition.png?raw=true)
This one here is an interesting one. Under "Euthanize", you can see that there is a noticeable chunk of animals that get euthanized not due to their health condition. This leads me to assume that healthy animals get euthanized due to the shelter's capacity issues, which is what I hope to mitigate using my models.

![outcome by animal type](https://github.com/eikim11/furever_home--pet_adoption_prediction/blob/master/img/outcome_type_hist_by_type.png?raw=true)
The animal types seem fairly balanced in all outcomes except for "Return (to owner)". This may be due to the fact that dogs get lost more frequently than cats, many of which are indoor cats.


### Part 1. Classification Model

To predict a given animal's outcome type - Euthanize (0), Transfer (1), Return (2), and Adoption (3) - I realized that XG Boost Classifier yielded the best results for Class 3 (Adoption) and Class 0 (Euthanize). The main evaluation metric used was recall (TP / (TP + FN)), which you can read more about here: ![Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall). The 4 outcome classes were not balanced, but the model performed better for our most important classes WITHOUT oversampling via SMOTE. The final XG Boost Classifier after finding the optimal parameters via GridSearchCV resulted in the following recall scores:

- Class 0 (Euthanize): 0.764

- Class 1 (Transfer): 0.601

- Class 2 (Return): 0.843

- Class 3 (Adoption): **0.864**

### Part 2. Regression Model

Next, to predict the number of days a given animal will spend at the shelter, I built a few different regressor models - Random Forest, Gradient Boosting, and XG Boost Regressor. Similar to the classification model, XG Boost Regressor yielded the best result here too. The main evaluation metric used was Mean Absolute Error (MAE). The reason why I did not use squared error metrics such as MSE or RMSE is to not penalize outliers too harshly. 

First, I calculated the baseline MAE, which was 16.33. The final XG Boost model gave an MAE of 11.29, which was a 31% decrease from the baseline value.

### Conclusion

As for the classification model in Part 1, the recall score for Class 1 (Transfer) was low as expected. As mentioned before, this may be because transfer depends more on the capacity and flexibility of neighboring shelters rather than specific animal characteristics. However, our model is able to correctly classify the Adoption class (3) over 86% of the time. 

As for the regression model in Part 2, my XG Boost Regressor model can predict the number of days an animal will spend at the shelter with over 30% less mean absolute error compared to the baseline metric.

Here are the top 5 most important features in each of my models excluding specific breeds:

![Top 5 features](https://github.com/eikim11/furever_home--pet_adoption_prediction/blob/master/img/Top5.png?raw=true)

Age, intake condition, and intake date have a significant influence in both models, which was expected. However, an animal that is fixed and has a name are also more likely to be adopted than those that do not. 

In order to optimize the adoption rate, thus reducing the number of unnecessary euthanasia, the shelter can make sure that the animals are fixed and given a name. Also, featuring the most adoptable pets in their marketing efforts may also increase the adoption rate and free up more capacity at the shelter. 

#### Technologies

- xgboost
- sklearn
- imblearn

[Back To The Top](#Table-of-Contents)

---

## How To Use
Running `python clean_data.py` will load the dataset, clean, create dummies, and save the final csv's into the data folder. After that, you can run `python run_models.py` which will load the cleaned dataset, instantiate models, train, predict, and give results.

---

## References

Source: https://data.sonomacounty.ca.gov/widgets/924a-vesw

[Back To The Top](#Table-of-Contents)

---

## Author Info

- email - [edward.kim9280@gmail.com](edward.kim9280@gmail.com)
- LinkedIn - [Edward Kim](https://www.linkedin.com/in/edwardkim11/)

[Back To The Top](#Table-of-Contents)
