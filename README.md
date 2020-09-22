![Project Image](https://github.com/eikim11/furever_home--pet_adoption_prediction/blob/master/img/title.png)

---

### Table of Contents

- [Description](#description)
- [How To Use](#how-to-use)
- [References](#references)
- [License](#license)
- [Author Info](#author-info)

---

## Description

![Background Info](https://github.com/eikim11/furever_home--pet_adoption_prediction/blob/master/img/annual_stats.png)

The original dataset contains about 20,000 rows describing each animal that enters an animal shelter in Sonoma County, CA from August, 2013 to September, 2020. It also contains 25 columns that include animal name, type, breed color, sex, size, approx. date of birth, outcome date, days in shelter, outcome type, intake/outcome condition, jurisdiction, location, date, etc.

As animal shelters across the nation are working hard to care for these companion friends to find their fur-ever homes, I decided to examine the data and accomplish the following:

> Part 1. Build and optimize a classification model to predict a given animal's outcome type - adoption, return to owner, transfer, and euthanize.
> Part 2. Build and optimize a regression model to predict the number of days a given animal will spend at the shelter.

#### Clean Data

First, I created an "Age" column using the approx. date of birth and intake date. About 24% of the animals did not have a date of birth value, for which I replaced with the median age. Whether an animal was fixed (spayed or neutered) was indicated in the "Sex" column, which I extracted to a separate boolean column, "Fixed", in order to have the Sex column only indicate either male or female.  Some of the categorial values were redundant but different (e.g. "DOMESTIC SH" vs. "SH DOMESTIC", "Return to owner" vs. "RTOS"). These were consolidated to group the same type together. The "Breed" and "Color" columns often had more than a single value, as there are a lot of mix breeds and colors. These values were seperated into 2 columns - Breed_1 & Breed_2, and Color_1 & Color_2. 

Then dummies were created, leaving the final dataset with about 550 columns. Running `clean_data.py` will load the raw dataset, clean, and save the final csv files into the data folder, which will be separated into "reg" for regression and "clf" for classifier.

#### EDA

First, I looked at the adoption distribution among different days of the week. As expected, Saturday (but not Sunday, perhaps they are closed) had the highest number of adoptions, followed by Tuesday and then Friday. 
![adoptions by dayofweek](https://github.com/eikim11/furever_home--pet_adoption_prediction/blob/master/img/num_adoptions_by_dow.png?raw=true)

Next, animals that are younger seem to be more likely to be adopted. Below is the density distribution of the animals' age by each of the 4 outcome types. 

Include Age graphs

Include days in shelter by outcome

Include outcome by animal type

#### Part 1. Classification Model

Describe XG Boost Classifier

#### Part 2. Regression Model

Describe XG Boost Regressor

#### Conclusion

Evaluation metric
Fix top 5 importances

#### Technologies

- Technology 1
- Technology 2

[Back To The Top](#read-me-template)

---

## How To Use

#### Installation



#### API Reference

```html
    <p>dummy code</p>
```
[Back To The Top](#read-me-template)

---

## References
[Back To The Top](#read-me-template)

---

## License

MIT License

Copyright (c) [2017] [James Q Quick]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

[Back To The Top](#read-me-template)

---

## Author Info

- Twitter - [@jamesqquick](https://twitter.com/jamesqquick)
- Website - [James Q Quick](https://jamesqquick.com)

[Back To The Top](#read-me-template)
