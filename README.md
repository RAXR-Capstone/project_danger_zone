
<img src="https://github.com/RAXR-Capstone/project_danger_zone/blob/master/work_space/Pictures/danger_banner.png">

# Project Danger Zone
Brought to you by data scientists:
 - Xavier Carter
 - Robert Murphy
 - Anna Vu
 - Ray Zapata
<br>
<br>

In Bexar County during 2020, with a reported 16,780 reported injuries and 200 deaths, there were nearly 50,000 motor vehicle collisions . Using vehicle crash statistics for 2021, our team of four is working to discover the drivers of increased injury rates among motorist with an eye toward providing insights and recommending data-driven action to appropriate agencies, such as TxDOT, Texas DPS, and local governments, in effort to minimize loss of life and injury and save tax payer dollars.

<br>
<br>

## Table of Contents

1.   [Project Overview              ](#1-project-overview)\
1a.   [Project Description          ](#1a-project-description)\
1b.   [Project Deliverables         ](#1b-project-deliverables)


2.  [Project Summary                ](#2-project-summary)\
2a.   [Goals                        ](#2a-goals)\
2b.   [Initial Thoughts & Hypothesis](#2b-initial-thoughts--hypothesis)\
2c.   [Findings & Next Steps        ](#2c-findings--next-steps)

3. [Data Context                 ](#c-data-context)\
3a.   [About Our Data            ](#3a-about-our-data)\
3b.   [Data Dictionary           ](#3b-data-dictionary)

4.  [Pipeline                     ](#4-pipeline)\
4a.   [Project Planning             ](#4a-project-planning)\
4b.   [Data Acquisition             ](#4b-data-acquisition)\
4c.   [Data Preparation             ](#4c-data-preparation)\
4d.   [Data Exploration             ](#4d-data-exploration)\
4e.   [Modeling & Evaluation        ](#4e-modeling--evaluation)\
4f.   [Product Delivery             ](#4f-product-delivery)

5.   [Modules                      ](#5-modules)

6.  [Project Reproduction         ](#6-project-reproduction)

7. [Ending Notes                  ](#7-ending-notes)

<br>


<br>

### 1. Project Overview
---

#### 1a. Project Description

Using 2021 San Antonio car crash data, web scraped from [myaccident](www.myaccident.org), we wanted to find what drives casualties in car accidents. We'll be acquiring, preparing, exploring, and using classification modeling to predict if certain features lead to injuries being reported.

<br>

#### 1b. Project Deliverables

- [Trello Board](https://trello.com/b/hyysoWdD/the-danger-zone)
- GitHub repository and this README with project overview, goals, findings, conclusion and summary
- Jupyter Notebook with a complete walkthrough of the data science pipeline, and commented with takeaways
- Any Python module(s) used to automate processes in the project 
- A presentation slide deck
- Live presentation for Codeup's Florence Data Scientist Day on September 3rd, 2021! 

<br>
<br>

### 2. Project Summary
---

#### 2a. Goals

Our goal is to predict whether or not an injury was reported in a vehicle at the time of an accident. Our selected classification model will be evaluated with the most emphasis on recall. We want to catch as many actual reported injuries as we can. We are hoping to find or create features that will help the model perform well.

<br>

#### 2b. Initial Thoughts & Hypothesis



**Ho:** *Vehicle year and whether an injury is reported is independent.*\
**Ha:** *Vehicle year and whether an injury is reported is not independent.*

**Ho:** *Vehicle type and whether an injury is reported is independent.*\
**Ha:** *Vehicle type and whether an injury is reported is not independent.*

**Ho:** *An accident caused by intoxication and whether an injury is reported is independent.*\
**Ha:** *An accident caused by intoxication and whether an injury is reported is not independent.*

**Ho:** *Weather, at the time of an accident, do not affect whether an injury is reported.*\
**Ha:** *Weather, at the time of an accident, does affect whether an injury is reported.*

**Ho:** *Number of occupants in the vehicle does not affect whether an injurt is reported.*\
**Ha:** *Number of occupants in the vehicle does affect whether an injurt is reported.*

**Ho:** *What part of town the accident occurs in does not affect if the vehicle reports an injury.*\
**Ha:** *What part of town the accident occurs in affects if the vehicle reports an injury.*

**Ho:** *The area's speed limit does not affect whether injuries are reported.*\
**Ha:** *The area's speed limit does affect whether injuries are reported.*

<br>

#### 2c. Findings & Next Steps

We found that attributes of the vehicle such as the year, and its type did contribute to whether an injury was reported.

Surprisingly, weather (as most San Antonians claim have an affect) statistically does not contribute to the number of injuries reported and neither does location!

- Most common accident type only involves 2 cars.
- Each car contains 1 person majority of the time, but an injury was more likely to be reported as the number of occupants increased
- The most frequent accident cause is driver inattention, followed by distaction and faulty manuevers.
- There is a variety of car makes and colors within the data set. White and black cars are the most commonly involved in recent accidents.
- In the last 6 months, roads where the speed limit are 45 MPH, followed by 35, and 65, have the most accidents involved.
- Cars followed by mpv(multi person vehicles ie. mini vans and crossovers) are involved in most accidents, followed by trucks
- Variables that correlate to accident by themselves were whether the air bag deployed, and the vehicle occupant count
- Many of the variables did not correlate alone, after clustering some of the variables together, we were able to find more correlation to the target variable
- the clusters created took speed and damage into account to help when modeling 
- Monday, Tuesday, Saturday early evening shows a decrease to injury rate before a sudden upward trend into the early hours of the following day
- Wednesday, Thursday seem to be more consistently near the mean rate versus other days
- Sunday early morning has the highest rate of traffic injuries throughout the data
- In the visual percentages of hour of accident with injury percentages, there is a marked increased in 0300 hours to 21%. Using X^2 testing, there is shown to be a statistical difference in the to categories of hour and if injury occurs.
- There is evidence to suggest that there is a difference in injuries reported during Fiesta
- There is not evidence to suggest that there is a difference in injuries reported during Spurs games (home games)
- There is evidence to suggest that there is a difference in injuries reported during July 4th weekend (and more of these accidents are due to intoxication than other days)
- Region within city does not affect injury rates


We created a Gradient Booster Classifier model (undersampling techniques used) that performed with 63% accuracy, and 61% recall. 
Though it did not outperform the baseline's accuracy of 83%, the baseline predicted that nobody got hurt it any accident. Our model is able to predict correctly for 61% of vehicles with injuries.

<br>

Next Steps:


 - Reevaluate what type of car caused the damage zone
 - Acquire seatbelt status of everyone in the vehicle
 - Find more relationships between variables
 - Individual street analysis

<br>
<br>

### 3. Data Context
---

#### 3a. About Our Data

This accident.csv was acquired from [www.myaccident.org]. We only took in accidents from Mid February - Mid August of 2021 that occurred in San Antonio, TX and surrounding areas. Every accident page will have the date, time, location, information of the cars, drivers, and cause as well as if they reported an injury, etc. 

We did create new features based on existing ones. We used redacted VIN to acquire all the vehicle types.

<br>



#### 3b. Data Dictionary

Here is a data dictionary for our `accident.csv`


| Column Name               | Description                                                                   |
|---------------------------|-------------------------------------------------------------------------------|
| accident_contained_injury | value of '1' if the accident had at least 1 person injured, or '0' if not     |
| crash_date                | the date and time the accident occurred                                       |
| crash_day_of_week         | the named day of the week the accident occurred on                            |
| crash_hour                | the hour in which the accident occurred (24 hour clock)                       |
| crash_id                  | police report crash identification number                                     |
| crash_latitude            | the accident location's latitude                                              |
| crash_location            | street address of the accident                                                |
| crash_longitude           | the accident location's longitude                                             |
| crash_occupant_count      | the number of occupants in the accident                                       |
| crash_time                | the time which the accident occurred at (24-hour-clock)                       |
| crash_vehicle_count       | the number of vehicles in the accident                                        |
| damage_airbag             | value of '1' if vehicle deployed its airbags, or '0' if not                   |
| damage_burned             | value of '1' if vehicle sustained burn damage, or '0' if not                  |
| damage_concentrated       | value of '1' if vehicle sustained concentrated damage, or '0' if not          |
| damage_distributed        | value of '1' if vehicle sustained distributed damage, or '0' if not           |
| damage_rollover           | value of '1' if vehicle sustained rollover damage, or '0' if not              |
| damage_zone               | indicates what area the vehicle was damaged in *                              |
| dl_cdl                    | value of '1' if driver has a CDL, or '0' if not                               |
| dl_class_a                | value of '1' if driver has a Class A license, or '0' if not                   |
| dl_class_b                | value of '1' if driver has a Class B license, or '0' if not                   |
| dl_class_m                | value of '1' if driver has a Class M license, or '0' if not                   |
| dl_state                  | State which driver's license was issued                                       |
| dl_unlicensed             | value of '1' if driver was not licensed to drive, '0' if they were licensed   |
| driver_age                | age of the driver in years                                                    |
| driver_male               | value of '1' if driver is male, '0' if female, '-1' if unknown                |
| driver_race               | 'H' Hispanic, 'W' White, 'B' Black, 'A' Asian, 'O' Other, 'N' for unknown     |
| factors_road              | description of road factors                                                   |
| factors_spd_lmt_mph       | speed limit of the road where accident occurred                               |
| factors_weather           | the weather conditions at the time of the accident                            |
| fault_class               | value of '1' if the driver was at fault, or '0' if not                        |
| fault_distraction         | value of '1' if the cause of the accident was distraction, or '0' if not      |
| fault_fatigue             | value of '1' if the cause of the accident was fatigue, or '0' if not          |
| fault_intoxication        | value of '1' if cause of accident was due to intoxication, or '0' if not      |
| fault_manuever            | value of '1' if cause of accident was due to poor manuever, or '0' if not     |
| fault_narrative           | the reported cause of motor vehicle accident                                  |
| fault_speed               | value of '1' if cause of accident was due to speeding, or '0' if not          |
| fault_yield               | value of '1' if cause of accident was due to failure to yield, or '0' if not  |
| injury_class*             | value of '1' if cause of vehicle contained at least one injury, or '0' if not |
| injury_crash_total        | the total number of injuries that resulted from the accident                  |
| vehicle_color             | the color of the vehicle                                                      |
| vehicle_id                | the vehicle's identification number                                           |
| vehicle_make              | the make of the vehicle                                                       |
| vehicle_occupant_count    | the number of occupants in the vehicle                                        |
| vehicle_type              | the type of vehicle                                                           |
| vehicle_year              | the vehicle's released year                                                   |






&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  * Target variable

<br>
<br>

### 4. Pipeline
---

#### 4a. Project Planning
:books: **Plan** ➜ ☐ _Acquire_ ➜ ☐ _Prepare_ ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Build this README containing:
    - Project overview
    - Initial thoughts and hypotheses
    - Data Dictionary
    - Walkthrough of the data science pipeline
    - Project summary
    - Instructions to reproduce
- [x] Plan stages of project and consider needs versus desires
    - Think about what target to do
    - Discuss how to approach the project with our team
    - Make a [Trello Board](https://trello.com/b/hyysoWdD/the-danger-zone)
    - Refresh with the lessons we will use in this project

#### 4b. Data Acquisition
✓ _Plan_ ➜ :open_book: **Acquire** ➜ ☐ _Prepare_ ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Web scrape San Antonio accident data from Feb 2021 - Aug 2021
- [x] Observe data structure
- [x] Save it to a local `.csv` for use. 

#### 4c. Data Preparation
✓ _Plan_ ➜ ✓ _Acquire_ ➜ :soap: **Prepare** ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Address missing values, and outliers. Assure all values are reasonable. 
- [x] Correct data types
- [x] Make any desirable object columns into machine-learning-friendly columns.
- [x] Create new features
- [x] Split data into train, and test sets (for cross validation)  


#### 4d. Data Exploration
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ :mag: **Explore** ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Explore univariate data
- [x] Explore bivariate data
- [x] Explore relationships between variables between each other, and the target.
- [x] Create clusters 
- [x] Form hypothesis and run statistical testing
- [x] Feature engineering with built in scikit modules
- [x] Make visuals

#### 4e. Modeling & Evaluation
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ :dart: **Model** ➜ ☐ _Deliver_

- [x] Establish baseline prediction
- [x] Create, fit, and predict with models
- [x] Use GridsearchCV to find optimal hyperparameters
- [x] Evaluate models with cross validation
- [x] Find ways to handle imbalanced data
- [x] Emphasis on recall score

#### 4f. Product Delivery
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ ✓ _Model_ ➜ :white_check_mark: **Deliver**
- [x] Prepare Jupyter Notebook with thorough walk-through of the data science pipeline
- [x] Review Panel I
- [x] Review Panel II
- [x] Review Panel III
- [x] Recording of Presentation
- [x] Live Presentation
- [x] Share findings 
- [x] Address next steps


<br>
<br>

### 5. Modules
---

 - `wrangle.py` = contains acquire and prepare functions used to retrieve and prepare the accident data for use.
 - `explore.py` = contains functions used to explore, visualize, and run statistical tests.
 - `evaluate.py` = contains modeling related functions

<br>
<br>

### 6. Project Reproduction
---

Recreate this project in a few easy steps:
 - Git clone this repository or:
     - Download `accident.csv`
     - Download `.py` modules
     - Download `final_danger_zone.ipynb` notebook

<br>
<br>

### 7. Ending Notes
---

Remember that you can do your part to protect yourself and others before an accident happens. 
 - Commute with care
     - Drive carefully and be respectful to others.
 - Always stay aware and focused
     - Pay attention to your surroundings, minimize distractions, hands on the wheel, and keep your eyes on the road (plus mirrors!)
 - Protect yourself, family, friends, and others
     - Wear your seatbelt and maintain a safe distance between other vehicles. Don't forget to turn on your headlights at night.
 - **Don't drink and drive**
     - Find a designated driver
     - Before major holidays, look out for potentially free tow and ride services offered in San Antonio in case you or someone you know could use it
     - If possible, avoid driving late at night.  



[Return to Top](#top)