
# The Danger Zone 
Brought to you by data scientists:
 - Xavier Carter
 - Robert Murphy
 - Anna Vu
 - Ray Zapata
<br>
<br>

San Antonio is the 7th most populated, and one of the fastest growing cities in the U.S.A. In Bexar County alone, there were nearly 50,000 car crashes in 2020. Of these, 16,780 were injured, 200 died, and most were preventable. With an increasing number of drivers on the roads, there is a recurring need to keep people safe. Using 2021 San Antonio car accident data, the Danger Zone project will look into features that are likely to cause casualties, so that insight can be delivered to entities such as TXDot, Bexar County Public Works, insurance companies, and the general public.

<br>
<br>

### Table of Contents

1.   [Project Overview              ](#1-project-overview)\  
1a.   [Project Description          ](#1a-project-description)\  
1b.   [Project Deliverables         ](#1b-project-deliverables)\  


2.  [Project Summary                ](#2-project-summary)\
2a.   [Goals                        ](#2a-goals)\
2b.   [Initial Thoughts & Hypothesis](#2b-initial-thoughts--hypothesis)\  
2c.   [Findings & Next Steps        ](#2c-findings--next-steps)\ 

3. [Data Context                 ](#c-data-context)\  
3a.   [About Our Data            ](#3a-about-our-data)\ 
3b.   [Data Dictionary           ](#3b-data-dictionary)\  

4.  [Pipeline                     ](#4-pipeline)\   
4a.   [Project Planning             ](#4a-project-planning)\  
4b.   [Data Acquisition             ](#4b-data-acquisition)\  
4c.   [Data Preparation             ](#4c-data-preparation)\  
4d.   [Data Exploration             ](#4d-data-exploration)\  
4e.   [Modeling & Evaluation        ](#4e-modeling--evaluation)\  
4f.   [Product Delivery             ](#4f-product-delivery)\  

E.   [Modules                      ](#e-modules)\  

F.  [Project Reproduction         ](#f-project-reproduction)\  

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

Our goal is to predict whether or not an injury was reported in a vehicle at the time of an accident. Our classification model should be able to beat the baseline score (assuming nobody reports injuries in every single vehicle after an accident.) We are hoping to find or create features that will help the model perform well.

#### 2b. Initial Thoughts & Hypothesis

$H$0: Vehicle year and whether an injury is reported is independent.
$H$A: Vehicle year and whether an injury is reported is not independent.

$H$0: An accident caused by intoxication and whether an injury is reported is independent.
$H$A: An accident caused by intoxication and whether an injury is reported is not independent.

$H$0: Weather, at the time of an accident, do not affect whether an injury is reported.
$H$A: Weather, at the time of an accident, does affect whether an injury is reported.

$H$0: Number of occupants in the vehicle does not affect whether an injurt is reported.
$H$A: Number of occupants in the vehicle does affect whether an injurt is reported.

$H$0: Number of occupants in the vehicle does not affect whether an injurt is reported.
$H$A: Number of occupants in the vehicle does affect whether an injurt is reported.

$H$0: What part of town the accident occurs in does not affect if the vehicle reports an injury.
$H$A: What part of town the accident occurs in affects if the vehicle reports an injury.

$H$0: The area's speed limit does not affect whether injuries are reported.
$H$A: The area's speed limit does affect whether injuries are reported.

#### 2c. Findings & Next Steps

Exploration findings here

model performance here
<br>

next steps here

<br>
<br>

### 3. Data Context
---

#### 3a. About Our Data

This accident.csv was acquired from [www.myaccident.org]. We only took in accidents from Mid February - Mid August of 2021 that occurred in San Antonio, TX and surrounding areas. Every accident page will have the date, time, location, information of the cars, drivers, and cause as well as if they reported an injury, etc. 

We did create new features based on existing ones. 


#### 3b. Data Dictionary

Here is a data dictionary for our accident.csv 


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

- [x] Web scrape San Antonio accident data
- [x] Observe data structure
- [x] Save it to a local .csv for use. 

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
- [x] Evaluate models with cross validation

#### 4f. Product Delivery
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ ✓ _Model_ ➜ :white_check_mark: **Deliver**
- [x] Prepare Jupyter Notebook with thorough walk-through of the data science pipeline
- [x] Share findings 
- [x] Address next steps


<br>
<br>

### E. Modules
---

 - wrangle.py = contains acquire and prepare functions used to retrieve and prepare the accident data for use.
 - explore.py = contains functions used to explore, visualize, and run statistical tests.

<br>
<br>

### F. Project Reproduction
---

Recreate this project in a few easy steps:
 - Download this repository or:
     - Download .py modules
     - Download final_danger_zone.ipynb notebook
     - Download accident.csv

<br>

