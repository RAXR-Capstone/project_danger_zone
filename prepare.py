#Data Preparation 

# import standard libraries
import pandas as pd
import numpy as np
import re

# import word cleaning tools
import unicodedata
import nltk

# import data tools
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import datetime as dt
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

# hide warnings
import warnings
warnings.filterwarnings("ignore")


#################### Prep MVC Data ####################


def encode_vehicle_type(df):
    '''
    encode vehicle type into numeric data type
    '''

    # make encoder object
    enc = OneHotEncoder(sparse=False, drop='first')
    # fit encoder and concat DataFrame with new columns
    one_hots_vehicle_type = pd.DataFrame(enc.fit_transform(
                                            df[['vehicle_type']]),
                                            columns=enc.get_feature_names(),
                                            index=df.index)

    return df


def misc_prep(df):
    '''

    Initial basic preparation of auto collision data. Remove duplicate
    observations and convert crash_data to approprate datatime format
    using 24-hour time. Drops unnecessary columns.

    '''

    # drop duplicate observations
    df = df.drop_duplicates()
    # convert crash_date to datetime dtype and 24-hour clock
    df.crash_date =  df.crash_date.apply(lambda row: \
                     pd.to_datetime(row).strftime('%m/%d/%Y %H:%M'))
    df.crash_date =  pd.to_datetime(df.crash_date)
    # drop columns
    df = df.drop(columns=['case_id', 'crash_city', 'crash_url',
                          'police_dept', 'crash_location',
                          'driver_residence', 'driver_insured'])
    # remove mph and convert speed_limit column to integer data type
    df.speed_limit = df.speed_limit.apply(lambda row:
                                          re.search(r'(\D?\d{1,2}\s)', row)\
                                    .group(1)).astype('int')
    # drop observations without latitude value
    df = df[df.crash_latitude != 0]
    # rename cols
    rename_dict = {'accident_factor':'fault_narrative',
                   'at_fault':'fault_class',
                   'num_of_injuries':'injury_crash_total',
                   'speed_limit':'factors_spd_lmt_mph',
                   'car_contained_injury':'injury_class',
                   'num_of_vehicles':'crash_vehicle_count',
                   'num_of_occupants':'crash_occupant_count',
                   'car_airbags_deployed':'damage_airbag',
                   'occupants_in_car':'vehicle_occupant_count'}
    df = df.rename(columns=rename_dict)

    return df


def feature_extraction(df):
    '''
    feature_extraction will create columns crash_day, crash_hour, driver_age_bin, and vehicle_year_bin 
    from our existing data by extracting features from our crash_date column or binning ages
    '''

    #get the weekday name from date
    df['crash_day'] = df.set_index('crash_date').index.day_name()
    #get the hour the crash happened from date
    df['crash_hour'] = df.crash_date.dt.hour
    #bin age into groups
    df['driver_age_bin'] = pd.cut(df.driver_age, [16, 25, 35, 45, 60, 120])
    #bin vehicle year into groups
    df['vehicle_year_bin'] = pd.cut(df.vehicle_year,
                                   bins=[1984, 1999, 2002,
                                         2003, 2004, 2005,
                                         2006, 2007, 2008,
                                         2009, 2010, 2011,
                                         2012, 2013, 2014,
                                         2015, 2016, 2017,
                                         2018, 2019, 2020, 2022])

    return df


#################### Prep Driver Data ####################


def create_dl_classes(df):
    '''

    Use existing "driver_license_type" column to derive boolean columns
    for if driver as a CDL, is unlicensed, and has a Class A, B, and or
    M driver's license.

    '''
    
    # create cdl boolean column where DL type contains "commericial"
    df['dl_cdl'] = df.driver_license_type.apply(lambda row: \
                    1 if str(row).lower().__contains__('commercial') else 0)
    # create unlicensed bool column wher DL type contains "unlicensed"
    df['dl_unlicensed'] = df.driver_license_type.apply(lambda row: \
                    1 if str(row).lower().__contains__('unlicensed') else 0)
    # create bool columns where DL type contains "a", "b", and "m"
    df['dl_class_a'] = df.driver_license_type.apply(lambda row: \
                    1 if str(row).lower().__contains__('class a') else 0)
    df['dl_class_b'] = df.driver_license_type.apply(lambda row: \
                    1 if str(row).lower().__contains__('class b') else 0)
    df['dl_class_m'] = df.driver_license_type.apply(lambda row: \
                    1 if str(row).lower().__contains__('class m') or \
                         str(row).lower().__contains__('and m') else 0)
    
    return df


def clean_driver_race(df):
    '''
    
    Takes existing racial data on driver and converts to a simplified
    US Census style abbreviation of race, replaces missing values with
    np.nan for future imputation or other handling

    '''

    # replace unknown with NaN
    df.driver_ethnicity = np.where(df.driver_ethnicity == \
                                   'Amer. indian/alaskan native',
                                   'indigenous', df.driver_ethnicity)
    df.driver_ethnicity = df.driver_ethnicity.apply(
                            lambda row: str(row).lower().strip())
    df.driver_ethnicity = df.driver_ethnicity.apply(
                            lambda row: np.nan
                            if any(x in row for x in ['unknown','nan'])
                            else row)
    df = df.rename(columns={'driver_ethnicity':'driver_race'})

    return df


def clean_driver_age(df):
    '''

    Converts driver_age column into integer data type and replaces
    missing or inappropriate values with NaN for future imputing or
    other means of handling

    '''

    # create mask where driver's age is non-digit
    mask = df.driver_age.apply(lambda row: bool(
                                re.search(r'\D*\d+\D*', str(row))))
    # replace non-digit drivers age with Nan
    df.driver_age = np.where(df.driver_age.isin(df[mask].driver_age),
                                                df.driver_age, np.nan)
    # replace driver_age less than 6 with NaN
    df.driver_age = np.where(df.driver_age >= 10, df.driver_age, np.nan)

    return df


def clean_driver_gender(df):
    '''
    clean driver gender to driver_male column. 1 as male, 0 as female, null for unknown
    '''

    # replace "Unknown" gender with NaN
    df.driver_gender = np.where(df.driver_gender == 'Unknown', np.nan,
                                                    df.driver_gender)
    # change to one-hot where male gender driver == 1
    df.driver_gender = np.where(df.driver_gender == 'Male', 1,
                                                    df.driver_gender)
    df.driver_gender = np.where(df.driver_gender == 'Female', 0,
                                                    df.driver_gender)
    # change dtype to int and rename
    df = df.rename(columns={'driver_gender':'driver_male'})

    return df


def prep_driver_data(df):
    '''

    Uses functions to prepare driver data with cleaned up demographics
    and drivers license data encoded variables for class and type, or
    if the driver was confirmed as unlicensed. Drops unneeded column
    used to derive other features

    '''

    # lowercase state value and strip terminal whitespace
    df.driver_license_state = df.driver_license_state\
                                .apply(lambda row: str(row).lower().strip())
    # set DL state as Texas, Other, or NaN
    df.driver_license_state = np.where(df.driver_license_state.isin(\
                        ['unknown', 'nan']), np.nan, df.driver_license_state)
    df.driver_license_state = np.where(df.driver_license_state.isin(\
                        ['texas', np.nan]), df.driver_license_state, 'other')
    # rename column
    df = df.rename(columns={'driver_license_state':'dl_state'})
    # use function to change gender to one-hot
    df = clean_driver_gender(df)
    # use function to create DL class bool columns
    df = create_dl_classes(df)
    # use function to clean driver_ethnicity column
    df = clean_driver_race(df)
    # use function to clean driver_age column
    df = clean_driver_age(df)
    # drop lengthy dl_type column used to derive DL classes
    df = df.drop(columns='driver_license_type')

    return df


#################### Prep Vehicle Data ####################


def clean_vin(df):
    '''

    Takes the auto collision data and prepares VIN by repalcing
    existing "X"s with asterisks to improve cross referencing with
    NHTSA VIN decoder for additional vehicle information

    '''
    
    # replace Xs for improved use with NTHSA cross reference
    df.car_vin = df.car_vin.apply(lambda row: re.sub(r'(\w{4})X{9}(\d{4})',
                                                     r'\1*********\2', row))
    # set mask for appropriate VIN values
    mask = df.car_vin.apply(lambda row: bool(re.search(r'(\w{4})\*{9}(\d{4})',
                                                                        row)))
    # replace inappropriate VIN values with "Unknown"
    df.car_vin = np.where(df.car_vin.isin(df[mask].car_vin),df.car_vin, np.nan)
    # rename column
    df = df.rename(columns={'car_vin':'vehicle_id'})
    
    return df


def clean_make(df):
    '''

    Takes existing vehicle make data and consolidates variations on
    labels by striping terminal whitespace and grouping mislabeled
    data. Additionally takes vehicles not in the top 25 makes and
    consolidates into "other" group for improved label encoding.
    Repalces unknown makes with np.nan for late imputation or other
    handling.

    '''


    # strip beginning and ending white space
    df.car_make = df.car_make.apply(lambda row: str(row).strip().lower())
    # fix partial and model matches to consolidate make groups
    df.car_make = df.car_make.apply(lambda row: 'gmc' \
                            if str(row).lower().__contains__('gm') else row)
    df.car_make = df.car_make.apply(lambda row: 'dodge' \
                            if str(row).lower().__contains__('ram') else row)
    # add mislabeled data to unknown category
    df.car_make = df.car_make.apply(lambda row: 'unknown' \
                            if str(row).lower() == 'd' else row)
    df.car_make = df.car_make.apply(lambda row: 'unknown' \
                            if str(row).lower() == 'nan' else row)
    # set all unknowns as np.nan for later imputation or other handling
    df.car_make = df.car_make.apply(lambda row: np.nan \
                            if str(row).lower().__contains__('unk') else row)
    # set top makes for filtering
    top_makes = df.car_make.value_counts(dropna=False).head(26).index
    # for all makes not in top 25 occurences, set as "other"
    df.car_make = np.where(df.car_make.isin(top_makes), df.car_make, 'other')
    # rename column
    df = df.rename(columns={'car_make':'vehicle_make'})
    
    return df


def clean_year(df):
    '''

    Takes existing vehicle manufactured year and removes non numerical
    values. Replaces value with NaN for vehicles without known
    manufacutring year for later imputation or other handling.

    '''
    
    # pull numerical, four digit vehicle manufacture years
    df.car_year = df.car_year.apply(lambda row: re.sub(r'\s?(\d+)(.0)?',
                                                       r'\1', str(row)))
    # set mislabeled and unknown years as NaN
    df.car_year = df.car_year.apply(lambda row: np.nan if row == 'nan'
                                                   or row == '0'
                                                   else row)
    # # rename column
    df = df.rename(columns={'car_year':'vehicle_year'})
    
    return df


def clean_color(df):
    '''

    Takes existing vehicle color data and strips terminal whitespace,
    replaces unknown values with np.nan for later handling, and groups
    less common types into "other" category for more efficient label
    encoding.

    '''
    
    # clean up vehicle color to contain only single word
    df.car_color = df[~df.car_color.isna()].car_color.apply(lambda row: \
                                        re.search(r'\W*(\w+)[\W]*', row)\
                                        .group(1).lower())
    # convert unknowns to NaN for later imputation or other handling
    df.car_color = np.where(df.car_color.isin(['Unknown']),
                                      np.nan, df.car_color)
    # group less common colors into "other" category
    df.car_color = np.where(df.car_color.isin(['Copper', 'Pink', 'Teal',
                                        'Bronze', 'Turquoise', 'Purple']),
                                        'Other', df.car_color)
    # rename column
    df = df.rename(columns={'car_color':'vehicle_color'})
    
    return df


def clean_type(df):
    '''
    clean_type will distinguish vehicle types further and simplify their names
    '''

    #
    df.car_type = df.car_type.apply(lambda row: str(row).strip().lower())
    non_pass = ['incomplete', 'trailer', 'low']
    df.car_type = df.car_type.apply(lambda row: row
                                    if all(x not in row for x in non_pass)
                                    else 'non-passenger')
    df.car_type = np.where(
                        df.car_type == 'multipurpose passenger vehicle (mpv)',
                        'mpv', df.car_type)
    df.car_type = np.where(df.car_type == 'passenger car', 'car', df.car_type)
    df.car_type = np.where(df.car_type == 'nan', np.nan, df.car_type)
    df = df.rename(columns={'car_type':'vehicle_type'})

    return df


def prep_vehicle_data(df):
    '''

    Used functions defined above to prepare data related to vehicle,
    removing inappropraite values and missing data for better imputing
    and handling, and returns vehicle data prepped for exploration.

    '''

    # use function to prepare vin with asterisk
    df = clean_vin(df)
    # use function to consolidate and clean vehicle make
    df = clean_make(df)
    # use function to remove inappropraite values from year
    df = clean_year(df)
    # use function to consolidate and clean vehicle color
    df = clean_color(df)
    #
    df = clean_type(df)

    return df


#################### Prep Damage Data ####################


def make_vehicle_dmg_zone(df):
    '''

    Takes in the auto collision DataFrame and creates
    vehicle_impact_zone column, where each integer value corresponds to
    a specific aspect of vehicle damage incurrect in incident:

        Zone  0 *** Undercarriage
        Zone  1 *** Front End
        Zone  2 *** Back End
        Zone  3 *** Driver Front Quarter
        Zone  4 *** Driver Side
        Zone  5 *** Driver Back Quarter
        Zone  6 *** Passenger Front Quarter
        Zone  7 *** Passenger Side
        Zone  8 *** Passenger Back Quarter
        Zone  9 *** Severe Damage (Burned, Rollover)
        Zone 10 *** Motorcycle, scooter, etc.

    '''
    
    # set pattern for regex search of damage column
    pattern = r'^(\S+){1,4}.+'
    # create series of impact types
    impact_type = df.driver_car_damage.apply(
                            lambda row: re.search(pattern, row).group(1))
    # create column for impact zone as described in docstring
    df['damage_zone'] = np.where(impact_type\
                            .isin(['VX', 'MC']), 0, np.nan)
    df['damage_zone'] = np.where(impact_type\
                            .isin(['FL','FR','FC','FD']), 1, df.damage_zone)
    df['damage_zone'] = np.where(impact_type\
                            .isin(['BL','BR','BC','BD']), 2, df.damage_zone)
    df['damage_zone'] = np.where(impact_type\
                            .isin(['LFQ']), 3, df.damage_zone)
    df['damage_zone'] = np.where(impact_type\
                            .isin(['LP', 'LD', 'L&T']), 4, df.damage_zone)
    df['damage_zone'] = np.where(impact_type\
                            .isin(['LBQ']), 5, df.damage_zone)
    df['damage_zone'] = np.where(impact_type\
                            .isin(['RFQ']), 6, df.damage_zone)
    df['damage_zone'] = np.where(impact_type\
                            .isin(['RD', 'RP', 'R&T']), 7, df.damage_zone)
    df['damage_zone'] = np.where(impact_type\
                            .isin(['RBQ']), 8, df.damage_zone)
    df['damage_zone'] = np.where(impact_type\
                            .isin(['TP','L&T','R&T','VB']), 9, df.damage_zone)
    df['damage_zone'] = np.where(impact_type\
                            .isin(['MC']), 10, df.damage_zone)

    return df


def make_dmg_type_columns(df):
    '''

    Takes in auto collision DataFrame creates columns with boolean
    values of damage type incurred in incident, where 0 is False and 1
    is True

        "concentrated_damage": damage caused by narrow object,
                               e.g. tree, utility pole
        "distributed_damage": damage cause by broad object,
                              e.g. building wall, another motor vehicle
        "rollover_damage": incident included at least partial vehicle
                           rotation with top damage
        "vehicle_burned": vehicle fire occured as a result of collision

    '''

    # set pattern for regex search of damage column
    pattern= r'^(\S+){1,4}.+'
    # create series of impact types
    impact_type = df.driver_car_damage.apply(
                    lambda row: re.search(pattern, row).group(1))
    # create columns for boolean of damage types
    df['damage_concentrated'] = np.where(impact_type.isin(['FC', 'BC']), 1, 0)
    df['damage_distributed'] = np.where(impact_type.isin(
                                            ['FD', 'BD', 'LD', 'RD']), 1, 0)
    df['damage_rollover'] = np.where(impact_type.isin(
                                            ['TP', 'L&T', 'R&T']), 1, 0)
    df['damage_burned'] = np.where(impact_type.isin(['VB']), 1, 0)

    return df


def prep_damage_data(df):
    '''

    Uses functions to create column with area of damage for the vehicle
    and columns of possible types of damage incurred, drops
    driver_car_damage column that contains lengthy string descriptor

    '''

    # use function to make vehicle_dmg_zone column
    df = make_vehicle_dmg_zone(df)
    # use function to make damage type columns
    df = make_dmg_type_columns(df)
    # drop wordy column used to construct above
    df = df.drop(columns='driver_car_damage')

    return df


#################### Language Prep Tools ####################


def basic_clean(string):
    '''
    basic_clean takes in a string and lowercases its contents, normalizes unicode characters,
    and replaces anything that is not a letter, number, whitespace, or single quote with nothing
    '''
    
    # convert applicable characters to lowercase
    string = string.lower()
    # normalize unicode characters
    string = unicodedata.normalize('NFKD', string)\
                        .encode('ascii', 'ignore')\
                        .decode('utf-8')
    # substitute non-alphanums, spaces, and
    # single quotes/apostrophes
    string = re.sub(r'[^0-9a-z\s\']', '', string)
    
    return string


def tokenize(string):
    '''
    tokenize will take in a string and tokenize all of the words in it
    '''
    
    # create tokenizer object
    tokenizer = nltk.tokenize.ToktokTokenizer()
    # tokenize string and return as string
    string = tokenizer.tokenize(string, return_str=True)
    
    return string


def remove_stopwords(string, extra_words=None, exclude_words=None):
    '''
    takes in some text and removes stop words
    '''
    
    # create stopwords list
    stopword_list = nltk.corpus.stopwords.words('english')
    # add more stop words if needed
    if extra_words != None:
        stopword_list.extend(extra_words)
    # remove stop words if needed
    if exclude_words != None:
        if len(exclude_words) > 1:
            for word in exclude_words:
                stopword_list.remove(word)
        else:
            stopword_list.remove(exclude_words[0])

    # obtain list and join filtered for stopwords
    string = ' '.join(
                [word for word in string.split() if word not in stopword_list])
    
    return string


def lemmatize(string):
    '''
    takes in a string and lemmatizes it
    '''
    
    # create lemmatizer object
    lemmer = nltk.stem.WordNetLemmatizer()
    # get list of lems for words in split string
    string_lems = [lemmer.lemmatize(word) for word in string.split()]
    # join stems back as string from list
    string = ' '.join(string_lems)
    
    return string


#################### Prep Factor Data ####################


def lang_prep_factor_col(df):
    '''

    Takes the fault_narrative column from the auto collision data and
    prepares it in a manner for deriving one-hot encoded features for
    MVC causes and future use in NLP exploration.

    '''

    # create list of common, non-insightful words in narrative
    more_words = ['"', '\'', 'driver', 'explain', 'narrative', 'nan', 
                   'undefined', 'unknown']
    # perform cleaning on text in fault_narrative
    df.fault_narrative = df.fault_narrative.apply(lambda row: 
                                    lemmatize(remove_stopwords(tokenize(
                                        basic_clean(str(row))), more_words)))
    # remove vehicle years from string
    df.fault_narrative = df.fault_narrative.apply(lambda row:
                                                  re.sub(r'\d{4}\s?', '', row))
    df.fault_narrative = df.fault_narrative.apply(lambda row: row.strip())

    return df


def create_fault_narrative_cols(df):
    '''

    Takes the auto collision DataFrame and uses keywords from the
    fault_narrative column to derive one-hot encoded columns for
    possible fault factors, where all 0s indicates "Other" causes

    '''
    
    df = lang_prep_factor_col(df)
    # create boolean col for "distraction" cause
    dist = ['inatt', 'distr', 'cell']
    df['fault_distraction'] = df.apply(
                                lambda row: 1
                                if any(x in row.fault_narrative for x in dist)
                                else 0, axis=1)
    # create boolean col for "meaneuver" related cause
    manu = ['lane','turn','follo','pas','back','evas','close']
    df['fault_maneuver'] = df.apply(
                                lambda row: 1
                                if any(x in row.fault_narrative for x in manu)
                                else 0, axis=1)
    # create boolean col for "speed" related cause
    df['fault_speed'] = df.apply(
                                lambda row: 1
                                if 'speed' in row.fault_narrative
                                else 0, axis=1)
    # create boolean col for intoxication realted causes
    intx = ['drink', 'infl', 'medi', 'alc','drunk']
    df['fault_intoxication'] = df.apply(
                                lambda row: 1
                                if any(x in row.fault_narrative for x in intx)
                                else 0, axis=1)
    # create boolean col for fatigue realted causes
    fati = ['sleep', 'fatig', 'ill','tired']
    df['fault_fatigue'] = df.apply(
                                lambda row: 1
                                if any(x in row.fault_narrative for x in fati)
                                else 0, axis=1)
    # create boolean col for failing to "yield" or stop related causes
    yild = ['stop', 'yiel']
    df['fault_yield'] = df.apply(
                                lambda row: 1
                                if any(x in row.fault_narrative for x in fati)
                                else 0, axis=1)
    # create boolean col for doing unsafe action
    df['fault_unsafe'] = df.apply(
                                lambda row: 1
                                if 'unsafe' in row.fault_narrative
                                else 0, axis=1)
    return df


#################### Prep Road Conditions ####################


def clean_traffic_cats(df):
    '''
    clean_traffic_cats will clean traffic conditions by simplifying their names and putting more uncommon conditions into 
    the "other" category, and rename the column as factors_road
    '''
    
    # turn object into string and lowercase, strip terminal whitespace
    df.traffic_conditions = df.traffic_conditions.apply(
                                lambda row: str(row).lower().strip())
    df.traffic_conditions = df.traffic_conditions.apply(
                                lambda row: 'signal light'
                                if 'signal' in row
                                else row)
    df.traffic_conditions = df.traffic_conditions.apply(
                                lambda row: 'flashing light'
                                if 'flashing' in row
                                else row)
    # convert categories no in road list into "other"
    road = ['marked', 'none', 'signal',
            'stop', 'flashing', 'center',
            'nan', 'yield', 'officer']
    df.traffic_conditions = df.traffic_conditions.apply(
                                lambda row: row
                                if any(x in row for x in road)
                                else 'other')
    # fill string "nan" values with np.nan for later handling
    df.traffic_conditions = df.traffic_conditions.apply(
                                lambda row: np.nan
                                if row == 'nan'
                                else row)
    # rename column
    df = df.rename(columns={'traffic_conditions':'factors_road'})

    return df


def clean_weather_cats(df):
    '''
    clean_weather_cats will clean the column weather_conditions, take lesser common ones
    and turn them into an 'other' category,
    and rename the column into factors_weather
    '''
    
    # turn object into string and lowercase, strip terminal whitespace
    df.weather_conditions = df.weather_conditions.apply(
                                lambda row: str(row).lower().strip())
    # convert categories no in weather list into "other"
    weather = ['clear', 'cloudy', 'rain',
               'sleet', 'hail', 'snow', 'nan']
    df.weather_conditions = df.weather_conditions.apply(
                                lambda row: row
                                if any(x in row for x in weather)
                                and 'sand' not in row
                                else 'other')
    # fill string "nan" values with np.nan for later handling
    df.weather_conditions = df.weather_conditions.apply(
                                lambda row: np.nan
                                if row == 'nan'
                                else row)
    # rename column
    df = df.rename(columns={'weather_conditions':'factors_weather'})
    
    return df


#################### Prep MVC Clusters ####################


def create_kmode_clusters(train, test, n, cluster_name, var=[]):
    '''

    takes in train, test sets, the number of clusters to make, variable names,
    and a desired cluster column name and will use kmode to create the cluster
    groups. It will also append the predicted clusters to the train and test
    dataframes and return them back

    '''

    #initialize kmode 
    kmode = KModes(n_clusters=n, init="random", n_init=5, verbose=0,
                                                    random_state=19)
    
    #make clusters for train
    clusters = kmode.fit_predict(train[var])
    #add to train set
    train[cluster_name] = clusters
    
    #make clusters for test set
    clusters_test = kmode.predict(test[var])
    #add to test set
    test[cluster_name] = clusters_test

    return train, test


def create_kmeans_clusters(train, test, n, cluster_name, var=[]):
    '''
    takes in train and test sets, a desired cluster name, and the variables to cluster on.
    will return clusters to both the train and test set
    '''

    #initialize kmode 
    kmeans = KMeans(n_clusters=n)
    
    # scaled data for kmeans
    scaler = StandardScaler()
    scaler.fit(train[var])
    train_scaled = pd.DataFrame(scaler.transform(train[var]),
                                columns=train[var].columns,
                                index=train[var].index)
    test_scaled = pd.DataFrame(scaler.transform(test[var]),
                               columns=test[var].columns,
                               index=test[var].index)
    # make clusters for train
    clusters = kmeans.fit_predict(train_scaled)
    #add to train set
    train[cluster_name] = clusters
    #make clusters for test set
    clusters_test = kmeans.predict(test_scaled)
    #add to test set
    test[cluster_name] = clusters_test

    return train, test


def create_mvc_clusters(train, test):
    '''
    ''' 'factors_spd_lmt_mph', 'fault_yield', 'vehicle_occupant_count'

    #
    train, test = create_kmode_clusters(train, test, 4,
                                    cluster_name='damage_air',
                                    var=['damage_zone','damage_airbag'])
    #
    train, test = create_kmode_clusters(train, test, 4,
                                    cluster_name='speed_speed_lm',
                                    var=['fault_speed','factors_spd_lmt_mph'])
    train, test = create_kmeans_clusters(train, test, 4,
                                    cluster_name='speed_yield_occu',
                                    var=['factors_spd_lmt_mph', 'fault_yield',
                                                    'vehicle_occupant_count'])

    return train, test


#################### Final Data Prep ####################


def clean_dtypes(df):
    '''
    fix data types into integers
    '''

    # set driver_age as integer dtype
    df.driver_age = df.driver_age.astype('int')
    # set driver_gender as integer dtype
    df.driver_male = df.driver_male.astype('int')
    # set car_year as integer dtype
    df.vehicle_year = df.vehicle_year.astype('int')
    # set damage_zone as integer dtype
    df.damage_zone = df.damage_zone.astype('int')
    # encode vehicle type
    df = encode_vehicle_type(df)
    # rename columns
    rename_dict = {'x0_car':'vehicle_car',
                   'x0_motorcycle':'vehicle_mc',
                   'x0_mpv':'vehicle_mpv',
                   'x0_non-passenger':'vehicle_non_passenger',
                   'x0_truck':'vehicle_truck'}
    df = df.rename(columns=rename_dict)

    return df


def clean_collision_data(dropna=True):
    '''
    clean_collision_data will take in the original csv, and perform all of the cleaning and prep functions
    (see previous functions), it will also any rows with at least one null value
    '''

    # read in csv
    df = pd.read_csv('accident_data.csv')
    # perform initial, misc prep work
    df = misc_prep(df)
    # prepare driver data
    df = prep_driver_data(df)
    # prepare vehicle data
    df = prep_vehicle_data(df)
    # prepare damage data
    df = prep_damage_data(df)
    # create fault factor cols
    df = create_fault_narrative_cols(df)
    # prep road conditions
    df = clean_traffic_cats(df)
    df = clean_weather_cats(df)
    if dropna == True:
        # drop null values
        df = df.dropna()
        # convert to appropriate data types
        df = clean_dtypes(df)
    #create new features from existing ones
    df = feature_extraction(df)

    return df


def collision_data(dropna=True):
    '''
    collision_data will clean our data, drop nulls, and split the data into train and test sets.
    Returns train set, and test set.
    '''

    #
    df = clean_collision_data(dropna=dropna)
    #
    train, test = train_test_split(df, test_size=0.2, random_state=19,
                                            stratify=df.injury_class)
    #
    train, test = create_mvc_clusters(train, test)
    # sort train columns alphabetically
    cols = train.columns.tolist()
    cols.sort()
    train = train[cols]
    # sort test columns alphabetically
    cols = test.columns.tolist()
    cols.sort()
    test = test[cols]

    return train, test

def scale_data(train, test, scale_type = None, to_scale = None):
    '''
    returns scaled data of specified type into data frame
    '''
    #create copies of our train set and test set
    train_copy = train.copy()
    test_copy = test.copy()
    
    #if no scaler specified, return the train and test sets
    if to_scale == None:
        return train_copy, test_copy
    
    #if scaler is specified
    else:
        #define an X set with columns (to scale) inputted
        X_train = train_copy[to_scale]
        X_test = test_copy[to_scale]
        
        #intialize minmax, robust, and standars scalers
        min_max_scaler = MinMaxScaler()
        robust_scaler = RobustScaler()
        standard_scaler = StandardScaler()
        
        #fit the scalers to X_train
        min_max_scaler.fit(X_train)
        robust_scaler.fit(X_train)
        standard_scaler.fit(X_train)
        
        #transform columns to scale with the different scalers on X_train
        mmX_train_scaled = min_max_scaler.transform(X_train)
        rX_train_scaled = robust_scaler.transform(X_train)
        sX_train_scaled = standard_scaler.transform(X_train)
    
        #transform columns to scale with the different scalers on X_test
        mmX_test_scaled = min_max_scaler.transform(X_test)
        rX_test_scaled = robust_scaler.transform(X_test)
        sX_test_scaled = standard_scaler.transform(X_test)
    
        #minmax scaled data into dataframe
        mmX_train_scaled = pd.DataFrame(mmX_train_scaled, columns=X_train.columns)
        mmX_test_scaled = pd.DataFrame(mmX_test_scaled, columns=X_test.columns)
        
        #robust scaled data into dataframe
        rX_train_scaled = pd.DataFrame(rX_train_scaled, columns=X_train.columns)
        rX_test_scaled = pd.DataFrame(rX_test_scaled, columns=X_test.columns)

        #standard scaled data into dataframe
        sX_train_scaled = pd.DataFrame(sX_train_scaled, columns=X_train.columns)
        sX_test_scaled = pd.DataFrame(sX_test_scaled, columns=X_test.columns)
    
    #if MinMax was specified, insert the minmax scaled data into the train and test copies
    if scale_type == 'MinMax':
        for i in mmX_train_scaled:
            train_copy[i] = mmX_train_scaled[i].values
            test_copy[i] = mmX_test_scaled[i].values
    #if Robust was specified, insert the robust scaled data into the train and test copies
    elif scale_type == 'Robust':
        for i in rX_train_scaled:
            train_copy[i] = rX_train_scaled[i].values
            test_copy[i] = rX_test_scaled[i].values
    #if Standard was specified, insert the standardized scaled data into the train and test copies
    elif scale_type == 'Standard':
          for i in sX_train_scaled:
            train_copy[i] = sX_train_scaled[i].values
            test_copy[i] = sX_test_scaled[i].values
            
    #return the train and test copy sets with scaled data
    return train_copy, test_copy
 
