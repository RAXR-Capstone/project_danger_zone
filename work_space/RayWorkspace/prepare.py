#Z0096


# import standard libraries
import pandas as pd
import numpy as np
import re


#################### Prep Driver Data ####################


def create_dl_classes(df):
    '''

    Use existing "driver_license_type" column to derive boolean columns
    for if driver as a CDL, is unlicensed, and has a Class A, B, and or
    M driver's license.

    '''
    
    # create cdl boolean column where DL type contains "commericial"
    df['cdl'] = df.driver_license_type.apply(lambda row: \
                    1 if str(row).lower().__contains__('commercial') else 0)
    # create unlicensed bool column wher DL type contains "unlicensed"
    df['unlicensed'] = df.driver_license_type.apply(lambda row: \
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


def abbr_driver_race(df):
    '''
    
    Takes existing racial data on driver and converts to a simplified
    US Census style abbreviation of race, replaces missing values with
    np.nan for future imputation or other handling

    '''

    # replace unknown with NaN
    df.driver_ethnicity = np.where(df.driver_ethnicity == 'Unknown', np.nan,
                                                        df.driver_ethnicity)
    # convert longer racial identifier to census style abbreviations
    df.driver_ethnicity = np.where(df.driver_ethnicity == 'Hispanic', 'H',
                                                        df.driver_ethnicity)
    df.driver_ethnicity = np.where(df.driver_ethnicity == 'White', 'W',
                                                        df.driver_ethnicity)
    df.driver_ethnicity = np.where(df.driver_ethnicity == 'Black', 'B',
                                                        df.driver_ethnicity)
    df.driver_ethnicity = np.where(df.driver_ethnicity == 'Asian', 'A',
                                                        df.driver_ethnicity)
    df.driver_ethnicity = np.where(df.driver_ethnicity == \
                    'Amer. indian/alaskan native', 'N', df.driver_ethnicity)
    df.driver_ethnicity = np.where(df.driver_ethnicity.isin(['Other']), 'O',
                                                        df.driver_ethnicity)

    return df


def clean_driver_age(df):
    '''

    Converts driver_age column into integer data type and replaces
    missing or inappropriate values with -1 for future imputing or
    other means of handling

    '''

    # create mask where driver's age is non-digit
    mask = df.driver_age.apply(lambda row: bool(
                                re.search(r'\D*\d+\D*', str(row))))
    # replace non-digit drivers age with -1
    df.driver_age = np.where(df.driver_age.isin(df[mask].driver_age),
                                                df.driver_age, -1)
    # replace driver_age less than 6 with -1
    df.driver_age = np.where(df.driver_age >= 10, df.driver_age, -1)
    # set driver_age data type as integer
    df.driver_age = df.driver_age.astype('int')

    return df


def prep_driver_data(df):
    '''

    Uses functions to prepare driver data with cleaned up demographics
    and drivers license data encoded variables for class and type, or
    if the driver was confirmed as unlicensed. Drops unneeded column
    used to derive other features

    '''

    # replace "Unknown" gender with NaN
    df.driver_gender = np.where(df.driver_gender == 'Unknown', np.nan,
                                                    df.driver_gender)
    # set DL state as Texas, Other, or NaN
    df.driver_license_state = np.where(df.driver_license_state.isin(\
                    ['UNKNOWN', 'Unknown']), np.nan, df.driver_license_state)
    df.driver_license_state = np.where(df.driver_license_state.isin(\
                    ['Texas', np.nan]), df.driver_license_state, 'Other')
    df = df.rename(columns={'driver_license_state':'dl_state'})
    # use function to create DL class bool columns
    df = create_dl_classes(df)
    # use function to clean driver_ethnicity column
    df = abbr_driver_race(df)
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
    
    # replaces Xs for improved use with NTHSA cross reference
    df.car_vin = df.car_vin.apply(lambda row: re.sub(r'(\w{4})X{9}(\d{4})',
                                                     r'\1*********\2', row))
    # set mask for appropriate VIN values
    mask = df.car_vin.apply(lambda row: bool(re.search(r'(\w{4})\*{9}(\d{4})',
                                                                        row)))
    # replace inappropriate VIN values with "Unknown"
    df.car_vin = np.where(df.car_vin.isin(df[mask].car_vin),df.car_vin, np.nan)
    # rename column
    df = df.rename(columns={'car_vin':'vin'})
    
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
    values. Replaces value with -1 for vehicles without known
    manufacutring year for later imputation or other handling.

    '''
    
    # pull numerical, four digit vehicle manufacture years
    df.car_year = df.car_year.apply(lambda row: re.sub(r'\s?(\d{4})(.0)?',
                                                        r'\1', str(row)))
    # set mislabeled and unknown years as -1
    df.car_year = df.car_year.apply(lambda row: re.sub(r'(\s*[A-Z\sa-z]+\s*)',
                                                            r'-1', str(row)))
    # convert data type to integer
    df.car_year = df.car_year.astype('int')
    # rename column
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
                                re.search(r'\W*(\w+)[\W]*', row).group(1))
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
    df['vehicle_dmg_zone'] = np.where(impact_type\
                        .isin(['VX', 'MC']), 0, -1)
    df['vehicle_dmg_zone'] = np.where(impact_type\
                        .isin(['FL','FR','FC','FD']), 1, df.vehicle_dmg_zone)
    df['vehicle_dmg_zone'] = np.where(impact_type\
                        .isin(['BL','BR','BC','BD']), 2, df.vehicle_dmg_zone)
    df['vehicle_dmg_zone'] = np.where(impact_type\
                        .isin(['LFQ']), 3, df.vehicle_dmg_zone)
    df['vehicle_dmg_zone'] = np.where(impact_type\
                        .isin(['LP', 'LD', 'L&T']), 4, df.vehicle_dmg_zone)
    df['vehicle_dmg_zone'] = np.where(impact_type\
                        .isin(['LBQ']), 5, df.vehicle_dmg_zone)
    df['vehicle_dmg_zone'] = np.where(impact_type\
                        .isin(['RFQ']), 6, df.vehicle_dmg_zone)
    df['vehicle_dmg_zone'] = np.where(impact_type\
                        .isin(['RD', 'RP', 'R&T']), 7, df.vehicle_dmg_zone)
    df['vehicle_dmg_zone'] = np.where(impact_type\
                        .isin(['RBQ']), 8, df.vehicle_dmg_zone)
    df['vehicle_dmg_zone'] = np.where(impact_type\
                        .isin(['TP','L&T','R&T','VB']), 9, df.vehicle_dmg_zone)
    df['vehicle_dmg_zone'] = np.where(impact_type\
                        .isin(['MC']), 10, df.vehicle_dmg_zone)
    df.vehicle_dmg_zone = df.vehicle_dmg_zone.astype('int')

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
    df['concentrated_damage'] = np.where(impact_type.isin(['FC', 'BC']), 1, 0)
    df['distributed_damage'] = np.where(impact_type.isin(
                                            ['FD', 'BD', 'LD', 'RD']), 1, 0)
    df['rollover_damage'] = np.where(impact_type.isin(
                                            ['TP', 'L&T', 'R&T']), 1, 0)
    df['vehicle_burned'] = np.where(impact_type.isin(['VB']), 1, 0)

    return df


def prep_damage_data(df):
    '''

    Uses functions to create column with area of damage for the vehicle and
    columns of possible types of damage incurred, drops driver_car_damage
    column that contains lengthy string descriptor

    '''

    # use function to make vehicle_dmg_zone column
    df = make_vehicle_dmg_zone(df)
    # use function to make damage type columns
    df = make_dmg_type_columns(df)
    # drop wordy column used to construct above
    df = df.drop(columns='driver_car_damage')

    return df
