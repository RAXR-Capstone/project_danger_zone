import pandas as pd
import numpy as np
import re
import requests 
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
import selenium.webdriver.support.ui as ui
import selenium.webdriver.support.expected_conditions as EC
import os
import time
import datetime


options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
dir_path = os.path.dirname(os.path.realpath('chromedriver'))
chromedriver = dir_path + '/chromedriver'
os.environ['webdriver.chrome.driver'] = chromedriver

#continue where left off if scraping file already exists
df = pd.DataFrame()
if os.path.isfile('accident_links_continued.csv'):
    df = pd.read_csv('accident_data.csv')
    print("Continuing where left off")
    links = pd.read_csv('accident_links_continued.csv')
else:
    links = pd.read_csv('accident_links.csv')
links = links.drop_duplicates()
for i in range(len(links)):
    list_of_dicts = []
    print(links.link[i])
    url = links.link[i]
    driver = webdriver.Chrome(executable_path=chromedriver)     
    driver.get(url)
    ui.WebDriverWait(driver, 15).until(EC.visibility_of_all_elements_located((By.ID, 'ACCIDENT')))
    
    ######ACCIDENT SUMMARY SPECIFICS HERE###############
    
    containers = driver.find_elements_by_id('accident-top')
    for container in containers:
        variables = container.text.split('\n')
        caseId = variables[1].split(':')
        crashId = variables[2].split(':')
        
    containers = driver.find_elements_by_id('accident-header')
    for container in containers:
        variables = container.text.split('\n')
        city = variables[1]
        date = variables[3]
        Police = variables[5]
        
    containers = driver.find_element_by_css_selector('div[style="position: relative;"]').find_element_by_tag_name('img').get_attribute('src')
    split = containers.split('=')
    latlong = split[1]
    latlong = latlong.replace('&style', '').split(',')
    latitude = latlong[0]
    longitude = latlong[1]
        
    containers = driver.find_elements_by_id('conditions')        
    for container in containers:
        variables = container.text.split('\n')
        if variables[variables.index('Road & Traffic Conditions') + 1] == 'Weather':
                variables.insert(variables.index('Road & Traffic Conditions') + 1, np.nan)
        if variables[-1] == 'Weather':
                variables.insert(variables.index('Weather')+1, np.nan)
        trafficConditions = variables[variables.index('Road & Traffic Conditions') + 1]
        weatherConditions = variables[variables.index('Weather') + 1]
 
    containers = driver.find_elements_by_class_name('row')
    summary = containers[0].text.split('\n')
    accident_factor = ''
    for i in range(len(summary)):
        if i > 0 and i < summary.index('Date & Time Of Crash'):
            accident_factor += summary[i] + ' '
        speedlimit = summary[summary.index("Speed Limit") + 1]
        accidentLocation = summary[-1]
        injuryNumber = summary[summary.index('Total Number of Injuries')-1]
        numberofVehicles = summary[summary.index('Total Number of Vehicles') -1]
        numberofOccupants = summary[summary.index('Total Number of Occupants')-1]
        
        
        
    #######CAR SPECIFIC DATA########
                
  
    
    for i in range(int(numberofVehicles)):        
            
        container = driver.find_elements_by_class_name('occupant')  
        table = container[i].find_element_by_tag_name('table').text.replace('\n', ' ').lower()
    
        if 'possible injury' in table or 'minor injury' in table:
            car_contained_injury = 1 
        else:
            car_contained_injury = 0
            
        # tie car to if the injury occured in the current vehicle being parsed        
        
        container = driver.find_elements_by_class_name('driver-vehicle')
        if len(container[i].find_elements_by_class_name('at-fault')) == 0:
            atFault = 0
        else:
            atFault = 1
                                            
        variables = container[i].text.split('\n')
        #ignore compensation text box, look for those elements in variables and remove them fromthe top of the list, then parse
        if 'Based on' in variables[0] and 'Accident' in variables[1]:
            variables.pop(0)
            variables.pop(0)  
            #catch missing color
        car = variables[0]
        if ',' not in car:
            carColor = np.nan
            variables.insert(1, np.nan)
        else:
            car = variables[0].split(',')
            carColor = car[1]
            #catch missing car info
        if car[0] == '':
            carMake = np.nan
            carYear = np.nan
            variables.insert(0, np.nan)
        else:
            #catch missing year
            if '-' not in car[0]:       
                carMake = car[0]
                carYear = np.nan
            else:
                makeYear = car[0].split('-')
                carMake = makeYear[0]
                carYear = makeYear[1]




        #if the next index is the next key and not a value for the current key, then the value is missing, fill with nan
        if variables[variables.index('Age') + 1] == 'Gender':
            variables.insert(variables.index('Age') + 1, np.nan)
        if variables[variables.index('Gender') + 1] == "Ethnicity":
            variables.insert(variables.index('Gender')+1, np.nan)
        if variables[variables.index('Ethnicity') + 1] == "Residence Of":
            variables.insert(variables.index('Ethnicity')+1, np.nan)
        if variables[variables.index('Residence Of') + 1] == "Damage Area":
            variables.insert(variables.index('Residence Of')+1, np.nan)
        if variables[variables.index('Damage Area') + 1] == "Driver License Type":
            variables.insert(variables.index('Damage Area')+1, np.nan)
        if variables[variables.index('Driver License Type') + 1] == "Vehicle License State ID":
            variables.insert(variables.index('Driver License Type')+1, np.nan)
        if 'VIN' not in variables:
            variables.insert(variables.index('Vehicle License State ID')+1, 'VIN') 
        if variables[variables.index('Vehicle License State ID') + 1] == "VIN":
            variables.insert(variables.index('Vehicle License State ID')+1, np.nan)
        if variables[variables.index('VIN') + 1] == "Insured":
            variables.insert(variables.index('VIN')+1, np.nan)
        if variables[variables.index('Insured') + 1] == "Towing Company":
            variables.insert(variables.index('Insured')+1, np.nan)
        if variables[-1] == 'Towing Company':
            variables.insert(variables.index('Towing Company')+1, np.nan)


        #now that the index lengths are uniform, we can assign the values
        driverAge = variables[variables.index('Age') + 1]
        driverGender = variables[variables.index('Gender') + 1]
        driverEthnicity = variables[variables.index('Ethnicity') + 1]
        driverResidence = variables[variables.index('Residence Of') + 1]
        driverCarDamage = variables[variables.index('Damage Area') + 1]
        driverLicenseType = variables[variables.index('Driver License Type') + 1]
        driverLicenseState = variables[variables.index('Vehicle License State ID') + 1]
        drivercarVIN = variables[variables.index('VIN') + 1]
        driverInsured = variables[variables.index('Insured') + 1]


        #store all values into a dictionary here, all this information is one observation
        #done within this indnent brecause each car in the accident is an onservation. so cars in the same accident will share the same information.
        dictionary = {'crash_url':url, 'case_id': caseId[1], 'crash_id' : crashId[1], 'crash_city': city, 'crash_date':date, 'crash_latitude': latitude, 'crash_longitude':longitude, 'police_dept': Police, 'accident_factor': accident_factor, 'speed_limit' : speedlimit \
                      ,'crash_location': accidentLocation, 'num_of_injuries': injuryNumber, 'num_of_vehicles': numberofVehicles , 'num_of_occupants': numberofOccupants, 'at_fault' : atFault \
                      ,'car_contained_injury': car_contained_injury
                      ,'car_make':carMake, 'car_year': carYear, 'car_color': carColor,'car_vin':drivercarVIN ,'driver_age':driverAge, 'driver_gender': driverGender, 'driver_ethnicity' : driverEthnicity \
                      ,'driver_residence': driverResidence,'driver_car_damage':driverCarDamage,'driver_license_type':driverLicenseType, 'driver_license_state': driverLicenseState\
                      ,'driver_insured': driverInsured, 'traffic_conditions':trafficConditions, 'weather_conditions': weatherConditions}
       
        
        list_of_dicts.append(dictionary)
    df = df.append(list_of_dicts)
    df.to_csv('accident_data.csv', index=False)
    
    
    
    links = links.iloc[1: , :]
    links.to_csv('accident_links_continued.csv',index=False)
    print('{} links to go'.format(len(links)))
    
    driver.close()
    

    
    

    
    
    
    
