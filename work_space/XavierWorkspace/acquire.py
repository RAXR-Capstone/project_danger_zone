import pandas as pd
import numpy as np
import re
import requests, json
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
import selenium.webdriver.support.ui as ui
import selenium.webdriver.support.expected_conditions as EC
import os
import time
import datetime
import skip




'''
Date: August 13, 2021
Author: Xavier Carter
This aquire function is script using the selinium webscraper and a 
chrome driver extention in order to extract the information found within the 
web page and stores it into a csv file. The webscraper takes in a list of 
links and parses through the list of links in order to know which web browser to open and scrape. 
If for any reason the link is broken, then the skip.py script is ran and  
removes the broken link from the list, allowing to continue without breaking the loop. 
of for any reason the webscraper fin allowing you to continue where left off.

'''


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
    links = pd.read_csv('accident_links_updated.csv')
links = links.drop_duplicates()
for i in range(len(links)):
    list_of_dicts = []
    print(links.link[i])
    url = links.link[i]
    driver = webdriver.Chrome(executable_path=chromedriver)     
    driver.get(url)
    
    try:
        #wrapping scraper in a try catch block, if webpage never loads with HTML tag ACCIDENT , then the webpage could not load and the file will be skipped
        ui.WebDriverWait(driver, 15).until(EC.visibility_of_all_elements_located((By.ID, 'ACCIDENT')))
    
        ######ACCIDENT SUMMARY SPECIFICS HERE###############
        # Will look at the header of the webpage to grab the crashID and Case ID
        containers = driver.find_elements_by_id('accident-top')
        for container in containers:
            variables = container.text.split('\n')
            caseId = variables[1].split(':')
            crashId = variables[2].split(':')
        #will look at the header to check to grab the location of the accident, the date of the accident and the police department that 
        
        containers = driver.find_elements_by_id('accident-header')
        for container in containers:
            variables = container.text.split('\n')
            city = variables[1]
            date = variables[3]
            Police = variables[5]
        #will look into the google chrome map picture in order to grab the latitude and longtitude of the crash  location
        containers = driver.find_element_by_css_selector('div[style="position: relative;"]').find_element_by_tag_name('img').get_attribute('src')
        split = containers.split('=')
        latlong = split[1]
        latlong = latlong.replace('&style', '').split(',')
        latitude = latlong[0]
        longitude = latlong[1]
        
        #will look into the conditions container to grab the weather at the time of the accident and the traffic conditions at the time
        containers = driver.find_elements_by_id('conditions')        
        for container in containers:
            variables = container.text.split('\n')
            if variables[variables.index('Road & Traffic Conditions') + 1] == 'Weather':
                variables.insert(variables.index('Road & Traffic Conditions') + 1, np.nan)
            if variables[-1] == 'Weather':
                variables.insert(variables.index('Weather')+1, np.nan)
            trafficConditions = variables[variables.index('Road & Traffic Conditions') + 1]
            weatherConditions = variables[variables.index('Weather') + 1]
         
        #looking into the first container and grabbing the accident specic
        containers = driver.find_elements_by_class_name('row')
        summary = containers[0].text.split('\n')
        accident_factor = ''
        for i in range(len(summary)):
            if i > 0 and i < summary.index('Date & Time Of Crash'):
                accident_factor += summary[i] + ' '
            speedlimit = summary[summary.index("Speed Limit") + 1]
            accidentLocation = summary[-1]
            injuryNumber = summary[summary.index('Total Number of Injuries')-1]
            numberofVehicles = int(summary[summary.index('Total Number of Vehicles') -1])
            numberofOccupants = int(summary[summary.index('Total Number of Occupants')-1])
        
        

        #########CAR SPECIFIC DATA########
                
  
        #check to see if common words that indicate an injury occur within the table, if they do
        #then someone was injured in the accident and reported it
        for i in range(int(numberofVehicles)):        
            
            container = driver.find_elements_by_class_name('occupant')  
            table = container[i].find_element_by_tag_name('table').text.replace('\n', ' ').lower()
    
            if 'possible injury' in table or 'minor injury' in table:
                car_contained_injury = 1 
            else:
                car_contained_injury = 0
        
       
            #check to see how many occupants there are in the car at the time of the accident 
            table = container[i].find_element_by_tag_name('table').text
            table = table.split('\n')
        
            #if occupant table doesnt exist, assume at least one driver
            occupant = 0
            for row in table:
                if row.isdigit():
                    occupant += 1
            if occupant == 0:
                occupant = 1
            if numberofOccupants == 0 and  occupant > 0:
                numberofOccupants += occupant
    
        
        #check to see if the table contains any red circles, if the red circle exists and contains the airbag picture, then at least one airbag was deployed.
        #if not, then the airbags were not deployed.
            bag_deployed = 0
            try:
                red_circles = container[i].find_element_by_class_name('circle.red').find_elements_by_tag_name('img')
                for red_circle in red_circles:
                    if red_circle.get_attribute('src') == 'https://app.myaccident.org/assets/icon-airbag.svg':
                        bag_deployed = 1
        #if the red airbag logo doesnt exist within the table, then airbags were not deployed
            except:
                bags_deployed = 0
            
        # if the at fault picture is within the container, then that specific car was at fault.
            container = driver.find_elements_by_class_name('driver-vehicle')
            if len(container[i].find_elements_by_class_name('at-fault')) == 0:
                atFault = 0
            else:
                atFault = 1
                                            
            variables = container[i].text.split('\n')
            #ignore compensation text box, look for those elements in variables and remove them from the top of the list, then parse
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
                carYear = 0
                carMake = ''
            #catch missing year
                if '-' not in car[0]:       
                    carMake = car[0]
                    carYear = np.nan
                else:
                    #catch hyphenated car makes (example: Mercades-benz)
                    makeYear = car[0].split('-')
                    for i in makeYear:
                        check = i.strip()
                        if check.isdigit():
                            carYear = check
                        else:
                            carMake += check + " "

                            
    


        #Grabbing the driver specific information, if the variable is missing then fill it with null.

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
            
            
        #use the vin number to find the vehicle type form the vin website api
            vin_url = 'https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/'
            post_fields = {'format': 'json', 'data': drivercarVIN};
            r = requests.post(vin_url, data=post_fields);
            VehicleType = r.json()['Results'][0]['VehicleType']

        # done within this indnent brecause each car in the accident

        # store all values into a dictionary here, all this information is one observatio is one observation. so cars in the same accident   
        # will share some of the same chrash information.
            dictionary = {'crash_url':url\
                          ,'case_id': caseId[1]\
                          ,'crash_id' : crashId[1]\
                          ,'crash_city': city\
                          ,'crash_date':date\
                          ,'crash_latitude': latitude\
                          ,'crash_longitude':longitude\
                          ,'police_dept': Police\
                          ,'accident_factor': accident_factor\
                          ,'speed_limit' : speedlimit\
                          ,'crash_location': accidentLocation\
                          ,'num_of_injuries': injuryNumber\
                          ,'num_of_vehicles': numberofVehicles\
                          ,'num_of_occupants': numberofOccupants\
                          ,'at_fault' : atFault\
                          ,'car_contained_injury': car_contained_injury\
                          ,'occupants_in_car' : occupant\
                          ,'car_airbags_deployed': bag_deployed\
                          ,'car_make':carMake\
                          ,'car_year': carYear\
                          ,'car_type': VehicleType\
                          ,'car_color': carColor\
                          ,'car_vin':drivercarVIN\
                          ,'driver_age':driverAge\
                          ,'driver_gender': driverGender\
                          ,'driver_ethnicity' : driverEthnicity\
                          ,'driver_residence': driverResidence\
                          ,'driver_car_damage':driverCarDamage\
                          ,'driver_license_type':driverLicenseType\
                          ,'driver_license_state': driverLicenseState\
                          ,'driver_insured': driverInsured\
                          ,'traffic_conditions':trafficConditions\
                          ,'weather_conditions': weatherConditions}
           
        
            list_of_dicts.append(dictionary)
        df = df.append(list_of_dicts)
        df.to_csv('accident_data.csv', index=False)
    
    
    
        links = links.iloc[1: , :]
        links.to_csv('accident_links_continued.csv',index=False)
        print('{} links to go'.format(len(links)))
    
        driver.close()
    except TimeoutException as ex:
        print("Bad Link, skipping Link")
        skip.skip()
        driver.close()
        
    
    

    
    

    
    
    
    
