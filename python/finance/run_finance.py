#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:12:18 2019

@author: weiya
"""

#import csv
import pandas as pd
#companies = []
#with open("constituents.csv") as csvFile:
#    reader = csv.reader(csvFile)
#    for row in reader:
#        if row[0] == 'Symbol':
#            pass
#        else:
#            companies.append(row[0])
#csvFile.close()

import requests
from bs4 import BeautifulSoup
req = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
page = req.content
soup = BeautifulSoup(page, "html.parser")
# refer to https://stackoverflow.com/questions/19591720/python-beautiful-soup-parse-a-table-with-a-specific-id
table = soup.find('table', id="constituents")
rows = table.findAll('tr')
companies = []
for row in rows[1:]:
    companies.append(row.td.text.strip().replace('.','-'))


for i in range(len(companies)):
#for i in range(10):
    try:        
        data = Fetcher(companies[i], [2010, 1, 1], [2019, 8, 19])
    except:
        print(f"{i} failed !!!!!!!!!!!!")
        continue
    
    ntries = 0
    noway = False
    while True:
        res = data.getHistorical()
        if len(res) == 0: # empty dataframe
            print(f"{companies[i]}: {ntries} retry... ")
        else:
            break
        ntries += 1
        if ntries > 10:
            print("No way!!!!!!!!!!")
            noway = True
            break
    if noway:
        continue
    try:
        if i == 0:
            open_prices = pd.concat([res.Open], axis = 1)
            close_prices = pd.concat([res.Close], axis = 1)
        else:
            open_prices = pd.concat([open_prices, res.Open], axis = 1)
            close_prices = pd.concat([close_prices, res.Close], axis = 1)
    except:
        print(i)
#    close_prices.append([res.Close])
    
open_prices.to_csv("open_prices.csv")
close_prices.to_csv("close_prices.csv")
    