# from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

# Read the file
Path = 'C:/Users/DaiYi/Desktop/2_data.txt'
Date = []
SplitedLine = []
for OriginalLine in open(Path):
    Splited = OriginalLine.replace('      ', ',').split(',')
    SplitedLine.append(Splited[0])
SplitedLine = SplitedLine[0:-1]
Date = [int(i) for i in SplitedLine]
Date.append(20081231)
SplitedLine = []
for OriginalLine in open(Path):
    Splited = OriginalLine.replace('     ', ',').split(',')
    SplitedLine.append(Splited[2])
Vwrtn = []
for i in SplitedLine:
    Si = i.split()
    Vwrtn = [float(i) for i in SplitedLine]


dta = pd.Series(Vwrtn)
dta.index = pd.Index(Date)
dta.plot(figsize=(12, 8))
# dta.index = pd.Index(sm.tsa.datetools.dates_from_range('192601', '200812'))

# date.append(datetime.datetime.strptime(h[0], ''))
# Thisline = []
# for line in open(Path):
#     line = line.replace('\n', '').split("	")
#     Thisline.append(line[0])
# print Thisline
总体
