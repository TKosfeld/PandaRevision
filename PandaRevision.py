#!/usr/bin/env python
# coding: utf-8

# In[1]:


# consider replacing hardcoded path with input selection, Can Jupyter Notebook utilize argparse?
import os
PATH = os.path.join("HumanData")

# populates list with all .csv files in hardcoded folder, does not differentiate between m, m_2, and m_5 data.
filelist = []
for file in os.listdir(PATH):
    if file.endswith(".csv"):
        filelist.append(file)
# housekeeping performance checks
print(len(filelist))
print(filelist)

# creation and population of m, m_2, and m_5 filelists
mfilelist = []
m2filelist = []
m5filelist = []
for file in filelist:
    if "_2" not in file and "_5" not in file:
        mfilelist.append(file)
    if "_2" in file:
        m2filelist.append(file)
    if "_5" in file:
        m5filelist.append(file)
#housekeeping performance checks
print(len(mfilelist))
print(len(m2filelist))
print(len(m5filelist))
print(mfilelist)
print(m2filelist)
print(m5filelist)

# creation and population of corresponding namelists, through snipping of various filelists
mnamelist = []
m2namelist = []
m5namelist = []
for file in mfilelist:
    mnamelist.append(file[:-4])
for file in m2filelist:
    m2namelist.append(file[:-4])
for file in m5filelist:
    m5namelist.append(file[:-4])
# housekeeping performance checks
print(mnamelist)
print(m2namelist)
print(m5namelist)
print(len(mnamelist))
print(len(m2namelist))
print(len(m5namelist))


# In[2]:


import pandas as pd

# oft implemented read csv function
def load_data(filename, path=PATH):
    csv_path = os.path.join(path, filename)
    return pd.read_csv(csv_path)


# In[3]:


# dictionary of matching data variables and csv assignments
mdata = {}

# populates dictionary for every m format data file in mfilelist
for x in range(len(mfilelist)):
    mdata["data{0}".format(x)] = load_data(mfilelist[x])
# housekeeping performance check
# print(mdata["data0"])


# In[4]:


for y in range(len(mdata)):
    mdata["data" + str(y)].columns = ["aminoAcid", mnamelist[y]]
    mdata["data" + str(y)] = mdata["data" + str(y)].groupby(["aminoAcid"], as_index = False).agg({mnamelist[y]: "sum"})


# In[5]:


m2data = {}
for x in range(len(m2filelist)):
    m2data["m2data{0}".format(x)] = load_data(m2filelist[x])
print(m2data)


# In[6]:


for y in range(len(m2data)):
    m2data["m2data" + str(y)].columns = ["aminoAcid", m2namelist[y]]
    m2data["m2data" + str(y)] = m2data["m2data" + str(y)].groupby(["aminoAcid"], as_index = False).agg({m2namelist[y]: "sum"})
    


# In[7]:


m5data = {}
for x in range(len(m5filelist)):
    m5data["m5data{0}".format(x)] = load_data(m5filelist[x])
print(m5data)


# In[8]:


for y in range(len(m5data)):
    m5data["m5data" + str(y)].columns = ["aminoAcid", m5namelist[y]]
    m5data["m5data" + str(y)] = m5data["m5data" + str(y)].groupby(["aminoAcid"], as_index = False).agg({m5namelist[y]: "sum"})
    


# In[9]:


datam = pd.merge(mdata["data0"], mdata["data1"], on = "aminoAcid", how = "outer")
for data in mdata:
    if data == "data0" or data == "data1":
       continue
    datam = pd.merge(datam, mdata[data], on = "aminoAcid", how = "outer")
    print(data)
for data in m2data:
    datam = pd.merge(datam, m2data[data], on = "aminoAcid", how = "outer")
    print(data)
for data in m5data:
    datam = pd.merge(datam, m5data[data], on = "aminoAcid", how = "outer")
    print(data)
datam.to_csv("datam.gz", compression ="gzip")


# In[ ]:


datam


# In[ ]:





trainers = ["KJW100_HLA-A2_6_PRE.tsv.chopped","KJW100_HLA-A2_7_PRE.tsv.chopped","KJW100_HLA-A2_8_PRE.tsv.chopped","KJW100_HLA-A2_9_PRE.tsv.chopped","KJW102_HLA-A2_11_PRE.tsv.chopped","KJW102_HLA-A2_16_PRE.tsv.chopped","KJW102_HLA-A2_17_PRE.tsv.chopped","KJW102_HLA-A2_18_PRE.tsv.chopped","KJW102_HLA-A2_33_PRE.tsv.chopped","KJW102_HLA-A2_35_PRE.tsv.chopped","KJW102_HLA-A2_37_PRE.tsv.chopped","KJW102_HLA-A2_40_PRE.tsv.chopped","KJW103_HLA-A2_41_PRE.tsv.chopped","KJW103_HLA-A2_43_PRE.tsv.chopped","KJW103_HLA-A2_45_PRE.tsv.chopped","KJW103_HLA-A2_46_PRE.tsv.chopped","KJW103_HLA-A2_49_PRE.tsv.chopped","KJW103_HLA-A2_50_PRE.tsv.chopped","KJW103_HLA-A2_52_PRE.tsv.chopped","KJW103_HLA-A2_53_PRE.tsv.chopped","KJW103_HLA-A2_54_PRE.tsv.chopped","KJW103_HLA-A2_57_PRE.tsv.chopped","KJW103_HLA-A2_59_PRE.tsv.chopped","KJW100_HLA-A2_1_14_day.tsv.chopped","KJW100_HLA-A2_10_14_day.tsv.chopped","KJW100_HLA-A2_2_14_day.tsv.chopped","KJW100_HLA-A2_20_14_day.tsv.chopped","KJW100_HLA-A2_22_14_day.tsv.chopped","KJW100_HLA-A2_23_14_day.tsv.chopped","KJW100_HLA-A2_24_14_day.tsv.chopped","KJW100_HLA-A2_25_14_day.tsv.chopped","KJW100_HLA-A2_26_14_day.tsv.chopped","KJW100_HLA-A2_27_14_day.tsv.chopped","KJW100_HLA-A2_28_14_day.tsv.chopped","KJW100_HLA-A2_29_14_day.tsv.chopped","KJW100_HLA-A2_3_14_day.tsv.chopped","KJW100_HLA-A2_6_14_day.tsv.chopped","KJW100_HLA-A2_7_14_day.tsv.chopped","KJW100_HLA-A2_8_14_day.tsv.chopped","KJW100_HLA-A2_9_14_day.tsv.chopped","KJW103_HLA-A2_41_14_day.tsv.chopped","KJW103_HLA-A2_43_14_day.tsv.chopped","KJW103_HLA-A2_44_14_day.tsv.chopped","KJW103_HLA-A2_50_14_day.tsv.chopped","KJW103_HLA-A2_52_14_day.tsv.chopped","KJW103_HLA-A2_53_14_day.tsv.chopped","KJW103_HLA-A2_54_14_day.tsv.chopped","KJW100_1_8wk.tsv.chopped","KJW100_10_8wk.tsv.chopped","KJW100_20_8wk.tsv.chopped","KJW100_21_8wk.tsv.chopped","KJW100_23_8wk.tsv.chopped","KJW100_25_8wk.tsv.chopped","KJW100_26_8wk.tsv.chopped","KJW100_27_8wk.tsv.chopped","KJW100_28_8wk.tsv.chopped","KJW100_29_8wk.tsv.chopped","KJW100_3_8wk.tsv.chopped","KJW100_4_8wk.tsv.chopped","KJW100_6_8wk.tsv.chopped","KJW100_7_8wk.tsv.chopped","KJW100_8_8wk.tsv.chopped","KJW100_9_8wk.tsv.chopped","KJW103_HLA-A2_42_8wk.tsv.chopped","KJW103_HLA-A2_44_8wk.tsv.chopped","KJW103_HLA-A2_45_8wk.tsv.chopped","KJW103_HLA-A2_50_8wk.tsv.chopped","KJW103_HLA-A2_52_8wk.tsv.chopped","KJW103_HLA-A2_53_8wk.tsv.chopped","KJW103_HLA-A2_54_8wk.tsv.chopped"]

predictor = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
idx =0
better_data.insert(loc=idx, column='Predictor', value=predictor)


better_data.head(90)attributes = better_data.columns[1:]
attributes

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('std_scaler',StandardScaler())
])

X = better_data[attributes]
xfit = num_pipeline.fit_transform(X)

import numpy as np
from sklearn.model_selection import train_test_split

testerz = ["KJW102_HLA-A2_31_PRE.tsv.chopped","KJW103_HLA-A2_51_8wk.tsv.chopped","KJW102_HLA-A2_39_PRE.tsv.chopped", "KJW103_HLA-A2_44_PRE.tsv.chopped", "KJW103_HLA-A2_42_14_day.tsv.chopped","KJW103_HLA-A2_43_8wk.tsv.chopped","KJW100_HLA-A2_29_PRE.tsv.chopped","KJW103_HLA-A2_45_14_day.tsv.chopped", "KJW100_2_8wk.tsv.chopped","KJW100_HLA-A2_4_14_day.tsv.chopped","KJW100_22_8wk.tsv.chopped","KJW103_HLA-A2_51_PRE.tsv.chopped", "KJW100_HLA-A2_21_14_day.tsv.chopped","KJW103_HLA-A2_42_PRE.tsv.chopped", "KJW100_HLA-A2_10_PRE.tsv.chopped", "KJW102_HLA-A2_38_PRE.tsv.chopped", "KJW103_HLA-A2_41_8wk.tsv.chopped","KJW103_HLA-A2_58_PRE.tsv.chopped","KJW100_24_8wk.tsv.chopped", "KJW103_HLA-A2_51_14_day.tsv.chopped"]
test_data = better_data.T[testerz]
test_data = test_data.T
test_data

flip = better_data.T
train_data= flip[trainers]
train_data= train_data.T
train_data

