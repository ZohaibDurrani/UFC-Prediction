#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing

# ## About
# 
# - The data we are using is about **UFC** Fight between multiple players. Every column contains the information about the fight. i.e Date, location, Methods, Winners and also contains the information of each single round from round 1 to round 5.
# - Every row contains an information regarding the fights and contain an accurate representation of each fights.
# - Our target variable is **Winner**. Predict the Winner from the given data

# ## Essential Library

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ```python
# pd.read_csv(file_path, sep = ",", encoding="latin", parse_dates=['Date'])
# ```
# - **Encoding**: This is a type of encoding and is used to solve the UnicodeDecodeError, while attempting to read a file in Python or Pandas. latin-1 is a single-byte encoding which uses the characters 0 through 127, so it can encode half as many characters as latin1.

# In[3]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
data_path = ['data']
file_path = os.sep.join(data_path+['UFC_Train.csv'])
data = pd.read_csv(file_path, sep = ",", encoding="latin", parse_dates=['Date'])
display(data.head())


# In[4]:


data.shape  ## Checking the Shape of the Data-Set


# In[4]:


list(data.columns) ## List all the coloumns which are in data-set


# `dtypes` where we used to check the type of a data where we have different types

# In[5]:


data.dtypes


# In[6]:


data.describe() ## Describes the numbers information by taking mean, count etc


# Dropping all those columns which are not neccessary some of the columns such as we have **Fighter1 Sig.Str** and we took all those columns which contains `Sig.Str` because in the dataset we already have other columns which represents the same values. For example we have another column name **Fighter1 Sig.Str.% Total** if we compare the **Fighter1 Sig.Str** with **Fighter1 Sig.Str.% Total** we find out that deviding the number have a same percentage we get in **Fighter1 Sig.Str.% Total** So we drop these columns.

# In[7]:


col = ['Eventname', 'Place', 'Referee', 'Details', 
            'Fighter1 Sig.Str. Total', 'Fighter1 TD Total', 'Fighter1 Sig.Str. Total.1',
           'Fighter2 Sig.Str. Total', 'Fighter2 TD Total', 'Fighter2 Sig.Str. Total.1',
           'Fighter1 Sig.Str. Round1','Fighter1 TD Round1', 'Fighter1 Sig.Str. Round1.1',
           'Fighter2 Sig.Str. Round1','Fighter2 TD Round1', 'Fighter2 Sig.Str. Round1.1',
           'Fighter1 Sig.Str. Round2','Fighter1 TD Round2', 'Fighter1 Sig.Str. Round2.1',
           'Fighter2 Sig.Str. Round2','Fighter2 TD Round2', 'Fighter2 Sig.Str. Round2.1',
           'Fighter1 Sig.Str. Round3','Fighter1 TD Round3', 'Fighter1 Sig.Str. Round3.1',
           'Fighter2 Sig.Str. Round3','Fighter2 TD Round3', 'Fighter2 Sig.Str. Round3.1',
           'Fighter1 Sig.Str. Round4','Fighter1 TD Round4', 'Fighter1 Sig.Str. Round4.1',
           'Fighter2 Sig.Str. Round4','Fighter2 TD Round4', 'Fighter2 Sig.Str. Round4.1']


# In[7]:


df = data.copy() ## Making copy of dataframe because if we want to use later.


# In[10]:


df.drop(col, axis=1, inplace=True)


# In[8]:


df2 = df.copy()


# **f_p**: Taking columns with having those rows which contains `of`. The approach is to take the rows and replace it with `/`. We are taking these actions because we want to divide these numbers.

# In[9]:


f_p = ['Fighter1 Total Str. Total', 'Fighter2 Total Str. Total',
      'Fighter1 Head Total', 'Fighter1 Body Total',
      'Fighter1 Leg Total', 'Fighter1 Distance Total',
      'Fighter1 Clinch Total', 'Fighter1 Ground  Total',
      'Fighter2 Head Total', 'Fighter2 Body Total',
      'Fighter2 Leg Total', 'Fighter2 Distance Total',
      'Fighter2 Clinch Total', 'Fighter2 Ground  Total',
      'Fighter1 Total Str. Round1', 'Fighter1 Head Round1',
       'Fighter1 Body Round1','Fighter1 Leg Round1',
       'Fighter1 Distance Round1','Fighter1 Clinch Round1',
       'Fighter1 Ground Round1',
       'Fighter2 Total Str. Round1', 'Fighter2 Head Round1',
       'Fighter2 Body Round1','Fighter2 Leg Round1',
       'Fighter2 Distance Round1','Fighter2 Clinch Round1',
       'Fighter2 Ground Round1',
       'Fighter1 Total Str. Round2', 'Fighter1 Head Round2',
       'Fighter1 Body Round2','Fighter1 Leg Round2',
       'Fighter1 Distance Round2','Fighter1 Clinch Round2',
       'Fighter1 Ground Round2',
       'Fighter2 Total Str. Round2', 'Fighter2 Head Round2',
       'Fighter2 Body Round2','Fighter2 Leg Round2',
       'Fighter2 Distance Round2','Fighter2 Clinch Round2',
       'Fighter2 Ground Round2',
       'Fighter1 Total Str. Round3', 'Fighter1 Head Round3',
       'Fighter1 Body Round3','Fighter1 Leg Round3',
       'Fighter1 Distance Round3','Fighter1 Clinch Round3',
       'Fighter1 Ground Round3',
       'Fighter2 Total Str. Round3', 'Fighter2 Head Round3',
       'Fighter2 Body Round3','Fighter2 Leg Round3',
       'Fighter2 Distance Round3','Fighter2 Clinch Round3',
       'Fighter2 Ground Round3',
       'Fighter1 Total Str. Round4', 'Fighter1 Head Round4',
       'Fighter1 Body Round4','Fighter1 Leg Round4',
       'Fighter1 Distance Round4','Fighter1 Clinch Round4',
       'Fighter1 Ground Round4',
       'Fighter2 Total Str. Round4', 'Fighter2 Head Round4',
       'Fighter2 Body Round4','Fighter2 Leg Round4',
       'Fighter2 Distance Round4','Fighter2 Clinch Round4',
       'Fighter2 Ground Round4',
       'Fighter1 Total Str. Round5', 'Fighter1 Head Round5',
       'Fighter1 Body Round5','Fighter1 Leg Round5',
       'Fighter1 Distance Round5','Fighter1 Clinch Round5',
       'Fighter1 Ground Round5',
       'Fighter2 Total Str. Round5', 'Fighter2 Head Round5',
       'Fighter2 Body Round5','Fighter2 Leg Round5',
       'Fighter2 Distance Round5','Fighter2 Clinch Round5',
       'Fighter2 Ground Round5',
       'Fighter1 Sig.Str. Round5', 'Fighter1 Sig.Str. Round5.1', 
       'Fighter2 Sig.Str. Round5', 'Fighter2 Sig.Str. Round5.1',
       'Fighter1 TD Round5', 'Fighter2 TD Round5'
       
      ]

df2.get(f_p).head()


# Here replacing 'of' with '/'. `regex=True`: Whether to interpret to_replace and/or value as regular expressions. If this is True then to_replace must be a string. Alternatively, this could be a regular expression or a list, dict, or array of regular expressions in which case to_replace must be None.

# In[18]:


df2[f_p] = df2[f_p].replace('of', '/', regex=True)


# In[20]:


df2[f_p] = df2[f_p].apply(lambda r: r.replace('0 / 0', '0')) # Replacing '0 / 0' with only 0


# In[22]:


df2.fillna('0', inplace=True)


# ```python
# df3[i] = df2[i].apply(lambda x: '{0:.3f}'.format(eval(x)))
# ``` 
# Now after some data preprocessing steps, we are doing some transformation and devide the numerator with the denominator and restrict it to 3 floating point numbers

# In[23]:


df3 = {}
for i in f_p:
    df3[i] = df2[i].apply(lambda x: '{0:.3f}'.format(eval(x)))

df3 = pd.DataFrame(df3)
df3 = df3.astype(float)
df3.head()


# In[25]:


df4 = df2.copy()
df4.update(df3)  ## Updating


# In[26]:


len(df2.columns) - len(df4.columns)


# In[27]:


df4 = df4.apply(lambda r: r.replace('---', '0')) ## Cleaning --- with 0


# In[29]:


df5 = df4.copy()


# In[30]:


df5 = df5.replace('%', '', regex=True,)


# Replacing % sign from colunms

# In[42]:


r_s = ['Fighter1 Sig.Str.% Total', 'Fighter1 TD% Total', 'Fighter1 Sig.Str.% Total.1', 'Fighter2 Sig.Str.% Total',
      'Fighter2 TD% Total','Fighter2 Sig.Str.% Total.1',
       'Fighter1 Sig.Str.% Round1', 'Fighter1 TD% Round1', 'Fighter1 Sig.Str.% Round1.1',
      'Fighter2 Sig.Str.% Round1', 'Fighter2 TD% Round1', 'Fighter2 Sig.Str.% Round1.1',
      'Fighter1 Sig.Str.% Round2', 'Fighter1 TD% Round2', 'Fighter1 Sig.Str.% Round2.1',
      'Fighter2 Sig.Str.% Round2', 'Fighter2 TD% Round2', 'Fighter2 Sig.Str.% Round2.1',
      'Fighter1 Sig.Str.% Round3', 'Fighter1 TD% Round3', 'Fighter1 Sig.Str.% Round3.1',
      'Fighter2 Sig.Str.% Round3', 'Fighter2 TD% Round3', 'Fighter2 Sig.Str.% Round3.1',
      'Fighter1 Sig.Str.% Round4', 'Fighter1 TD% Round4', 'Fighter1 Sig.Str.% Round4.1',
      'Fighter2 Sig.Str.% Round4', 'Fighter2 TD% Round4', 'Fighter2 Sig.Str.% Round4.1',
      'Fighter1 Sig.Str.% Round5', 'Fighter1 TD% Round5', 'Fighter1 Sig.Str.% Round5.1',
      'Fighter2 Sig.Str.% Round5', 'Fighter2 TD% Round5', 'Fighter2 Sig.Str.% Round5.1']

df5[r_s] = df5[r_s].replace('%', '', regex=True)


# In[43]:


df5[r_s] = df5[r_s].apply(lambda r: r.replace('---', '0'))


# Changing type of the following % Columns from String to floating point number

# In[44]:


df5[r_s] = df5[r_s].astype(np.float64)


# In[46]:


df5[r_s] = df5[r_s].apply(lambda x: x/100)


# Replacing `0` in Winner Colunm with `Draw` 

# In[50]:


df5['Winner'] = df5['Winner'].apply(lambda r: r.replace('0', 'Draw'))


# Changing types from `str` to `float`

# In[55]:


obj_col = ['Fighter1 KD Total',
       'Fighter1 Total Str. Total', 'Fighter1 Subb.Att Total',
       'Fighter1 Rev. Total', 'Fighter2 KD Total',
       'Fighter2 Total Str. Total', 'Fighter2 Subb.Att Total',
       'Fighter2 Rev. Total','Fighter1 KD Round1',
       'Fighter1 Total Str. Round1', 'Fighter1 Subb.Att Round1',
       'Fighter1 Rev. Round1', 'Fighter2 KD Round1',
       'Fighter2 Total Str. Round1', 'Fighter2 Subb.Att Round1',
       'Fighter2 Rev. Round1', 'Fighter1 KD Round2',
       'Fighter1 Total Str. Round2', 'Fighter1 Subb.Att Round2',
       'Fighter1 Rev. Round2', 'Fighter2 KD Round2',
       'Fighter2 Total Str. Round2', 'Fighter2 Subb.Att Round2',
       'Fighter2 Rev. Round2', 'Fighter1 KD Round3',
       'Fighter1 Total Str. Round3', 'Fighter1 Subb.Att Round3',
       'Fighter1 Rev. Round3', 'Fighter2 KD Round3',
       'Fighter2 Total Str. Round3', 'Fighter2 Subb.Att Round3',
       'Fighter2 Rev. Round3', 'Fighter1 KD Round4',
       'Fighter1 Total Str. Round4', 'Fighter1 Subb.Att Round4',
       'Fighter1 Rev. Round4', 'Fighter2 KD Round4',
       'Fighter2 Subb.Att Round4', 'Fighter2 Rev. Round4',
       'Fighter1 KD Round5',
       'Fighter1 Total Str. Round5', 'Fighter1 Subb.Att Round5',
       'Fighter1 Rev. Round5', 'Fighter2 KD Round5',
       'Fighter2 Total Str. Round5', 'Fighter2 Subb.Att Round5',
       'Fighter2 Rev. Round5']
df5[obj_col] = df5[obj_col].astype(float)


# Here now we check that how much data is skewed. Generally we set 0.75 the skew_limit above 0.75 is heavily skew distribution and we should probably run some type of transformation

# In[59]:


num_cols = df5.select_dtypes('number').columns
skew_limit = 0.75
skew_vals = df5[num_cols].skew()


# In[60]:


skew_vals


# In[61]:


skew_cols = skew_vals[abs(skew_vals)>skew_limit].sort_values(ascending=False)
skew_cols


# In[63]:


df6 = df5.copy()


# In[66]:


all_col = list(df6.columns)
df6 = df6.reindex(columns=all_col, fill_value='0')


# ## Actions
# 
# **The next approach is to extract some valuable information from the columns. Here are**
# 
# - [x] Separate landed of attempted to separate columns
# 
# - [x] Convert last_round_time to total_time_fought by using last_round and Format
# 
# - [x] Convert CTRL to time_in_CTRL
# 
# - [x] Convert percentages to fractions
# 
# - [x] Create current_win_streak, current_lose_streak, longest_win_streak, wins, losses, draw

# The data contains single fight, we have to convert it into a format that shows the compilation data of each fighter up until that fight. Now on every row will look a lot different than it looks now.
# 
# **NOTE** `Time Format` column

# In[76]:


format_time = {'3 Rnd (5-5-5)': 5*60, '5 Rnd (5-5-5-5-5)': 5*60, '1 Rnd + OT (12-3)': 12*60,
       'No Time Limit': 1, '3 Rnd + OT (5-5-5-5)': 5*60, '1 Rnd (20)': 1*20,
       '2 Rnd (5-5)': 5*60, '1 Rnd (15)': 15*60, '1 Rnd (10)': 10*60,
       '1 Rnd (12)':12*60, '1 Rnd + OT (30-5)': 30*60, '1 Rnd (18)': 18*60, '1 Rnd + OT (15-3)': 15*60,
       '1 Rnd (30)': 30*60, '1 Rnd + OT (31-5)': 31*5,
       '1 Rnd + OT (27-3)': 27*60, '1 Rnd + OT (30-3)': 30*60}

exception_format_time = {'1 Rnd + 2OT (15-3-3)': [15*60, 3*60], '1 Rnd + 2OT (24-3-3)': [24*60, 3*60]}


# In[77]:


df6['all_round_time'] = df6['Time'].apply(lambda X: int(X.split(':')[0])*60 + int(X.split(':')[1]))


# In[79]:


def get_total_time(row):
    if row['Time Format'] in format_time.keys():
        return (row['Rounds'] - 1) * format_time[row['Time Format']] + row['all_round_time']
    elif row['Time Format'] in exception_format_time.keys():
        if (row['Rounds'] - 1) >= 2:
            return exception_format_time[row['Time Format']][0] + (row['Rounds'] - 2) *                     exception_format_time[row['Time Format']][1] + row['all_round_time']
        else:
            return (row['Rounds'] - 1) * exception_format_time[row['Time Format']][0] + row['all_round_time']


# In[80]:


df6["Total_time_fought(sec)"] = df6.apply(get_total_time, axis=1)


# In[82]:


def get_no_of_rounds(X):
    if X == 'No Time Limit':
        return 1
    else:
        return len(X.split('(')[1].replace(')', '').split('-'))


# In[83]:


df6['no_of_rounds'] = df6['Time Format'].apply(get_no_of_rounds)


# In[84]:


## Treat Ctrl of a fighter time str
dt_c = ['Fighter1 Ctrl. Total', 'Fighter2 Ctrl. Total',
       'Fighter1 Ctrl. Round1', 'Fighter2 Ctrl. Round1',
       'Fighter1 Ctrl. Round2', 'Fighter2 Ctrl. Round2',
       'Fighter1 Ctrl. Round3', 'Fighter2 Ctrl. Round3',
       'Fighter1 Ctrl. Round4', 'Fighter2 Ctrl. Round4',
       'Fighter1 Ctrl. Round5', 'Fighter2 Ctrl. Round5']

df6[dt_c] = df6[dt_c].astype(str)


# **dt_c** columns are CTRL columns are the time, which contains some ``nan`` which we have changed to `0`. 0 means no time mentioned because from now on we convert the time in seconds

# In[85]:


df6[dt_c] = df6[dt_c].apply(lambda x: x.replace('nan','0'))


# In[87]:


df6[dt_c] = df6[dt_c].apply(lambda x: x.replace('--','0'))


# In[89]:


def conv_to_sec(X):
    if X != '--' and X !='0':
        return int(X.split(':')[0])*60 + int(X.split(':')[1])
    else:
        return 0
for column in dt_c:
    df6[column+'_time(sec)'] = df6[column].apply(conv_to_sec)

df6.head(1)


# In[91]:


## Data Per fighter
fighter1 = df6['Fighter1'].value_counts().index
fighter2 = df6['Fighter2'].value_counts().index
fighters = list(set(fighter1) | set(fighter2))


# In[92]:


df6['Winner'] = df6['Winner'].astype(str)


# In[93]:


df6['Winner'].value_counts()


# Taking the `Winner` coloumn and if the fighter1 won then it fighter1 in the row and if fighter2 won then it returns fighter2won 

# In[97]:


def get_renamed_winner(row):
    if row['Fighter1'] == row['Winner']:
        return 'Fighter1Won'
    elif row['Fighter2'] == row['Winner']:
        return 'Fighter2Won'
    elif row['Winner'] == 'Draw':
        return 'Draw'


# In[98]:


df6['Win_by_fighter'] = df6[['Fighter1', 'Fighter2', 'Winner']].apply(get_renamed_winner, axis=1)


# In[99]:


df6 = pd.concat([df6,pd.get_dummies(df6['Method'], prefix='win_by')],axis=1)
df6.drop(['Method'],axis=1, inplace=True)


# In[100]:


df6.drop(r_s, axis=1, inplace=True)


# In[101]:


df7 = df6.copy()


# In[ ]:


Taking


# In[103]:


l = []
n_col = ['Fighter1 KD Total', 'Fighter1 Subb.Att Total', 'Fighter1 Rev. Total', 
         'Fighter1 Total Str. Total', 'Fighter2 Total Str. Total',
         'Fighter1 Head Total','Fighter1 Body Total','Fighter1 Leg Total','Fighter1 Distance Total','Fighter1 Clinch Total',
         'Fighter1 Ground  Total',
         'Fighter2 KD Total','Fighter2 Subb.Att Total', 'Fighter2 Rev. Total','Fighter2 Head Total',
         'Fighter2 Body Total','Fighter2 Leg Total','Fighter2 Distance Total',
         'Fighter2 Clinch Total','Fighter2 Ground  Total',
         'Fighter1 Total Str. Round1','Fighter1 Head Round1',
         'Fighter1 KD Round1', 'Fighter1 Subb.Att Round1','Fighter1 Rev. Round1',
         'Fighter2 KD Round1', 'Fighter2 Subb.Att Round1','Fighter2 Rev. Round1',
         'Fighter1 KD Round2', 'Fighter1 Subb.Att Round2','Fighter1 Rev. Round2',
         'Fighter2 KD Round2', 'Fighter2 Subb.Att Round2','Fighter2 Rev. Round2',
         'Fighter1 KD Round3', 'Fighter1 Subb.Att Round3','Fighter1 Rev. Round3',
         'Fighter2 KD Round3', 'Fighter2 Subb.Att Round3','Fighter2 Rev. Round3',
         'Fighter1 KD Round4', 'Fighter1 Subb.Att Round4','Fighter1 Rev. Round4',
         'Fighter2 KD Round4', 'Fighter2 Subb.Att Round4','Fighter2 Rev. Round4',
         'Fighter1 KD Round5', 'Fighter1 Subb.Att Round5','Fighter1 Rev. Round5',
         'Fighter2 KD Round5', 'Fighter2 Subb.Att Round5','Fighter2 Rev. Round5',

         'Fighter1 Body Round1','Fighter1 Leg Round1','Fighter1 Distance Round1','Fighter1 Clinch Round1',
         'Fighter1 Ground Round1','Fighter2 Total Str. Round1','Fighter2 Head Round1','Fighter2 Body Round1',
         'Fighter2 Leg Round1','Fighter2 Distance Round1','Fighter2 Clinch Round1','Fighter2 Ground Round1',
         'Fighter1 Total Str. Round2','Fighter1 Head Round2','Fighter1 Body Round2','Fighter1 Leg Round2',
         'Fighter1 Distance Round2','Fighter1 Clinch Round2','Fighter1 Ground Round2','Fighter2 Total Str. Round2',
         'Fighter2 Head Round2','Fighter2 Body Round2','Fighter2 Leg Round2','Fighter2 Distance Round2',
         'Fighter2 Clinch Round2','Fighter2 Ground Round2','Fighter1 Total Str. Round3','Fighter1 Head Round3',
         'Fighter1 Body Round3','Fighter1 Leg Round3','Fighter1 Distance Round3','Fighter1 Clinch Round3',
         'Fighter1 Ground Round3','Fighter2 Total Str. Round3','Fighter2 Head Round3','Fighter2 Body Round3',
         'Fighter2 Leg Round3','Fighter2 Distance Round3','Fighter2 Clinch Round3','Fighter2 Ground Round3',
         'Fighter1 Total Str. Round4','Fighter1 Head Round4','Fighter1 Body Round4','Fighter1 Leg Round4',
         'Fighter1 Distance Round4','Fighter1 Clinch Round4','Fighter1 Ground Round4',
         'Fighter2 Total Str. Round4','Fighter2 Head Round4','Fighter2 Body Round4',
         'Fighter2 Leg Round4','Fighter2 Distance Round4','Fighter2 Clinch Round4','Fighter2 Ground Round4',
         'Fighter1 Total Str. Round5','Fighter1 Head Round5','Fighter1 Body Round5','Fighter1 Leg Round5',
         'Fighter1 Distance Round5','Fighter1 Clinch Round5','Fighter1 Ground Round5','Fighter2 Total Str. Round5',
         'Fighter2 Head Round5','Fighter2 Body Round5','Fighter2 Leg Round5','Fighter2 Distance Round5',
         'Fighter2 Clinch Round5','Fighter2 Ground Round5','Fighter1 Sig.Str. Round5', 'Fighter1 TD Round5',
         'Fighter2 TD Round5',
         'Total_time_fought(sec)', 'Fighter1 Ctrl. Total_time(sec)','Fighter2 Ctrl. Total_time(sec)',
         'Fighter1 Ctrl. Round1_time(sec)', 'Fighter2 Ctrl. Round1_time(sec)',
         'Fighter1 Ctrl. Round2_time(sec)', 'Fighter2 Ctrl. Round2_time(sec)',
         'Fighter1 Ctrl. Round3_time(sec)', 'Fighter2 Ctrl. Round3_time(sec)',
         'Fighter1 Ctrl. Round4_time(sec)', 'Fighter2 Ctrl. Round4_time(sec)',
         'Fighter1 Ctrl. Round5_time(sec)', 'Fighter2 Ctrl. Round5_time(sec)']

df7[n_col] = df7[n_col].astype(float)


# In[105]:


# Taking by each Fighter
f1_ = df7.groupby('Fighter1')
f2_ = df7.groupby('Fighter2')


# In[106]:


## Replacing the names 
import re

def lreplace(pattern, sub, string):
    """
    Replaces 'pattern' in 'string' with 'sub' if 'pattern' starts 'string'.
    """
    return re.sub('^%s' % pattern, sub, string)


# In[107]:


def fighter1Won(fighter_name):
    try:
        fighter1_won = f1_.get_group(fighter_name)
    except:
        return None
    rename_columns = {}
    for column in fighter1_won.columns:
        if re.search('^Fighter1', column) is not None:
            rename_columns[column] = lreplace('Fighter1', 'hero', column)
        elif re.search('^Fighter2', column) is not None:
            rename_columns[column] = lreplace('Fighter2', 'opp', column)
    fighter1_won = fighter1_won.rename(rename_columns, axis='columns')
    return fighter1_won


# In[108]:


def fighter2Won(fighter_name):
    try:
        fighter2_won = f2_.get_group(fighter_name)
    except:
        return None
    rename_columns = {}
    for column in fighter2_won.columns:
        if re.search('^Fighter1', column) is not None:
            rename_columns[column] = lreplace('Fighter1', 'hero', column)
        elif re.search('^Fighter2', column) is not None:
            rename_columns[column] = lreplace('Fighter2', 'opp', column)
    fighter2_won = fighter2_won.rename(rename_columns, axis='columns')
    return fighter2_won


# In[109]:


def get_result_stats(result_list):
    result_list.reverse() 
    current_win_streak = 0
    current_lose_streak = 0
    longest_win_streak = 0
    wins = 0
    losses = 0
    draw = 0
    for result in result_list:
        if result == 'hero':
            wins += 1
            current_win_streak += 1
            current_lose_streak = 0
            if longest_win_streak < current_win_streak:
                longest_win_streak += 1
        elif result == 'opp':
            losses += 1
            current_win_streak = 0
            current_lose_streak += 1
        elif result == 'Draw':
            draw += 1
            current_lose_streak = 0
            current_win_streak = 0
            
    return current_win_streak, current_lose_streak, longest_win_streak, wins, losses, draw


# In[117]:


for x in n_col:
    if re.search('^Fighter1', x):
        l.append(lreplace('Fighter1', 'hero', x))
    elif re.search('^Fighter2', x):
        l.append(lreplace('Fighter2', 'opp', x))


# In[119]:


df7.rename(columns = {"win_by_TKO - Doctor's Stoppage":"win_by_TKO_Doctors_Stoppage"}, inplace = True)


# In[122]:


methods = ['win_by_Could Not Continue', 'win_by_DQ', 'win_by_Decision - Majority', 
           'win_by_Decision - Split','win_by_Decision - Unanimous', 'win_by_KO/TKO', 'win_by_Other',
           'win_by_Overturned', 'win_by_Submission','win_by_TKO_Doctors_Stoppage']


# In[124]:


f1_frame = pd.DataFrame()
f2_frame = pd.DataFrame()


result_stats = ['current_win_streak', 'current_lose_streak', 'longest_win_streak', 'wins', 'losses', 'draw']

for fighter_name in fighters:
    fighter_1 = fighter1Won(fighter_name)
    fighter_2 = fighter2Won(fighter_name)
    fighter_index = None
    
    if fighter_1 is None:
        fighter = fighter_2
        fighter_index = 'Fighter2Won'
    elif fighter_2 is None:
        fighter = fighter_1
        fighter_index = 'Fighter1Won'
    else:
        fighter = pd.concat([fighter_1, fighter_2]).sort_index()
##         print(fighter)
    fighter['Winner'] = fighter['Winner'].apply(lambda X: 'hero' if X == fighter_name else 'opp')

    for i, index in enumerate(fighter.index):
        fighter_slice = fighter[(i+1):].sort_index(ascending=False)
        s = fighter_slice[l].ewm(span=3, adjust=False).mean().tail(1)
        if len(s) != 0:
            pass
        else:
            s.loc[len(s)] = [np.NaN for _ in s.columns]
        s['Total_round_fought'] = fighter_slice['Rounds'].sum()
        s['hero_fighter'] = fighter_name
        results = get_result_stats(list(fighter_slice['Winner']))
        for result_stat, result in zip(result_stats, results):
            s[result_stat] = result
##         print(s)
        win_by_results = fighter_slice[fighter_slice['Winner'] == 'hero'][methods].sum()

        for win_by_column,win_by_result in zip(methods, win_by_results):
            s[win_by_column] = win_by_result
        s.index = [index]


        if fighter_index is None:
            if index in fighter_2.index:
                f2_frame = f2_frame.append(s)
            elif index in fighter_1.index:
                f1_frame = f1_frame.append(s)
        elif fighter_index == 'Fighter2Won':
            f2_frame = f2_frame.append(s)
        elif fighter_index == 'Fighter1Won':
            f1_frame = f1_frame.append(s)


# In[125]:


f1_frame.T


# In[126]:


f1_frame.shape


# In[127]:


list(f1_frame.columns)


# In[128]:


f2_frame.T


# In[132]:


second_f = f2_frame.add_prefix('F2_')
first_f = f1_frame.add_prefix('F1_')


# In[133]:


new_df = first_f.join(second_f, how='outer')


# In[137]:


r_col = {}
for x in new_df.columns:
    if re.search('_hero', x):
        r_col[x] = re.sub("[a-z]*_hero", "_avg", x)
    if re.search('_opp', x):
        r_col[x] = re.sub("[a-z]*_opp", "_avg_opp", x)
    if 'win_by' in x:
        r_col[x] = x.replace(' ', '').replace('-', '_')


# In[139]:


new_df.rename(r_col, axis='columns', inplace=True)


# In[145]:


df8 = df7.join(new_df, how='outer')


# In[148]:


df8.drop(f_p, axis=1, inplace=True)


# In[149]:


for col in df8.columns[5:81]:
    if col == 'Win_by_fighter':
        continue
    else:
        df8.drop(col, axis=1, inplace=True)


# In[150]:


df8['Winner'] = df8['Win_by_fighter']


# In[151]:


df8.drop('Win_by_fighter', axis=1, inplace=True)


# In[152]:


df8.rename({'Rounds':'all_rounds'}, axis=1, inplace=True)


# In[155]:


for col in df8.columns:
    if df8[col].isnull().sum() != 0:
        print(f"{col}: {df8[col].isnull().sum()}")


# In[156]:


stats_df = df8.describe()
stats_df.loc['range'] = stats_df.loc['max'] - stats_df.loc['min']
out_fields = ['mean', '25%','50%','75%','range']
stats_df = stats_df.loc[out_fields]
stats_df.rename({'50%':'median'}, inplace=True)
stats_df


# In[157]:


df8.fillna(df8.median(), inplace=True)


# In[159]:


df8.drop(['Date', 'Fighter1', 'Fighter2'], axis=1, inplace=True)


# In[160]:


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
scaler = StandardScaler()

n_types = df8.select_dtypes(include=[float, int])
df8[list(n_types.columns)] = scaler.fit_transform(df8[list(n_types.columns)])


# In[161]:


X = df8.drop(list(df8.select_dtypes('O').columns), axis=1)
y = df8['Winner']


# In[162]:


X.head()


# In[163]:


X.shape


# In[164]:


y.head()


# In[165]:


y.shape


# # Spliting the data into train and test

# In[166]:


# split data in to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[167]:


print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# # 1. Logistic Regression Model

# In[168]:


#training  logistic regression model]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[169]:


# train model
log=LogisticRegression(solver='lbfgs', max_iter=1000)
log.fit(X_train, y_train)


# In[170]:


# logistic regression model accuracy
y_pred = log.predict(X_test)
logAccuracy=accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {logAccuracy}")


# In[172]:


# confusion_matrix for logistic regression model
import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')


# In[173]:


# confusion metrix report for logistic regression model
print(classification_report(y_test, y_pred))


# # 2. Random Forest Model

# In[178]:


# Random forest model 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000)


# In[179]:


# train randorm forest model
rf.fit(X_train, y_train)


# In[180]:


# random forest model accuracy
y_pred = rf.predict(X_test)
rfAccuracy=accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {rfAccuracy}")


# In[181]:


# confusion_matrix for random forest model
import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')


# In[182]:


# confusion matrix report for random forest model
print(classification_report(y_test, y_pred))


# # 3. Neural Network Model

# In[183]:


# Neural Network Model
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


# In[184]:


# train NN model
mlp.fit(X_train, y_train)


# In[185]:


# accuracy for NN model
y_pred = mlp.predict(X_test)
mlpAccuracy=accuracy_score(y_test, y_pred)
print(f"Neural Network Accuracy: {mlpAccuracy}")


# In[187]:


# confusion_matrix for NN model
import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')


# In[188]:



# confusion matrix report for NN model
print(classification_report(y_test, y_pred))


# # comparing the above 3 models

# In[197]:


# comparing the accuracy of the models
comparison = pd.DataFrame({'Model':['Logistic Regression', 'Random Forest', 'Neural Network'], 'Accuracy':[logAccuracy, rfAccuracy, mlpAccuracy]})
ax=sns.barplot(x='Model', y='Accuracy', data=comparison,palette = "Blues")
ax.set(xlabel='Model', ylabel='Accuracy')
ax.title.set_text('Model Accuracy')
ax.bar_label(ax.containers[0])


# In[205]:


# Confidence Intervals for all the models
plt.figure(figsize=(10,5))
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.swarmplot(x='Model', y='Accuracy', data=comparison, palette="Blues", size=10, edgecolor="black", linewidth=1)


# In[ ]:





# In[ ]:




