#!/usr/bin/env python
# coding: utf-8

# # ***NO SHOW ANALYSIS STUDY*** 
# ___
# 
# 
# 
# ## Table of Contents
# ### 1. Introduction
# 1. [what is no show](#what-is-no-show)
# 
# ### 2.  ***Steps for analysis process***
# 1. [Steps for analysis process](#Steps-for-analysis-process)
# 
# ### 3. ***questions***
# 1. [1.1- questions](#1.1--questions)
# 
# 
# ### 4. ***Data Collection and Wrangling***
# 1. [a-gathering data](#a-gathering-data)
# * [b-assessing data](#b-assessing-data)
# * [c-cleaning data](#c-cleaning-data)
# * [Observations](#Observations)
# 
# ### 5. ***DATA ANALYSIS and  exploration ***
# 
# * [EDA(EXPLORATORY DATA ANALYSIS )](#EDA(EXPLORATORY-DATA-ANALYSIS-))
# 
# * [finding pattern](#finding-pattern)
# 
# * [1- age study:](#1--age-study:)
# 
#    * [visualize relationship](#visualize-relationship)
#    
#    * [build intution](#build-intution)
#    
#    * [creat more and describtive feture](#creat-more-and-describtive-feture)
#    
#    * [conclusion :](#conclusion-:)
#    
#  
# * [2-waiting days study:](#2-waiting-days-study:)
# 
#   * [1.visualize relationship](#1.visualize-relationship)
# 
#   * [2.build intution](#2.build-intution)
#   
#   * [3.creat more and describtive feture](#3.creat-more-and-describtive-feture)
#   
#   * [4.conclusion :](#4.conclusion-:)
#   
# 
# * [3- gender study:](#3--gender-study:)
# 
#    * [3.1visualize relationship](#3.1visualize-relationship)
#    
#    * [3.2build intution](#3.2build-intution)
#    
#    * [3.3creat more and describtive feture](#3.3creat-more-and-describtive-feture)
#    
#    * [3.4conclusion :](#3.4conclusion-:)  
#   
# ---
# ### 1. Introduction
# 
# # what is no show
# 
# ###### This dataset collects information from 100k medical appointments inBrazil and is focused on the quetion <br>
# ###### of whether or not patients show upfor their appointment.<br>
# ###### A number ofcharacteristics about the patient are included in each row.<br>
# ###### ● ‘ScheduledDay’ tells us on what day the patient set up their appointment.<br>
# ###### ● ‘Neighborhood’ indicates the location of the hospital.<br>
# ###### ● ‘Scholarship’ indicates whether or not the patient is enrolled in Brasilian welfare program Bolsa Família.<br>
# ###### ● Be careful about the encoding of the last column: it says ‘No’ ifthe patient showed up to their appointment,
# ###### and ‘Yes’ if theydid not show up.
# ---
# 

# ___
# # ***Steps for analysis process***
# ### 2.  ***Steps for analysis process***
# ### 1- questions
# ### 2- wrangle
# * a- gather 
# * b- assess 
# * c- clean
# 
# ### 3- explor
# 
# * EDA(exploratotry data analysis ) 
# * AUGMENTIG data to maximize the potential of analysis ,visualization and modeles
# * finding pattern
# * visualize relationship
# * build intution
# * remove outliers
# * creat more and decribtive feture
# 
# ### 4- draw conclousion
# ### 5- comunicate
# 
# ___

# # ***1.1- questions***
# <div class="alert alert-block alert-info">
# <b>
# This study focuses on three factors, age ,sex and gender and tries to find a relationship through which to predict the extent of the commitment of patients to their show up at appointment dates of visits specified to them and tries to answer questions like<br>
# 1- What factors may help to predict if a patient will show up for their scheduled appointment?<br>
# 2- Why do 30% of patients miss their scheduled appointments?<br>
# 3- is that possible to predict someone to no-show an appointment?<br>
# 4- Is there any relation between patients age and their commitment to appointment attendens?<br>    
# 5- Is there any  relationship of the waiting days between the schedule_day and appointment_day affect on the commitment to attends?<br>
# 6- Is there any relation between patients gender and their commitment to appointment attendens?<br>    
# 7- does gender in different age stage affects their commitment to show up on the appointement date?
#  
# </div>

# ### 2. ***Data Collection and Wrangling***
# 

# 
# #### ***a-gathering data***
# 
# we already have data so we need to have general look to see what data we have to see what type of question we can aske in this analysis 

# In[1]:


# we will import all liberies we need in our analysis
import pandas as pd
import numpy as np
import datetime
from time import strftime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

df = pd.read_csv('C:\\Users\\zas\\Downloads\\class3\\no show\\noshowappointments-kagglev2-may-2016.csv',sep=',')
df.head()


# ### ***b-assessing data***

# In[2]:


df.head()


# In[3]:


# same thing applies to `.tail()` which returns the last few rows
df.tail()


# In[4]:


# this returns a tuple of the dimensions of the dataframe
print('total number of rows in data => {}'.format(df.shape[0]))
print('total number of columns in data => {}'.format(df.shape[1]))


# In[5]:


# general information about data
df.info()


# ### ***c-cleaning data***

# ### ***check data for :***<br>***1-missing data*** <br>  ***2-duplicate data***  <br>   ***3-incorrect data types***
# 

# In[6]:


# check missing value as see in above no missing values but to confirm:

print(df.isnull().sum())


# ### ======>>>>>   :     No null values

# In[7]:


#check duplicated data
df.duplicated().sum()


# ### ======>>>>>   :     No  duplicated data

# 
# 
# <div class="alert alert-block alert-success">
# <b>what type of data we have?</b> 
# </div>
# 

# ### object data type need further investigation to shows - what is it?

# In[8]:


type(df['Gender'][0])


# In[9]:


type(df['AppointmentDay'][0])


# In[10]:


type(df['Neighbourhood'][0])


# In[11]:


type(df['ScheduledDay'][0])


# In[12]:


type(df['No-show'][0])


# ### Observations

# ### ===>>> : 
# <div class="alert alert-block alert-success">
# <b>some fieldes of incorect data type and  need to change</b> 
# </div>
# 

# In[13]:


# we have date time object type  so we need to change it to date time
df['PatientId'] = df['PatientId'].astype('int64')
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.date.astype('datetime64[ns]')
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.date.astype('datetime64[ns]')


# In[14]:


# Get Day of the Week for ScheduledDay and AppointmentDay
df['schedule_day_week'] = df['ScheduledDay'].dt.weekday_name
df['appointment_day_week'] = df['AppointmentDay'].dt.weekday_name


# In[15]:


# Get the Waiting Time in Days of the Patients.
df['waiting_days'] = df['AppointmentDay'] - df['ScheduledDay']
df['waiting_days'] = df['waiting_days'].dt.days


# # Sort by date

# In[16]:


df.sort_values(["ScheduledDay","AppointmentDay"], inplace=True, ascending=True) 


# # rename columns 

# In[17]:


df.rename(columns={'PatientId':'patient_id','AppointmentID': 'appointment_id' , 'Neighbourhood':'neighborhood','Gender':'gender','Scholarship':'scholarship',
                   'Hipertension':'hypertension', 'Handcap':'handicap','Diabetes':'diabetes','Alcoholism':'alcoholism',
                   'SMS_received':'sms_received','ScheduledDay':'schedule_day', 'AppointmentDay':'appointment_day','Age':'age',
                   'No-show':'no_show'}, inplace=True)


# In[18]:


#check data 
df.info()


# In[19]:


# check data 
df.head(3)


# In[20]:


# this returns the number of unique values in each column FOR ALL
df.nunique()


# ### what are the unique values for:
# 
# 
# <div class="alert alert-block alert-info">
# <b>gender,age,scholarship,hypertension,diabetes,alcoholism,handicap,smreceived and no_show ?</b>
# </div>
# 

# In[21]:


print("Unique Values in `Gender` => {}".format(df.gender.unique()))
print("Unique Values in `Scholarship` => {}".format(df.scholarship.unique()))
print("Unique Values in `Hypertension` => {}".format(df.hypertension.unique()))
print("Unique Values in `Diabetes` => {}".format(df.diabetes.unique()))
print("Unique Values in `Alcoholism` => {}".format(df.alcoholism.unique()))
print("Unique Values in `Handicap` => {}".format(df.handicap.unique()))
print("Unique Values in `Sms_received` => {}".format(df.sms_received.unique()))
print("Unique Values in `No_show` => {}".format(df.no_show.unique()))


# In[22]:


# unique data for patient_id 
print('Number of unque values of patient_id =>  : {}'.format(df.patient_id.unique().shape[0]))
print('percent of patient who registered for appointment more than one time =>  : {} % of total patient'.format((1-(df.patient_id.unique().shape[0])/(df.patient_id.shape[0]))*100))


# In[23]:


# Print Unique Values for 'schedule_day'
print("Unique Values in `schedule_day` => {}".format(np.sort(df.schedule_day.dt.strftime('%Y-%m-%d').unique())))


# 
# <div class="alert alert-block alert-success">
# <b> We can see from the above details that the schedule_day for appointments are: <br>starting from 2015-11-10 upto 2016-06-08 <br> that's around 7 months .</b> 
# </div>

# In[24]:


# Print Unique Values for 'AppointmentDay'
print("Unique Values in `appointment_day` => {}".format(np.sort(df.appointment_day.dt.strftime('%Y-%m-%d').unique())))


# ###  starting from 2016-04-29 upto 2016-06-08. that's around 1 month

# In[25]:


# Print Unique Values for 'waiting_days'
print("Unique Values in `waiting_days` => {}".format(np.sort(df.waiting_days.unique())))


# In[26]:


#cleaning data for waiting days
df[df['waiting_days']==-6]


# In[27]:


print(df[df['waiting_days']==-6].shape[0])


# In[28]:


df[df['waiting_days']==-1]


# In[29]:


df[df['waiting_days']==-1].shape


# # ===>>>( -6 waitng_days) this may be by mistake <br>===>>> (-1 waiting_days) this may be by mistake  

# In[30]:


# to drop this wrong data by using index
df.drop([71533,72362,64175,27033,55226],inplace= True)


# In[31]:


# Print Unique Values for 'waiting_days to check cleaning process'
print("Unique Values in `waiting_days` => {}".format(np.sort(df.waiting_days.unique())))


# In[32]:


# unique values in age and clean data
df['age'][0]
type(df['age'][0])


# In[33]:


print("Unique Values in `age` => {}".format(np.sort(df.age.unique())))


# In[34]:


df.query('age == -1')


# **Note there is one row that contain age -1 in min . So lets drop that row.**<br>
# * Note (-1) may be mistak in recording or she is pregnant wamen and the appointment for embryo investigation *

# In[35]:


# to drop this wrong data by using index
df.drop([99832],inplace= True)


# In[36]:


# to check cleaning process
print("Unique Values in `age` => {}".format(np.sort(df.age.unique())))


# In[37]:


# Print Unique Values for 'neighborhood'
print("Unique Values in `neighborhood` => {}".format(np.sort(df.neighborhood.unique())))


# In[38]:


# check all the data with general look
# this returns useful descriptive statistics for each column of data
df.describe()


# In[39]:


df.info()


# In[40]:


df.sample(3)


# In[41]:


# checking data for unique  value for patient_id   
df.shape


# In[42]:


no_show_mask=df[df['no_show']== 'No']
no_show_mask.groupby(['no_show']).patient_id.value_counts()


# In[43]:


no_show_mask.groupby(['no_show']).patient_id.value_counts().describe()


# # ohh !!!
# <div class="alert alert-block alert-success">
# <b>from above describtive data there are some observartion</b> 
# </div>
#  
# #### 1. range of patients who show up start from : 1 show up time up to : 87 show up time 
# #### 2. we need to investigate patients who were show up <= 2 times as 75% of patient showup 2 times or less
# #### 3. we need to investigate patient who were show up > 87 times  as 25% of patient showup 2 times or more
# 
# 

# 
# <div class="alert alert-block alert-info">
# <b>so  who is this patient that book 87 appointment?</b>
# </div>
# 

# In[44]:


no_show_mask.query('patient_id== 822145925426128').nunique()


# In[45]:


no_show_mask.query('patient_id== 822145925426128').schedule_day.value_counts()


# In[46]:


print("frequencey of Unique Values in `schedule_day` => {}".format(np.sort(no_show_mask.query('patient_id== 822145925426128').schedule_day.value_counts())))


# In[47]:


print("values in `schedule_day` => {}".format(np.sort(no_show_mask.query('patient_id== 822145925426128').schedule_day)))


# In[48]:


# describtive data for patient_id== 822145925426128
no_show_mask.query('patient_id== 822145925426128').describe()


# In[49]:


print("frequencey of Unique Values in `sms_received` => {}".format(np.sort(no_show_mask.query('patient_id== 822145925426128').sms_received.value_counts())))


# In[50]:


no_show_mask.query('patient_id== 822145925426128').sms_received.value_counts()


# In[51]:


print("frequencey of Unique Values in `waiting_days` => {}".format(np.sort(no_show_mask.query('patient_id== 822145925426128').waiting_days.value_counts())))


# In[52]:


no_show_mask.query('patient_id== 822145925426128').waiting_days.value_counts()


# 
# <div class="alert alert-block alert-success">
# <b> data shows that:</b> 
# </div>
# 
# 1. this patient registered from  1:7 times per day <br>
# * this patient have been shown up almost daily and go back to his home during saturday and sunday<br>
# * this patient is male and 38 years old 
# * this patient have No scholarship ,No hypertension,NO diabetes, No alcoholism, No handicap
# * if this patient has no chronic disease and not in_patient case this hospital data need to be reviewed 
# * if this patient has chronic disease or he was in_patient case it is very difficult to expect what are main factors that affect show up because data mixed by about 25% but we will try  
# 

# In[53]:


# here we are going to drop about 25% of paient to clean data and will investigate 75% that appear here 
no_show_mask.groupby(['no_show']).patient_id.value_counts().describe()


# In[54]:


fr=df.groupby('no_show').patient_id.value_counts();
ffr=fr.sort_values(ascending=False)
f=ffr.to_frame(name='id_value_counts')
merged = pd.merge(df, f, on='patient_id',how='inner')
df=merged
df.head()


# In[55]:


df.shape


# In[56]:


df.drop_duplicates(['appointment_id'],inplace =True)


# In[57]:


df.shape


# In[58]:


df.info()


# In[59]:


#test data
df.query('patient_id==832256398961987')


# 
# <div class="alert alert-block alert-success">
# <b> check cleaned data</b> 
# </div>

# In[60]:


df.nunique()


# In[61]:


df=df[df['id_value_counts'] <=2]


# In[62]:


df.info()


# In[63]:


df.id_value_counts.value_counts()


# In[64]:


df.shape


# In[65]:


df.nunique()


# In[66]:


df.query('patient_id==832256398961987')


# In[67]:


df.patient_id.value_counts()


# In[68]:


df.query('patient_id==441433296929249')


# # ***Now data ready to be analysied***
# 
# <div class="alert alert-block alert-success">
# <b> Now data ready to be analysied
# </b> 
# </div>

# # ***DATA ANALYSIS and  exploration***
# 
# ### EDA(EXPLORATORY DATA ANALYSIS )

# ## By drawing histogram and scatter matrix  we are looking for patterns for different types of data columns and now we will go in more deep investigation 

# ### using scatter_matrix() and histogaram  to have quecik look for all data and trying to find pattern 

# In[69]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
pd.plotting.scatter_matrix(df,figsize=(15,15));


# # ***finding pattern***

# In[70]:


df.hist(figsize=(15,15));


# # ***1- age study:***

# 
# <div class="alert alert-block alert-success">
# <b>  Is there any relation between patient age and their commitment to appointment attendens?
# </b> 
# </div>

# In[71]:


df.age.count()


# In[72]:


df.age.hist();


#  
# <div class="alert alert-block alert-info">
# <b>
#  separation and visualization of patients who show up according to their ages <br>
# 
# </div>

# In[73]:


no_show_mask=df[df['no_show']== 'No']


# In[74]:


print('Total number of patient who record Appointment Booking {} patient'.format(df.no_show.shape[0]))
print('Number of patient who show up from id :  {}  patient'.format(no_show_mask.shape[0]))
print('percent of patient who show up {}'.format(no_show_mask.shape[0]/(df.no_show.shape[0])*100))
print('percent of patient who were Not show up {} patient'.format((1-no_show_mask.shape[0]/(df.no_show.shape[0]))*100))


# ### ***visualize relationship***

# In[75]:


from  matplotlib import pyplot
import seaborn as sns
sns.set(style='ticks')
get_ipython().run_line_magic('matplotlib', 'inline')

mal_noshow_zeroage=df[(df.age) & (df.no_show == 'No')].shape[0]
mal_yesshow_zeroage=df[(df.age) & (df.no_show == 'Yes')].shape[0]
total=mal_noshow_zeroage+mal_yesshow_zeroage
proportion_no=mal_noshow_zeroage/total
proportion_yes=mal_yesshow_zeroage/total


# In[76]:


colors=['blue','orange']
plt.bar(["No", "yes"], [proportion_no, proportion_yes],color=colors)
plt.title("no_show proportion ")
plt.xlabel("no_show")
plt.ylabel("proportion of no_show ");


# ###### count number of patients who book appointments according to their ages

# In[77]:


df.age.value_counts().plot(kind='bar',figsize=(20,5))
plt.title('comparison for  count of patients in different age')
plt.xlabel('age')
plt.ylabel('count of patients');


#   
# <div class="alert alert-block alert-info">
# <b>
#   find out trend lines of patient who show up and not show up according to their ages <br>
# 
# </div>

# #### Bulding up new column to follow no_show counts in every patient's age  

# In[78]:


counts_no_yes=df.groupby('age').no_show.value_counts()
counts_sorting=counts_no_yes.sort_values(ascending=True)
counts_sorting_to_fram=counts_sorting.to_frame(name='no_show_count')
merged = pd.merge(df, counts_sorting_to_fram, on='age',how='inner')


# In[79]:


merged.shape


# In[80]:


merged.drop_duplicates(['appointment_id'],inplace =True)


# In[81]:


merged.shape


# In[82]:


df=merged


# ## ***Trend lines of no_show according to patient's age factor***

# In[83]:


df_no=df[df['no_show']=='No'] 
plt.figure(figsize = (20,4))
x = df_no['age']
y = df_no['no_show_count']
plt.title('relation beteen age and count number of no_show')
plt.xlabel('age')
plt.ylabel('count of No in No_show')
plt.scatter(x, y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")
plt.show()



df_yes=df[df['no_show']=='Yes'] 
plt.figure(figsize = (20,4))
x = df_yes['age']
y = df_yes['no_show_count']
plt.title('relation beteen age and count number of no_show')
plt.xlabel('age')
plt.ylabel('count of Yes in No_show')
plt.scatter(x, y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")
plt.show()


#   
# <div class="alert alert-block alert-info">
# <b>
#  from trend line we see: <br>
#     
# </div>
# 1. in general count number of patients who book appointment is inversely proportional to their age <br>
# * there are some some periodes where the trend line is directrly proprtional to their age 

# # ***build intution***

# ### we want to slice this big range according to different stage 
# 

# to study the relation between age and no_show first we need to clasify age in to groups

# 1. Fetus (Unborn)<br>
# * Newborn (Birth - 1 month)<br>
# * Baby (1 month and 1 day - 2 years)<br>
# * Toddler (3 - 5)<br> 
# * Kids (6 - 9)<br>
# * Pre-Teen (10 - 12)<br>
# * Teenager (13 - 17)<br>
# * Young Adult (18 - 20)<br>
# * Adult (21 - 39)<br>
# * Young Middle-Aged Adult (40 - 49)<br>
# * Middle-Aged Adult (50 - 54)<br>
# * Very Young Senior Citizen (55 - 64)<br>
# * Young Senior Citizen (65 - 74)<br>
# * Senior Citizen (75 - 84)<br> 
# * Old Senior Citizen (85+)
# 

# 
#  
# <div class="alert alert-block alert-info">
# <b>
#  Adding new coulmn to clasify patients into age stages <br>
#     
# </div>

# In[84]:



# Bin edges that will be used to "cut" the data into groups
# we use -1 the start of sries to include babys with 0 age
bin_edges = [-1,3,6,10 ,13,18,21,40,50,55,65,75,85,116] # Fill in this list with five values you just found

# Labels for different age_stage groups
bin_names = ['Baby', 'Toddler','Kids', 'Pre-Teen', 'Teenager','Young_Adult','Adult','Young_Middle_Aged_Adult','Middle_Aged_Adult','Very_Young_Senior_Citizen','Young_Senior_Citizen', 'Senior_Citizen','Old_Senior_Citizen' ] # Name each age stage
# Creates age_stage column
df['age_stage'] = pd.cut(df['age'], bin_edges, labels=bin_names)

#df['age_stage']=df.loc['bin_edges','bin_names'] 
# Checks for successful creation of this column
df.head(3)


# In[85]:


# to confirm there is no null
df[df.age_stage.isnull()]


# In[86]:


df.isnull().sum()


# In[87]:


df.info()


# # ***creat more and describtive feture***

# 
#  
# <div class="alert alert-block alert-info">
# <b>
#  comparison for different age stages versuse no_show <br>
#     
# </div>

# In[88]:


df.groupby('no_show').age_stage.count()


# In[89]:


df.groupby('no_show').age_stage.count().plot(kind ='bar',figsize=(7,4))
plt.title('comparison for total count of patients')
plt.xlabel('age')
plt.ylabel('count of No_show');


# In[90]:


print("proportion of `No_show` for age_stage  => {}".format(df.groupby('no_show').age_stage.count()/df.age_stage.count()))


# In[91]:


(df.groupby('no_show').age_stage.count()/df.age_stage.count()).plot(kind ='bar',figsize=(7,4))
plt.title('proportion of `No_show` for age_stage')
plt.xlabel('age')
plt.ylabel('count of No_show');


# # ***sorting  data sets for different age stages*** 
# 

# In[92]:


df.age_stage.value_counts()


# In[93]:


df['age_stage'].value_counts().plot(kind='bar',figsize=(15,4));


# ## ***comparison of counts of patients for no_show in different age_stage***

# In[94]:


fr=df.groupby('age_stage').no_show.value_counts();
fr.sort_values(ascending=False).plot(kind='bar',figsize=(15,4))
plt.title('comparison for counts of patients in different age_stage')
plt.xlabel('age')
plt.ylabel('count of No_show');


# ## ***proportional comparison of counts of patients for no_show in different age_stage***

# In[95]:


tr=((df.groupby('age_stage').no_show.value_counts())/df.age_stage.count())*100
tr.sort_values(ascending=False).plot(kind='bar',figsize=(15,4))
plt.title('proportion of comparison for counts of patients in different age_stage')
plt.xlabel('age')
plt.ylabel('count of No_show');


# 
# <div class="alert alert-block alert-info">
# <b>
# neumerical proportional comparison for count of patients versus no_show in different age_stage  <br>
#     
# </div>
# 

# In[96]:


tr=((df.groupby('age_stage').no_show.value_counts())/df.age_stage.count())*100
tr.sort_values(ascending=False)


# # ***Adult age stage stage :***

# In[97]:


mal_noshow_zeroage=df[(df.age_stage == 'Adult') & (df.no_show == 'No')].shape[0]
mal_yesshow_zeroage=df[(df.age_stage == 'Adult') & (df.no_show == 'Yes')].shape[0]
total=mal_noshow_zeroage+mal_yesshow_zeroage
proportion_no=mal_noshow_zeroage/total
proportion_yes=mal_yesshow_zeroage/total
print('proportion of no_show in Adult stage   "No":{}'.format(proportion_no))
print('proportion of no_show in Adult  "Yes":{}'.format(proportion_yes))


# In[98]:


colors=['blue','orange']
plt.bar(["No", "yes"], [proportion_no, proportion_yes],color=colors)
plt.title("no_show proportion for Adult ")
plt.xlabel("no_show")
plt.ylabel("proportion of no_show ");


# # ***Very_Young_Senior_Citizen:***

# In[99]:


v_noshow=df[(df.age_stage == 'Very_Young_Senior_Citizen') & (df.no_show == 'No')].shape[0]
v_yesshow=df[(df.age_stage == 'Very_Young_Senior_Citizen') & (df.no_show == 'Yes')].shape[0]
total=v_noshow+v_yesshow
proportion_no=v_noshow/total
proportion_yes=v_yesshow/total
print('proportion of no_show in Adult stage   "No":{}'.format(proportion_no))
print('proportion of no_show in Adult  "Yes":{}'.format(proportion_yes))


# In[100]:


colors=['blue','orange']
plt.bar(["No", "yes"], [proportion_no, proportion_yes],color=colors)
plt.title("no_show proportion for Very_Young_Senior_Citizen ")
plt.xlabel("no_show")
plt.ylabel("proportion of no_show ");


# # ***Young_Middle_Aged_Adult  age stage :***

# In[101]:


mal_noshow_zeroage=df[(df.age_stage == 'Young_Middle_Aged_Adult') & (df.no_show == 'No')].shape[0]
mal_yesshow_zeroage=df[(df.age_stage == 'Young_Middle_Aged_Adult') & (df.no_show == 'Yes')].shape[0]
total=mal_noshow_zeroage+mal_yesshow_zeroage
proportion_no=mal_noshow_zeroage/total
proportion_yes=mal_yesshow_zeroage/total
print('proportion of no_show in Adult stage   "No":{}'.format(proportion_no))
print('proportion of no_show in Adult  "Yes":{}'.format(proportion_yes))


# In[102]:


colors=['blue','orange']
plt.bar(["No", "yes"], [proportion_no, proportion_yes],color = colors)
plt.title("no_show proportion for Young_Middle_Aged_Adult ")
plt.xlabel("no_show")
plt.ylabel("proportion of no_show ");


# # ***Baby age stage :***

# In[103]:


mal_noshow_zeroage=df[(df.age_stage == 'Baby') & (df.no_show == 'No')].shape[0]
mal_yesshow_zeroage=df[(df.age_stage == 'Baby') & (df.no_show == 'Yes')].shape[0]
total=mal_noshow_zeroage+mal_yesshow_zeroage
proportion_no=mal_noshow_zeroage/total
proportion_yes=mal_yesshow_zeroage/total
print('proportion of no_show in Baby stage   "No":{}'.format(proportion_no))
print('proportion of no_show in Baby stage   "Yes":{}'.format(proportion_yes))


# In[104]:


colors=['blue','orange']
plt.bar(["No", "yes"], [proportion_no, proportion_yes],color=colors)
plt.title("no_show proportion for babies ")
plt.xlabel("no_show")
plt.ylabel("proportion of no_show ");


# # ***Young_Senior_Citizen :***

# In[105]:


mal_noshow_zeroage=df[(df.age_stage == 'Young_Senior_Citizen') & (df.no_show == 'No')].shape[0]
mal_yesshow_zeroage=df[(df.age_stage == 'Young_Senior_Citizen') & (df.no_show == 'Yes')].shape[0]
total=mal_noshow_zeroage+mal_yesshow_zeroage
proportion_no=mal_noshow_zeroage/total
proportion_yes=mal_yesshow_zeroage/total
print('proportion of no_show in baby stage   "No":{}'.format(proportion_no))
print('proportion of no_show in baby stage   "Yes":{}'.format(proportion_yes))


# In[106]:


colors=['blue','orange']
plt.bar(["No", "yes"], [proportion_no, proportion_yes],color=colors)
plt.title("no_show proportion for Young_Senior_Citizen ")
plt.xlabel("no_show")
plt.ylabel("proportion of no_show ");


# # ***Middle_Aged_Adult :***

# In[107]:


mal_noshow_zeroage=df[(df.age_stage == 'Middle_Aged_Adult') & (df.no_show == 'No')].shape[0]
mal_yesshow_zeroage=df[(df.age_stage == 'Middle_Aged_Adult') & (df.no_show == 'Yes')].shape[0]
total=mal_noshow_zeroage+mal_yesshow_zeroage
proportion_no=mal_noshow_zeroage/total
proportion_yes=mal_yesshow_zeroage/total
print('proportion of no_show in Middle_Aged_Adult stage   "No":{}'.format(proportion_no))
print('proportion of no_show in Middle_Aged_Adult stage   "Yes":{}'.format(proportion_yes))


# In[108]:


colors=['blue','orange']
plt.bar(["No", "yes"], [proportion_no, proportion_yes],color=colors)
plt.title("no_show proportion for Middle_Aged_Adult")
plt.xlabel("no_show")
plt.ylabel("proportion of no_show ");


# # ***Teenager :***

# In[109]:


mal_noshow_zeroage=df[(df.age_stage == 'Teenager') & (df.no_show == 'No')].shape[0]
mal_yesshow_zeroage=df[(df.age_stage == 'Teenager') & (df.no_show == 'Yes')].shape[0]
total=mal_noshow_zeroage+mal_yesshow_zeroage
proportion_no=mal_noshow_zeroage/total
proportion_yes=mal_yesshow_zeroage/total
print('proportion of no_show in Teenager stage   "No":{}'.format(proportion_no))
print('proportion of no_show in Teenager stage   "Yes":{}'.format(proportion_yes))


# In[110]:


colors=['blue','orange']
plt.bar(["No", "yes"], [proportion_no, proportion_yes],color=colors)
plt.title("no_show proportion for Teenager ")
plt.xlabel("no_show")
plt.ylabel("proportion of no_show ");


# # ***Kids :***

# In[111]:


mal_noshow_zeroage=df[(df.age_stage == 'Kids') & (df.no_show == 'No')].shape[0]
mal_yesshow_zeroage=df[(df.age_stage == 'Kids') & (df.no_show == 'Yes')].shape[0]
total=mal_noshow_zeroage+mal_yesshow_zeroage
proportion_no=mal_noshow_zeroage/total
proportion_yes=mal_yesshow_zeroage/total
print('proportion of no_show in Kid  stage   "No":{}'.format(proportion_no))
print('proportion of no_show in Kid  stage   "Yes":{}'.format(proportion_yes))


# In[112]:


colors=['blue','orange']
plt.bar(["No", "yes"], [proportion_no, proportion_yes],color=colors)
plt.title("no_show proportion for Kids")
plt.xlabel("no_show")
plt.ylabel("proportion of no_show ");


# # ***Young_Adult  :***

# In[113]:


mal_noshow_zeroage=df[(df.age_stage == 'Young_Adult') & (df.no_show == 'No')].shape[0]
mal_yesshow_zeroage=df[(df.age_stage == 'Young_Adult') & (df.no_show == 'Yes')].shape[0]
total=mal_noshow_zeroage+mal_yesshow_zeroage
proportion_no=mal_noshow_zeroage/total
proportion_yes=mal_yesshow_zeroage/total
print('proportion of no_show in Young_Adult   stage   "No":{}'.format(proportion_no))
print('proportion of no_show in Young_Adult  stage   "Yes":{}'.format(proportion_yes))


# In[114]:


colors=['blue','orange']
plt.bar(["No", "yes"], [proportion_no, proportion_yes],color=colors)
plt.title("no_show proportion for Young_Adult   ")
plt.xlabel("no_show")
plt.ylabel("proportion of no_show ");


# # ***Senior_Citizen :***

# In[115]:


mal_noshow_zeroage=df[(df.age_stage == 'Senior_Citizen') & (df.no_show == 'No')].shape[0]
mal_yesshow_zeroage=df[(df.age_stage == 'Senior_Citizen') & (df.no_show == 'Yes')].shape[0]
total=mal_noshow_zeroage+mal_yesshow_zeroage
proportion_no=mal_noshow_zeroage/total
proportion_yes=mal_yesshow_zeroage/total
print('proportion of no_show in Senior_Citizen  stage   "No":{}'.format(proportion_no))
print('proportion of no_show in Senior_Citizen  stage   "Yes":{}'.format(proportion_yes))


# In[116]:


colors=['blue','orange']
plt.bar(["No", "yes"], [proportion_no, proportion_yes],color=colors)
plt.title("no_show proportion for Senior_Citizen  ")
plt.xlabel("no_show")
plt.ylabel("proportion of no_show ");


# # ***Pre-Teen:***

# In[117]:


mal_noshow_zeroage=df[(df.age_stage == 'Pre-Teen') & (df.no_show == 'No')].shape[0]
mal_yesshow_zeroage=df[(df.age_stage == 'Pre-Teen') & (df.no_show == 'Yes')].shape[0]
total=mal_noshow_zeroage+mal_yesshow_zeroage
proportion_no=mal_noshow_zeroage/total
proportion_yes=mal_yesshow_zeroage/total
print('proportion of no_show in Pre-Teen   "No":{}'.format(proportion_no))
print('proportion of no_show in Pre-Teen   "Yes":{}'.format(proportion_yes))


# In[118]:


colors=['blue','orange']
plt.bar(["No", "yes"], [proportion_no, proportion_yes],color=colors)
plt.title("no_show proportion for Pre-Teen ")
plt.xlabel("no_show")
plt.ylabel("proportion of no_show ");


# # ***Old_Senior_Citizen :***

# In[119]:


mal_noshow_zeroage=df[(df.age_stage == 'Old_Senior_Citizen') & (df.no_show == 'No')].shape[0]
mal_yesshow_zeroage=df[(df.age_stage == 'Old_Senior_Citizen') & (df.no_show == 'Yes')].shape[0]
total=mal_noshow_zeroage+mal_yesshow_zeroage
proportion_no=mal_noshow_zeroage/total
proportion_yes=mal_yesshow_zeroage/total
print('proportion of no_show in Old_Senior_Citizen  stage   "No":{}'.format(proportion_no))
print('proportion of no_show in Old_Senior_Citizen  stage   "Yes":{}'.format(proportion_yes))


# In[120]:


colors=['blue','orange']
plt.bar(["No", "yes"], [proportion_no, proportion_yes],color=colors)
plt.title("no_show proportion for Old_Senior_Citizen  ")
plt.xlabel("no_show")
plt.ylabel("proportion of no_show ");


# 
# <div class="alert alert-block alert-info">
# <b>
#   rankingof the percentage of the patients who will come to their appointment on time according to their age <br>
#     
# </div>

# In[121]:


ranking= df.query('no_show=="No"').age_stage.value_counts()
rankin= df.age_stage.value_counts()
per=ranking/rankin*100 
per.sort_values(ascending=False)


# In[122]:


per.sort_values(ascending=False).plot(kind='bar',figsize=(15,5));


# <div class="alert alert-block alert-info">
# <b>
#   rankingof the percentage of the patients who will not come to their appointment  according to their age <br>
#     
# </div>
# 

# In[123]:


ranking= df.query('no_show=="Yes"').age_stage.value_counts()
rankin= df.age_stage.value_counts()
per=ranking/rankin*100
per.sort_values(ascending=False)


# In[124]:


per.sort_values(ascending=False).plot(kind='bar',figsize=(15,5));


# # ***conclusion :***

# 1. count number of patients who are bookig appointment in general is inversely proportional to their age which is not logic where the logic is as people become elder they will more expose to have diseases than youngers
# * to discover the reason for that we need to see main clasification of peoples in Brazil
# * patient with age more than or equal 55 years old have commitment for no_show appointment date more than 80% 
# * Babies  with age less than 3 years old have commitment for no_show appointment date almost 80%
# * patient with age range from 50 : 54  years old have commitment for no_show appointment date almost 80% 
# * patient with age range from 3 : 12  years old have commitment for no_show appointment date more than 75% but less than 80%
# *  patient with age range from 13 : 49  years old have commitment for no_show appointment date more than 70% and less than 75% 

# 
# <table>
#        
# | index |  age of patient | commitment for no_show appointment date |
# |:-----:|:---------------:|:---------------------------------------:|
# |   1   |   3 years old   |                   >80%                  |
# |   2   | 50:54 years old |               almost =80%               |
# |   3   |  3:12 years old |                 75%:80%                 |
# |   4   | 13:49 years old |                 70%:75%                 |
#    
# </table>

# In[125]:


df.head(3)


# # 2-waiting days study:

# 
# 
# <div class="alert alert-block alert-success">
# <b> Is there any  relationship of the waiting days between the schedule_day and appointment_day affect on the commitment to attends?
# </b> 
# </div>

# In[126]:


df.waiting_days.hist();


# In[127]:


df.waiting_days.describe()


# <div class="alert alert-block alert-info">
# <b>
#    what we have ? <br>
#     
# </div>
# <br>
# 1. **The range of waiting days start from 0 day up to 179 day which mean more than 4 monthes !!!** 
# * **The good news is that 75% of waiting days is 17 days or less** 
# 

# ### **To start analysis of relationshipe between waiting days and appointment attendens we need to answer**<br>
# # what are the amjor priods for waiting days ? 
# 

# # ***1.visualize relationship***

# In[128]:


df.waiting_days.value_counts().plot(kind='pie',figsize=(25,10));


# #### **what are the Descriptive statistics of waiting days more than 17 day which is equal to 75% of waiting days of data**

# In[129]:


df.query('waiting_days>17').describe()


#  
# 
# <div class="alert alert-block alert-info">
# <b>
# filtering the data based on waiting days less than or equal 37 days which is equal to 93.75% of total data*  <br>
#     
# </div>
# 

# In[130]:


df_37=df[df['waiting_days']<=37]


# In[131]:


no_show_mask_37=df_37[df_37['no_show']== 'No']
no_show_mask_37.shape


# In[132]:


print('Total number of patient who have waiting days upto 37 days are {} patient'.format(df_37.waiting_days.shape[0]))
print('Number of patient who show up  :  {}  patient'.format(no_show_mask_37.shape[0]))
print('percent of patient who show up {}'.format(no_show_mask_37.shape[0]/(df_37.waiting_days.shape[0])*100))
print('percent of patient who were Not show up {} patient'.format((1-no_show_mask_37.shape[0]/(df_37.waiting_days.shape[0]))*100))


# In[133]:


mal_noshow_37=df[(df.waiting_days<37) & (df.no_show == 'No')].shape[0]
mal_yesshow_37=df[(df.waiting_days<37) & (df.no_show =='Yes')].shape[0]
total=mal_noshow_37+mal_yesshow_37
proportion_no_37=mal_noshow_37/total
proportion_yes_37=mal_yesshow_37/total

colors=['blue','orange']
plt.bar(["No", "yes"], [proportion_no_37, proportion_yes_37],color=colors)
plt.title("no_show proportion grouped by waiting days ")
plt.xlabel("no_show")
plt.ylabel("proportion of no_show ");


# ## ***count number of patients who book appointments according to waiting days***

# In[134]:


df_37.waiting_days.value_counts().plot(kind='bar',figsize=(20,5))
plt.title('comparison for  count of patients in different waiting days')
plt.xlabel('waiting days')
plt.ylabel('count of No_show');


# ### ***find out trend lines of patient who show up and not show up according to waiting days***

# ## ***Trend lines of no_show according to patient's waiting days factor***

# ## ***Bulding up new column to follow no_show counts for every patient's waiting days***

# In[135]:


counts_no_yes=df.groupby('waiting_days').no_show.value_counts()
counts_sorting=counts_no_yes.sort_values(ascending=True)
counts_sorting_to_fram=counts_sorting.to_frame(name='no_show_count_waiting_days')
merged = pd.merge(df, counts_sorting_to_fram, on='waiting_days',how='inner')


# In[136]:


merged.shape


# In[137]:


merged.drop_duplicates(['appointment_id'],inplace =True)


# In[138]:


merged.shape


# In[139]:


df=merged


# In[140]:


df_37=df[df['waiting_days']<=37]


# In[141]:


df_no=df_37[df_37['no_show']=='No']


# In[142]:


df_no=df_37[df_37['no_show']=='No']
plt.figure(figsize = (20,4))
x = df_no['waiting_days']
y = df_no['no_show_count_waiting_days']
plt.title('relation beteen waiting days and count number of no_show')
plt.xlabel('waiting_days')
plt.ylabel('count of No in No_show')
plt.scatter(x, y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")
plt.show()


df_no=df_37[df_37['no_show']=='Yes']
plt.figure(figsize = (20,4))
x = df_no['waiting_days']
y = df_no['no_show_count_waiting_days']
plt.title('relation beteen waiting days and count number of no_show')
plt.xlabel('waiting_days')
plt.ylabel('count of yes in No_show')
plt.scatter(x, y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")
plt.show()


# 
# <div class="alert alert-block alert-info">
# <b>
#    from trend lines we see <br>
#     
# </div>
# 
# ### 1- in general count number of patients who show up is inversely proportional to their waiting days 
# ### 2- there are some some periodes where the trend line is directrly proprtional to their waiting days
# 
# #### we want to slice this  range according to different stageto study the relation between waiting days and no_show 
# 

# # ***2.build intution***

# # clasify waiting days in to different groups 
# The will be divided into 2 groups where 0 day mean today or in other words schedual day is appointment day<br>
# there are 11 group to cover the data range of waiting days<br>
# A:(0:2)waiting dayes<br>
# B:(3:6)waiting dayes<br>
# C:(7:9)waiting dayes<br>
# D:(10:13)waiting dayes<br>
# E:(14:16)waiting dayes<br>
# F:(17:20)waiting dayes<br>
# G:(21:23)waiting dayes<br>
# H:(24:27)waiting dayes<br>
# I:(28:30)waiting dayes<br>
# J:(31:34)waiting dayes<br>
# K:(35:37)waiting dayes<br>
# 
# ___

# # ***comparison for different waiting days periodes versuse no_show***

# In[143]:


# Bin edges that will be used to "cut" the data into groups
# we use -1 the start of sries to include  0 day
bin_edges = [-1,3,7,10,14,17,21,24,28,31,35,38] # list for range of waiting days 
# Labels for different age_stage groups
bin_names = ['A','B','C','D','E','F','G','H','I','J','K'] # Name each waiting period
# Creates waiting_days_period column
df['waiting_days_periodes'] = pd.cut(df['waiting_days'], bin_edges, labels=bin_names )
# Checks for successful creation of this column on 37 waiting days data frame
df_37=df[df['waiting_days']<=37]
df_37.head(100)


# In[144]:


df_37.groupby('no_show').waiting_days_periodes.count()


# In[145]:


df_37.groupby('no_show').waiting_days_periodes.count().plot(kind ='bar',figsize=(7,4));
plt.title('comparison for total count of patients')
plt.xlabel('waiting_days_periodes')
plt.ylabel('count of No_show');


# In[146]:


print("proportion of `No_show` for waiting_days_periodes  => {}".format(df_37.groupby('no_show').waiting_days_periodes.count()/df_37.waiting_days_periodes.count()))


# In[147]:


(df_37.groupby('no_show').waiting_days_periodes.count()/df_37.waiting_days_periodes.count()).plot(kind ='bar',figsize=(7,4))
plt.title('proportion of `No_show` for waiting_days_periodes')
plt.xlabel('waiting_days_periodes')
plt.ylabel('count of No_show');


# # ***sorting data sets for different waiting_days_periodes***

# In[148]:


df_37.waiting_days_periodes.value_counts()


# In[149]:


df_37['waiting_days_periodes'].value_counts().plot(kind='bar',figsize=(15,4));


# ## ***comparison of counts of patients for no_show in different waiting_days_periodes***

# In[150]:


fr=df_37.groupby('waiting_days_periodes').no_show.value_counts();
fr.sort_values(ascending=False).plot(kind='bar',figsize=(15,4))
plt.title('comparison for counts of patients in different waiting_days_periodes')
plt.xlabel('waiting_days_periodes')
plt.ylabel('count of No_show');


# ### proportional comparison of counts of patients for no_show in different waiting_days_periodes

# In[151]:


tr=((df_37.groupby('waiting_days_periodes').no_show.value_counts())/df_37.age_stage.count())*100
tr.sort_values(ascending=False).plot(kind='bar',figsize=(15,4))
plt.title('proportion of comparison for counts of patients in different waiting_days_periodes')
plt.xlabel('waiting_days_periodes')
plt.ylabel('count of No_show');


# 
# 
# <div class="alert alert-block alert-info">
# <b>
# neumerical proportional comparison for count of patients versus no_show in different waiting_days_periodes  <br>
#     
# </div>
# 

# In[152]:


tr=((df_37.groupby('waiting_days_periodes').no_show.value_counts())/df_37.waiting_days_periodes.count())*100
tr.sort_values(ascending=False)


# # ***3.creat more and describtive feture***

# 
# 
# <div class="alert alert-block alert-info">
# <b> rankingof the percentage of the patients who will come to their appointment on time according to their waiting_days_periodes  <br>
#     
# </div>
# 

# In[153]:


ranking= df_37.query('no_show=="No"'). waiting_days_periodes.value_counts()
rankin= df_37. waiting_days_periodes.value_counts()
per1=ranking/rankin*100 
per1.sort_values(ascending=False)


# In[154]:


per1.sort_values(ascending=False).plot(kind='bar',figsize=(15,5));


# 
# <div class="alert alert-block alert-info">
# <b> rankingof the percentage of the patients who will not come to their appointment on time according to their waiting_days_periodes  <br>
#     
# </div>
# 

# In[155]:


ranking= df_37.query('no_show=="Yes"').waiting_days_periodes.value_counts()
rankin= df_37.waiting_days_periodes.value_counts()
per2=ranking/rankin*100 
per2.sort_values(ascending=False)


# In[156]:


per2.sort_values(ascending=False).plot(kind='bar',figsize=(15,5));


# 
# # ***4.conclusion :***<br> 
#     
# count number of patients who are bookig appointment in general is inversely proportional to their waiting days which is logic<br>  
# 1. patient who wait from 0:2 dayes they have  commitment for no_show appointment date almost 90%<br> 
# * patient who wait from 3:6 dayes  the commitment for no_show appointment date drpoed to 71,12%<br> 
# * patient who wait from 7:9 dayes they have  commitment for no_show appointment date almost 67.67%<br> 
# ### surprse 1:<br>     
# * patient who wait from 28:30 dayes  have  commitment for no_show appointment date almost 66.39% which is not logic so those patients need more investigation!!!<br> 
# * patient who wait from 14:16 dayes they have  commitment for no_show appointment date almost  65.61%<br> 
# * patient who wait from 17:20 dayes they have  commitment for no_show appointment date almost 64.84%<br> 
# ### surprise2: <br>    
# * patient who wait from 10:13 dayes  have  commitment for no_show appointment date almost 64.60% which is not logic so those patients need more investigation!!!<br>     
# * patient who wait from 24:27 dayes they have  commitment for no_show appointment date almost  64.50% <br> 
# ### surprise3:<br>     
# * patient who wait from 35:37 dayes they have  commitment for no_show appointment date almost  64.33%  which is not logic so those patients need more investigation!!! <br>  
# * patient who wait from 31:34 dayes they have  commitment for no_show appointment date almost  63.26%<br> 
# ### surprise4:<br>     
# * patient who wait from 21:23 dayes they have  commitment for no_show appointment date almost   62.53%  which is not logic so those patients need more investigation!!!<br>       

# # ***NOTES:***<br>
# 
# 1. patient who wait from 28:30 dayes  have  commitment for no_show appointment date almost 66.39%  <br> 
# * patient who wait from 10:13 dayes  have  commitment for no_show appointment date almost 64.60% <br> 
# * patient who wait from 35:37 dayes they have  commitment for no_show appointment date almost  64.33%<br>     
# * patient who wait from 21:23 dayes they have  commitment for no_show appointment date almost   62.53% <br> 
# 
# ###### ***NOTE:***
# ***PERIOD BETWEEN 28:37 WAITING DAYS have commitment for no_show appointment date  MORE THAN PERIOD BETWEEN 10:23 WAITING DAYS***<br> 
# ***THAT MEAN ending and starting of month better than middel of the month may be that related to financial reasons***<br>             

# # ***3- gender study:***

# 
# <div class="alert alert-block alert-info">
# <b>Is there any relation between patients gender and their commitment to appointment attendens? </b>
# </div>
# 

# In[157]:


df.head(3)


# 
#  
# <div class="alert alert-block alert-info">
# <b>Note we have Nan values in waiting_dayes_periodes those NaN are for patient's waiting days more than 37 days here we need not to drop them</b>
# </div>
# 

# In[158]:


df.gender.describe()


# In[159]:


df_fem=df[df['gender']=='F']
df_mal=df[df['gender']=='M']


# In[160]:


f_count=df_fem.gender.count()
m_count=df_mal.gender.count()


# In[161]:


print('Total number of patient according to gender who record Appointment Booking {} patient in cleand data'.format(df.gender.shape[0]))
print('Number of femal patients :  ({})  patient'.format(df_fem.gender.count()))
print('Number of male patients :  ({})  patient'.format(df_mal.gender.count()))
print('percent of patient who are femals ({}) patient'.format(df_fem.gender.count()/(df.gender.shape[0])*100))
print('percent of patient who are male ({}) patient'.format(df_mal.gender.count()/(df.gender.shape[0])*100))


# In[162]:


total=df.gender.shape[0]
f_count=df_fem.gender.count()
m_count=df_mal.gender.count()
proportion_f=df_fem.gender.count()/(df.gender.shape[0])*100
proportion_m=df_mal.gender.count()/(df.gender.shape[0])*100
colors=['blue','orange']
plt.bar(["female gender", "male gender"], [proportion_f, proportion_m],color=colors)
plt.title("count number of gender ")
plt.xlabel("gender")
plt.ylabel("count proportion");


# # ***3.1visualize relationship***

# In[163]:


df.groupby(['gender','no_show']).count().age


# In[164]:


get_ipython().run_line_magic('matplotlib', 'inline')
df.groupby(['gender','no_show']).count().age.plot(kind='bar');


# In[165]:


df.groupby(['gender','no_show']).age.mean()


# In[166]:


get_ipython().run_line_magic('matplotlib', 'inline')
df.groupby(['gender','no_show']).mean().age.plot(kind='bar');


# In[167]:


df_fem.describe()


# In[168]:


df_mal.describe()


# <table>
# 
#  | index                    	| femal geder 	| male gender 	|
#  |--------------------------	|-------------	|-------------	|
#  |  age mean                	| 38.24       	| 32.8        	|
#  |  age range               	| 0 upto 115  	| 0 upto 100  	|
#  | 75% of data under age of 	| 56          	| 54          	|
# 
# <table><br>
# 
# 

# # ***3.2build intution***

# ___
# # ***what is age distribution for both males and females***

# In[169]:


# female age distribution
df_fem.age.hist();


# In[170]:


#male age distribution
df_mal.age.hist();


# 
# 
# <div class="alert alert-block alert-info">
# <b>Note</b>
# </div>
# 1. in general females are more healthy than males in data we have where the age mean in femals greater than males and max.age in females greater than male<br>
# *  age distribution for male and females clarify that Females are more concerned with their health more than males
# * To study if there is any relation between gender versuse no_show we will clasify this two groups according to their age<br> 

# 
# <div class="alert alert-block alert-info">
# <b>what is the trend comparison between females and males who are  show up according to their ages </b>
# </div>

# In[171]:


#femal trend
df_fem_no=df[(df.no_show == 'No') & (df.gender =='F')]
plt.figure(figsize = (20,4))
x = df_fem_no['age']
y = df_fem_no['no_show_count']
plt.title('relation beteen age and count number of no_show in females')
plt.xlabel('age')
plt.ylabel('count of No in No_show')
plt.scatter(x, y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")
plt.show()



#male trend
df_mal_no=df[(df.no_show == 'No') & (df.gender =='M')] 
plt.figure(figsize = (20,4))
x = df_mal_no['age']
y = df_mal_no['no_show_count']
plt.title('relation beteen age and count number of no_show for male')
plt.xlabel('age')
plt.ylabel('count of No in No_show')
plt.scatter(x, y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")
plt.show()


# 
# 
# <div class="alert alert-block alert-info">
# <b>what is the trend comparison between females and males who are not  show up according to their ages </b>
# </div>

# In[172]:


#femal trend 
df_fem_yes=df[(df.no_show == 'Yes') & (df.gender =='F')]
plt.figure(figsize = (20,4))
x = df_fem_yes['age']
y = df_fem_yes['no_show_count']
plt.title('relation beteen age and count number of no_show in females')
plt.xlabel('age')
plt.ylabel('count of Yes in No_show')
plt.scatter(x, y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")
plt.show()

#male trend
df_mal_yes=df[(df.no_show == 'Yes') & (df.gender =='M')] 
plt.figure(figsize = (20,4))
x = df_mal_yes['age']
y = df_mal_yes['no_show_count']
plt.title('relation beteen age and count number of no_show for male')
plt.xlabel('age')
plt.ylabel('count of Yes in No_show')
plt.scatter(x, y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")
plt.show()


# 
# 
# <div class="alert alert-block alert-info">
# <b>Note </b>
# </div>
# 1. trend lines slope for females are inverversely proportional to show up and not show up
# * trend lines for males are inverversely proportional to show up and not show up
# * There is a clear difference between the two genders in their numbers compared to the same age period

# 
# 
# <div class="alert alert-block alert-info">
# <b>does gender in different age stage affects their commitment to show up on the appointement date?</b>
# </div>

# In[173]:


#the count number of females in different age stage
df_fem.age.value_counts().plot('bar',figsize=(20,5))
plt.title('comparison for  count of females in different age')
plt.xlabel('age')
plt.ylabel('count of patients');


# In[174]:


#the count number of males in different age stage
df_mal.age.value_counts().plot(kind='bar',figsize=(20,4))
plt.title('comparison for  count of males in different age')
plt.xlabel('age')
plt.ylabel('count of patients');


# # ***count of patients in every age stage are completly different***
# ##### we going to make deep analysis for different age stage according to their gender 

# # 3.3creat more and describtive feture

# ___
# # age stages
# 1. Fetus (Unborn)<br>
# * Newborn (Birth - 1 month)<br>
# * Baby (1 month and 1 day - 2 years)<br>
# * Toddler (3 - 5)<br> 
# * Kids (6 - 9)<br>
# * Pre-Teen (10 - 12)<br>
# * Teenager (13 - 17)<br>
# * Young Adult (18 - 20)<br>
# * Adult (21 - 39)<br>
# * Young Middle-Aged Adult (40 - 49)<br>
# * Middle-Aged Adult (50 - 54)<br>
# * Very Young Senior Citizen (55 - 64)<br>
# * Young Senior Citizen (65 - 74)<br>
# * Senior Citizen (75 - 84)<br> 
# * Old Senior Citizen (85+)
# 

# 
# 
# <div class="alert alert-block alert-info">
# <b>comparison for different age stages versuse no_show in different gender </b>
# </div>
# 
# ___

# # femal study:

# In[175]:


df_fem.groupby('no_show').age_stage.count()


# In[176]:


df_fem.groupby('no_show').age_stage.count().plot(kind ='bar',figsize=(7,4))
plt.title('comparison for total count of female patients')
plt.xlabel('age')
plt.ylabel('count of No_show');


# In[177]:


print("proportion of `No_show` for age_stage  => {}".format(df_fem.groupby('no_show').age_stage.count()/df_fem.age_stage.count()))


# In[178]:


(df_fem.groupby('no_show').age_stage.count()/df_fem.age_stage.count()).plot(kind ='bar',figsize=(7,4))
plt.title('proportion of `No_show` for age_stage')
plt.xlabel('age')
plt.ylabel('count of No_show');


# # male study:

# In[179]:


df_mal.groupby('no_show').age_stage.count()


# In[180]:


df_mal.groupby('no_show').age_stage.count().plot(kind ='bar',figsize=(7,4))
plt.title('comparison for total count of male patients')
plt.xlabel('age')
plt.ylabel('count of No_show');


# In[181]:


print("proportion of `No_show` for age_stage  => {}".format(df_mal.groupby('no_show').age_stage.count()/df_mal.age_stage.count()))


# In[182]:


(df_mal.groupby('no_show').age_stage.count()/df_mal.age_stage.count()).plot(kind ='bar',figsize=(7,4))
plt.title('proportion of `No_show` for age_stage')
plt.xlabel('age')
plt.ylabel('count of No_show');


# 
# ___
# <div class="alert alert-block alert-info">
# <b>sorting data sets for different age stages</b>
# </div>

# # femal study

# In[183]:


df_fem.age_stage.value_counts()


# In[184]:


df_fem['age_stage'].value_counts().plot(kind='bar',figsize=(15,4));


# # male study

# In[185]:


df_mal.age_stage.value_counts()


# In[186]:


df_mal['age_stage'].value_counts().plot(kind='bar',figsize=(15,4));


# 
# ___
# <div class="alert alert-block alert-info">
# <b>comparison of counts of patients for no_show in different age_stage </b>
# </div>

# # femal study

# In[187]:


fr=df_fem.groupby('age_stage').no_show.value_counts();
fr.sort_values(ascending=False).plot(kind='bar',figsize=(15,4))
plt.title('comparison for counts of patients in different age_stage')
plt.xlabel('age')
plt.ylabel('count of No_show');


# ___
# <div class="alert alert-block alert-info">
# <b>proportional comparison of counts of females for no_show in different age_stage</b>
# </div>

# In[188]:


tr=((df_fem.groupby('age_stage').no_show.value_counts())/df.age_stage.count())*100
tr.sort_values(ascending=False).plot(kind='bar',figsize=(15,4))
plt.title('proportion of comparison for counts of patients in different age_stage')
plt.xlabel('age')
plt.ylabel('count of No_show');


# # male study

# In[189]:


fr=df_mal.groupby('age_stage').no_show.value_counts();
fr.sort_values(ascending=False).plot(kind='bar',figsize=(15,4))
plt.title('comparison for counts of patients in different age_stage')
plt.xlabel('age')
plt.ylabel('count of No_show');


# ___
# <div class="alert alert-block alert-info">
# <b>proportional comparison of counts of males for no_show in different age_stage</b>
# </div>

# In[190]:


tr=((df_mal.groupby('age_stage').no_show.value_counts())/df.age_stage.count())*100
tr.sort_values(ascending=False).plot(kind='bar',figsize=(15,4))
plt.title('proportion of comparison for counts of patients in different age_stage')
plt.xlabel('age')
plt.ylabel('count of No_show');


# ___
# <div class="alert alert-block alert-info">
# <b>neumerical proportional comparison for count of patients versus no_show in different age_stage</b>
# </div>

# # femal study

# In[191]:


tr=((df_fem.groupby('age_stage').no_show.value_counts())/df_fem.age_stage.count())*100
tr.sort_values(ascending=False)


# # male study

# In[192]:


tr=((df_mal.groupby('age_stage').no_show.value_counts())/df_mal.age_stage.count())*100
tr.sort_values(ascending=False)


# ___
# 
# <div class="alert alert-block alert-info">
# <b> ranking of the percentage of the patients who will come to their appointment on time according to their age</b>
# </div>

# # femal study

# In[193]:


ranking= df_fem.query('no_show=="No"').age_stage.value_counts()
rankin= df_fem.age_stage.value_counts()
per=ranking/rankin*100 
per.sort_values(ascending=False)


# In[194]:


per.sort_values(ascending=False).plot(kind='bar',figsize=(15,5));


# # male study

# In[195]:


ranking= df_mal.query('no_show=="No"').age_stage.value_counts()
rankin= df_mal.age_stage.value_counts()
per=ranking/rankin*100 
per.sort_values(ascending=False)


# In[196]:


per.sort_values(ascending=False).plot(kind='bar',figsize=(15,5));


# 
# ___
# <div class="alert alert-block alert-info">
# <b> ranking of the percentage of the patients who will not come to their appointment on time according to their age</b>
# </div>

# # femal study

# In[197]:


ranking= df_fem.query('no_show=="Yes"').age_stage.value_counts()
rankin= df_fem.age_stage.value_counts()
per=ranking/rankin*100 
per.sort_values(ascending=False)


# In[198]:


per.sort_values(ascending=False).plot(kind='bar',figsize=(15,5));


# # male study

# In[199]:


ranking= df_mal.query('no_show=="Yes"').age_stage.value_counts()
rankin= df_mal.age_stage.value_counts()
per=ranking/rankin*100 
per.sort_values(ascending=False)


# In[200]:


per.sort_values(ascending=False).plot(kind='bar',figsize=(15,5));


# # ***3.4conclusion :***
# 
# ___

# <table>
# 
# | age stage                 	| age in years                	| % of expected commitment for   appointment date   in females  (no_show==NO)   	| % of expected commitment for   appointment date   in males  (no_show==NO) 	|
# |---------------------------	|-----------------------------	|:-----------------------------------------------------------------------------:	|:-------------------------------------------------------------------------:	|
# | Baby                      	| 1 month and 1 day - 2 years 	| 79.156                                                                        	| 80.039                                                                    	|
# | Toddler                   	| 3 - 5                       	| 74.596                                                                        	| 76.825                                                                    	|
# | Kids                      	| 6 - 9                       	| 76.325                                                                        	| 75.130                                                                    	|
# | Pre-Teen                  	| 10 - 12                     	| 74.673                                                                        	| 75.636                                                                    	|
# | Teenage                   	| 13 - 17                     	| 72.847                                                                        	| 72968                                                                     	|
# | Young Adult               	| 18 - 20                     	| 72.906                                                                        	| 75.539                                                                    	|
# | Adult                     	| 21 - 39                     	| 73.850                                                                        	| 73.607                                                                    	|
# | Young Middle-Aged Adult   	| 40 - 49                     	| 77.717                                                                        	| 77.205                                                                    	|
# | Middle-Aged Adult         	| 50 - 54                     	| 80.789                                                                        	| 77.113                                                                    	|
# | Very Young Senior Citizen 	| 55 - 64                     	| 81.730                                                                        	| 83.066                                                                    	|
# | Young Senior Citizen      	| 65 - 74                     	| 83.181                                                                        	| 84.153                                                                    	|
# | Senior Citizen            	| 75 - 84                     	| 82.689                                                                        	| 83.405                                                                    	|
# | Old Senior Citizen        	| 85+                         	| 83.448                                                                        	| 82.027                                                                    	|
# 
# <table>

# 1. % of expected commitment for appointment date for age stage from  0 upto 20 years old in males greater than females <br> except in for age stage from 6:9 years
# *  % of expected commitment for appointment date for age stage from  55 upto 84 years old in males greater than females <br> 
# ___
# *  % of expected commitment for appointment date for age stage from  21 upto 54 years old in males greater than females <br>
# *  % of expected commitment for appointment date for age stage from  +85 years old in females greater than males <br> 
# 

# # final conclusion 
# ___
# 
#         
#   
# 
# <div class="alert alert-block alert-success">
# <b>age ,waiting days and gender of patient affect  % of expected commitment for appointment date  </b> 
# </div>
# 
