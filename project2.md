# Project: Investigate a Dataset (no show appointments!)
<ul>
<li><a href="#Introduction">Introduction</a></li>
<li><a href="#wrangling">Data Wrangling</a></li>
<li><a href="#eda">Exploratory Data Analysis</a></li>
<li><a href="#conclusions">Conclusions</a></li>
</ul>

# <a id='intro'></a>
## Introduction

in this project i wiil analyze (no show appointments)


first i will understand the data set which is provided in Kaggl
then clean it and apply  function to it, then I will ask questions  then i will try to answer



the following questions


1-Do patients who receive SMS messages have a higher rate of attendance to their appointments than others?

2 Does a patient with diabetes have a higher attendance rate?

3- Does have a Handcap affect showing up the appointment?

4 Does the age affect showing up the appointment?



```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

<a id='wrangling'></a>
## Data Wrangling



first i wii read the fill From Kaggl


```python
df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
df.head(20)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentID</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handcap</th>
      <th>SMS_received</th>
      <th>No-show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.987250e+13</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29T18:38:08Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.589978e+14</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29T16:08:27Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.262962e+12</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29T16:19:04Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.679512e+11</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29T17:29:31Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.841186e+12</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29T16:07:23Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9.598513e+13</td>
      <td>5626772</td>
      <td>F</td>
      <td>2016-04-27T08:36:51Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>76</td>
      <td>REPÚBLICA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.336882e+14</td>
      <td>5630279</td>
      <td>F</td>
      <td>2016-04-27T15:05:12Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>23</td>
      <td>GOIABEIRAS</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.449833e+12</td>
      <td>5630575</td>
      <td>F</td>
      <td>2016-04-27T15:39:58Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>39</td>
      <td>GOIABEIRAS</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5.639473e+13</td>
      <td>5638447</td>
      <td>F</td>
      <td>2016-04-29T08:02:16Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>21</td>
      <td>ANDORINHAS</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7.812456e+13</td>
      <td>5629123</td>
      <td>F</td>
      <td>2016-04-27T12:48:25Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>19</td>
      <td>CONQUISTA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>10</th>
      <td>7.345362e+14</td>
      <td>5630213</td>
      <td>F</td>
      <td>2016-04-27T14:58:11Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>30</td>
      <td>NOVA PALESTINA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>11</th>
      <td>7.542951e+12</td>
      <td>5620163</td>
      <td>M</td>
      <td>2016-04-26T08:44:12Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>29</td>
      <td>NOVA PALESTINA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5.666548e+14</td>
      <td>5634718</td>
      <td>F</td>
      <td>2016-04-28T11:33:51Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>22</td>
      <td>NOVA PALESTINA</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>13</th>
      <td>9.113946e+14</td>
      <td>5636249</td>
      <td>M</td>
      <td>2016-04-28T14:52:07Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>28</td>
      <td>NOVA PALESTINA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>14</th>
      <td>9.988472e+13</td>
      <td>5633951</td>
      <td>F</td>
      <td>2016-04-28T10:06:24Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>54</td>
      <td>NOVA PALESTINA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>15</th>
      <td>9.994839e+10</td>
      <td>5620206</td>
      <td>F</td>
      <td>2016-04-26T08:47:27Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>15</td>
      <td>NOVA PALESTINA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>No</td>
    </tr>
    <tr>
      <th>16</th>
      <td>8.457439e+13</td>
      <td>5633121</td>
      <td>M</td>
      <td>2016-04-28T08:51:47Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>50</td>
      <td>NOVA PALESTINA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.479497e+13</td>
      <td>5633460</td>
      <td>F</td>
      <td>2016-04-28T09:28:57Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>40</td>
      <td>CONQUISTA</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1.713538e+13</td>
      <td>5621836</td>
      <td>F</td>
      <td>2016-04-26T10:54:18Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>30</td>
      <td>NOVA PALESTINA</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>No</td>
    </tr>
    <tr>
      <th>19</th>
      <td>7.223289e+12</td>
      <td>5640433</td>
      <td>F</td>
      <td>2016-04-29T10:43:14Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>46</td>
      <td>DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



## look of some calculations


```python
df.tail()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentID</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handcap</th>
      <th>SMS_received</th>
      <th>No-show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110522</th>
      <td>2.572134e+12</td>
      <td>5651768</td>
      <td>F</td>
      <td>2016-05-03T09:15:35Z</td>
      <td>2016-06-07T00:00:00Z</td>
      <td>56</td>
      <td>MARIA ORTIZ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>No</td>
    </tr>
    <tr>
      <th>110523</th>
      <td>3.596266e+12</td>
      <td>5650093</td>
      <td>F</td>
      <td>2016-05-03T07:27:33Z</td>
      <td>2016-06-07T00:00:00Z</td>
      <td>51</td>
      <td>MARIA ORTIZ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>No</td>
    </tr>
    <tr>
      <th>110524</th>
      <td>1.557663e+13</td>
      <td>5630692</td>
      <td>F</td>
      <td>2016-04-27T16:03:52Z</td>
      <td>2016-06-07T00:00:00Z</td>
      <td>21</td>
      <td>MARIA ORTIZ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>No</td>
    </tr>
    <tr>
      <th>110525</th>
      <td>9.213493e+13</td>
      <td>5630323</td>
      <td>F</td>
      <td>2016-04-27T15:09:23Z</td>
      <td>2016-06-07T00:00:00Z</td>
      <td>38</td>
      <td>MARIA ORTIZ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>No</td>
    </tr>
    <tr>
      <th>110526</th>
      <td>3.775115e+14</td>
      <td>5629448</td>
      <td>F</td>
      <td>2016-04-27T13:30:56Z</td>
      <td>2016-06-07T00:00:00Z</td>
      <td>54</td>
      <td>MARIA ORTIZ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentID</th>
      <th>Age</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handcap</th>
      <th>SMS_received</th>
      <th>NoShow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.105260e+05</td>
      <td>1.105260e+05</td>
      <td>110526.000000</td>
      <td>110526.000000</td>
      <td>110526.000000</td>
      <td>110526.000000</td>
      <td>110526.000000</td>
      <td>110526.000000</td>
      <td>110526.000000</td>
      <td>110526.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.474934e+14</td>
      <td>5.675304e+06</td>
      <td>37.089219</td>
      <td>0.098266</td>
      <td>0.197248</td>
      <td>0.071865</td>
      <td>0.030400</td>
      <td>0.022248</td>
      <td>0.321029</td>
      <td>0.201934</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.560943e+14</td>
      <td>7.129544e+04</td>
      <td>23.110026</td>
      <td>0.297676</td>
      <td>0.397923</td>
      <td>0.258266</td>
      <td>0.171686</td>
      <td>0.161543</td>
      <td>0.466874</td>
      <td>0.401445</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.921784e+04</td>
      <td>5.030230e+06</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.172536e+12</td>
      <td>5.640285e+06</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.173184e+13</td>
      <td>5.680572e+06</td>
      <td>37.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.438963e+13</td>
      <td>5.725523e+06</td>
      <td>55.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.999816e+14</td>
      <td>5.790484e+06</td>
      <td>115.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (110527, 14)



## 1-i want to check null value




```python
#check for missing values
df.isnull().sum()
```




    PatientId         0
    AppointmentID     0
    Gender            0
    ScheduledDay      0
    AppointmentDay    0
    Age               0
    Neighbourhood     0
    Scholarship       0
    Hipertension      0
    Diabetes          0
    Alcoholism        0
    Handcap           0
    SMS_received      0
    NoShow            0
    dtype: int64



i didnot find null value


## 2 then i check the duplicate value



```python
sum(df.duplicated())

```




    0




```python
df.dtypes

```




    PatientId                     float64
    AppointmentID                   int64
    Gender                         object
    ScheduledDay      datetime64[ns, UTC]
    AppointmentDay    datetime64[ns, UTC]
    Age                             int64
    Neighbourhood                  object
    Scholarship                     int64
    Hipertension                    int64
    Diabetes                        int64
    Alcoholism                      int64
    Handcap                         int64
    SMS_received                    int64
    NoShow                          int64
    dtype: object



##  information before cleaning




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 110527 entries, 0 to 110526
    Data columns (total 14 columns):
     #   Column          Non-Null Count   Dtype              
    ---  ------          --------------   -----              
     0   PatientId       110527 non-null  float64            
     1   AppointmentID   110527 non-null  int64              
     2   Gender          110527 non-null  object             
     3   ScheduledDay    110527 non-null  datetime64[ns, UTC]
     4   AppointmentDay  110527 non-null  datetime64[ns, UTC]
     5   Age             110527 non-null  int64              
     6   Neighbourhood   110527 non-null  object             
     7   Scholarship     110527 non-null  int64              
     8   Hipertension    110527 non-null  int64              
     9   Diabetes        110527 non-null  int64              
     10  Alcoholism      110527 non-null  int64              
     11  Handcap         110527 non-null  int64              
     12  SMS_received    110527 non-null  int64              
     13  No-show         110527 non-null  object             
    dtypes: datetime64[ns, UTC](2), float64(1), int64(8), object(3)
    memory usage: 11.8+ MB
    

## clean the dataset
 after the assum the data set using by  function df.info()" and "df.duplicated().sum()"
 there is no null value and duplicate value
 so in  next step i will change the  type of data of ScheduledDay and AppointmentDay becususe so difficult work in the string value. also i shoud be change the nigtive age


```python
df['AppointmentDay']= pd.to_datetime(df['AppointmentDay'])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentID</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handcap</th>
      <th>SMS_received</th>
      <th>No-show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.987250e+13</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29 18:38:08+00:00</td>
      <td>2016-04-29 00:00:00+00:00</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.589978e+14</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29 16:08:27+00:00</td>
      <td>2016-04-29 00:00:00+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.262962e+12</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29 16:19:04+00:00</td>
      <td>2016-04-29 00:00:00+00:00</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.679512e+11</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29 17:29:31+00:00</td>
      <td>2016-04-29 00:00:00+00:00</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.841186e+12</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29 16:07:23+00:00</td>
      <td>2016-04-29 00:00:00+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentID</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handcap</th>
      <th>SMS_received</th>
      <th>No-show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.987250e+13</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29 18:38:08+00:00</td>
      <td>2016-04-29 00:00:00+00:00</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.589978e+14</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29 16:08:27+00:00</td>
      <td>2016-04-29 00:00:00+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.262962e+12</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29 16:19:04+00:00</td>
      <td>2016-04-29 00:00:00+00:00</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.679512e+11</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29 17:29:31+00:00</td>
      <td>2016-04-29 00:00:00+00:00</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.841186e+12</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29 16:07:23+00:00</td>
      <td>2016-04-29 00:00:00+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.query('Age < 0').count()

```




    PatientId         0
    AppointmentID     0
    Gender            0
    ScheduledDay      0
    AppointmentDay    0
    Age               0
    Neighbourhood     0
    Scholarship       0
    Hipertension      0
    Diabetes          0
    Alcoholism        0
    Handcap           0
    SMS_received      0
    NoShow            0
    dtype: int64




```python
df.info()

```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 110526 entries, 0 to 110526
    Data columns (total 14 columns):
     #   Column          Non-Null Count   Dtype              
    ---  ------          --------------   -----              
     0   PatientId       110526 non-null  float64            
     1   AppointmentID   110526 non-null  int64              
     2   Gender          110526 non-null  object             
     3   ScheduledDay    110526 non-null  datetime64[ns, UTC]
     4   AppointmentDay  110526 non-null  datetime64[ns, UTC]
     5   Age             110526 non-null  int64              
     6   Neighbourhood   110526 non-null  object             
     7   Scholarship     110526 non-null  int64              
     8   Hipertension    110526 non-null  int64              
     9   Diabetes        110526 non-null  int64              
     10  Alcoholism      110526 non-null  int64              
     11  Handcap         110526 non-null  int64              
     12  SMS_received    110526 non-null  int64              
     13  NoShow          110526 non-null  int64              
    dtypes: datetime64[ns, UTC](2), float64(1), int64(9), object(2)
    memory usage: 12.6+ MB
    


```python
df.head(20)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentID</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handcap</th>
      <th>SMS_received</th>
      <th>No-show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.987250e+13</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29T18:38:08Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.589978e+14</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29T16:08:27Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.262962e+12</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29T16:19:04Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.679512e+11</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29T17:29:31Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.841186e+12</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29T16:07:23Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9.598513e+13</td>
      <td>5626772</td>
      <td>F</td>
      <td>2016-04-27T08:36:51Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>76</td>
      <td>REPÚBLICA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.336882e+14</td>
      <td>5630279</td>
      <td>F</td>
      <td>2016-04-27T15:05:12Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>23</td>
      <td>GOIABEIRAS</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.449833e+12</td>
      <td>5630575</td>
      <td>F</td>
      <td>2016-04-27T15:39:58Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>39</td>
      <td>GOIABEIRAS</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5.639473e+13</td>
      <td>5638447</td>
      <td>F</td>
      <td>2016-04-29T08:02:16Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>21</td>
      <td>ANDORINHAS</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7.812456e+13</td>
      <td>5629123</td>
      <td>F</td>
      <td>2016-04-27T12:48:25Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>19</td>
      <td>CONQUISTA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>10</th>
      <td>7.345362e+14</td>
      <td>5630213</td>
      <td>F</td>
      <td>2016-04-27T14:58:11Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>30</td>
      <td>NOVA PALESTINA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>11</th>
      <td>7.542951e+12</td>
      <td>5620163</td>
      <td>M</td>
      <td>2016-04-26T08:44:12Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>29</td>
      <td>NOVA PALESTINA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5.666548e+14</td>
      <td>5634718</td>
      <td>F</td>
      <td>2016-04-28T11:33:51Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>22</td>
      <td>NOVA PALESTINA</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>13</th>
      <td>9.113946e+14</td>
      <td>5636249</td>
      <td>M</td>
      <td>2016-04-28T14:52:07Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>28</td>
      <td>NOVA PALESTINA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>14</th>
      <td>9.988472e+13</td>
      <td>5633951</td>
      <td>F</td>
      <td>2016-04-28T10:06:24Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>54</td>
      <td>NOVA PALESTINA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>15</th>
      <td>9.994839e+10</td>
      <td>5620206</td>
      <td>F</td>
      <td>2016-04-26T08:47:27Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>15</td>
      <td>NOVA PALESTINA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>No</td>
    </tr>
    <tr>
      <th>16</th>
      <td>8.457439e+13</td>
      <td>5633121</td>
      <td>M</td>
      <td>2016-04-28T08:51:47Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>50</td>
      <td>NOVA PALESTINA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.479497e+13</td>
      <td>5633460</td>
      <td>F</td>
      <td>2016-04-28T09:28:57Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>40</td>
      <td>CONQUISTA</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1.713538e+13</td>
      <td>5621836</td>
      <td>F</td>
      <td>2016-04-26T10:54:18Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>30</td>
      <td>NOVA PALESTINA</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>No</td>
    </tr>
    <tr>
      <th>19</th>
      <td>7.223289e+12</td>
      <td>5640433</td>
      <td>F</td>
      <td>2016-04-29T10:43:14Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>46</td>
      <td>DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



## EDA 
Exploratory Data Analysis 

## 1 Do patients who receive SMS messages have a higher rate of attendance to their appointments than others?





```python
df.hist(figsize=(20,10))

```




    array([[<AxesSubplot:title={'center':'PatientId'}>,
            <AxesSubplot:title={'center':'AppointmentID'}>,
            <AxesSubplot:title={'center':'Age'}>],
           [<AxesSubplot:title={'center':'Scholarship'}>,
            <AxesSubplot:title={'center':'Hipertension'}>,
            <AxesSubplot:title={'center':'Diabetes'}>],
           [<AxesSubplot:title={'center':'Alcoholism'}>,
            <AxesSubplot:title={'center':'Handcap'}>,
            <AxesSubplot:title={'center':'SMS_received'}>]], dtype=object)




    
![png](output_26_1.png)
    



```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 110527 entries, 0 to 110526
    Data columns (total 14 columns):
     #   Column          Non-Null Count   Dtype  
    ---  ------          --------------   -----  
     0   PatientId       110527 non-null  float64
     1   AppointmentID   110527 non-null  int64  
     2   Gender          110527 non-null  object 
     3   ScheduledDay    110527 non-null  object 
     4   AppointmentDay  110527 non-null  object 
     5   Age             110527 non-null  int64  
     6   Neighbourhood   110527 non-null  object 
     7   Scholarship     110527 non-null  int64  
     8   Hipertension    110527 non-null  int64  
     9   Diabetes        110527 non-null  int64  
     10  Alcoholism      110527 non-null  int64  
     11  Handcap         110527 non-null  int64  
     12  SMS_received    110527 non-null  int64  
     13  NoShow          110527 non-null  object 
    dtypes: float64(1), int64(8), object(5)
    memory usage: 11.8+ MB
    


```python
df.rename(columns={'No-show' : 'NoShow'},inplace=True)
```


```python
df.NoShow.hist()
```




    <AxesSubplot:>




    
![png](output_29_1.png)
    



```python
sh = (df['NoShow'] == 'Yes')
n_sh = (df['NoShow'] == 'No')
```


```python
df.groupby(['NoShow']).mean()['SMS_received']
```




    NoShow
    No     0.291334
    Yes    0.438371
    Name: SMS_received, dtype: float64




```python
df.SMS_received[n_sh].hist(alpha=0.9, label=' no show')
df.SMS_received[sh].hist(alpha=0.9, label= 'show')
plt.title('Association between SMS_received and NoShow')
plt.xlabel('show up')
plt.ylabel('SMS_received')
plt.legend();
```


    
![png](output_32_0.png)
    


when lock at this map we see pepole  did not reseved the messege SMS  had a greater rate to show up than those pepole who received SMS

  ##  2 Does a patient with diabetes have a higher attendance rate?


```python
df.groupby(['NoShow']).mean().Diabetes

```




    NoShow
    No     0.073837
    Yes    0.064071
    Name: Diabetes, dtype: float64




```python
def setLabels(X , Y , T):
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.title(T)
    plt.legend();

```


```python
df.Diabetes[n_sh].hist(alpha=0.8, label='have not Diabetes')
df.Diabetes[sh].hist(alpha=0.8, label='have Diabetes')

setLabels('show up' ,'Diabetes', 'Association between Diabetes and NoShow' )
# plt.xlabel('show up')
# plt.ylabel('Diabetes')
# plt.title('Association between Diabetes and NoShow')
# plt.legend();
```


    
![png](output_37_0.png)
    


Diabetics who attend their appointments are higher than others

## 3 Does have a Handcap affect showing up the appointment?


```python
df.Handcap[n_sh].hist(alpha=0.8, label='have not Handcap')
df.Handcap[sh].hist(alpha=0.8, label='have Handcap')
setLabels('show up' ,'Handcap', 'Association between Handcap and NoShow' )

# plt.xlabel('show up')
# plt.ylabel('Handcap')
# plt.title('Association between Handcap and NoShow')
# plt.legend();
```


    
![png](output_40_0.png)
    


when see this map Patients with Handcap will have a higher rate of attendance at the appointment

##  4 Does the age affect showing up the appointment?



```python
df.Age[n_sh].hist(alpha=0.8, label='have not Age')
df.Age[sh].hist(alpha=0.8, label='have Age')
setLabels('show up' ,'Age', 'Association between Age and NoShow' )

# plt.xlabel('show up')
# plt.ylabel('Age')
# plt.title('Association between Age and NoShow')
# plt.legend();
```


    
![png](output_43_0.png)
    


## Conclusions

Finally this report, I say that the result came after hardwork with facing some limitations 

I work with this datasets and I managed a some problem For-example



 ckeck null value 

ckeck duplicate value

ages that are not possible.

 ## after doing analysis the dataset i can answer those questions below:

1 Do patients who receive SMS messages have a higher rate of attendance to their appointments than others?

The patients that who did not receive SMS had a greater rate to show up than those who received SMS

2 Does a patient with diabetes have a higher attendance rate?

Diabetics who attend their appointments are higher than others

3 Does have a Handcap affect showing up the appointment?

 Patients with Handcap will have a higher rate of attendance at the appointment

4 Does the age affect showing up the appointment?

Yes, because of  no show having age more

## limitations

I found the results of a dataset  (no show appointments!), but the analysis would have been more helpful if the data had details about appointments for each year in brazil
