#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#QUestion 1


import pandas as pd

url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user"

users = pd.read_csv(url, sep='|')

mean_age_per_occupation = users.groupby('occupation')['age'].mean()

print("Mean age per occupation:")

print(mean_age_per_occupation)

def male_ratio(group):
    male_count = (group['gender'] == 'M').sum()
    total_count = group['gender'].count()
    return male_count / total_count

male_ratio_per_occupation = users.groupby('occupation').apply(male_ratio).sort_values(ascending=False)
print("\nMale ratio per occupation (sorted):")
print(male_ratio_per_occupation)


min_max_age_per_occupation = users.groupby('occupation')['age'].agg(['min', 'max'])
print("\nMinimum and maximum ages per occupation:")
print(min_max_age_per_occupation)


mean_age_per_occupation_sex = users.groupby(['occupation', 'gender'])['age'].mean()
print("\nMean age for each combination of occupation and sex:")
print(mean_age_per_occupation_sex)


occupation_gender_counts = users.groupby(['occupation', 'gender']).size()
occupation_total_counts = users.groupby('occupation').size()

occupation_gender_percentages = (occupation_gender_counts / occupation_total_counts * 100).unstack()
occupation_gender_percentages = occupation_gender_percentages.rename(columns={'M': 'Male%', 'F': 'Female%'})
print("\nPercentage of women and men for each occupation:")
print(occupation_gender_percentages)


# In[ ]:


#Question 2
import pandas as pd

url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv"

euro12 = pd.read_csv(url)

goal_column = euro12['Goals']
print("Goal column:")
print(goal_column)

num_teams = euro12['Team'].nunique()
print("\nNumber of teams participated in Euro2012:", num_teams)

num_columns = len(euro12.columns)
print("\nNumber of columns in the dataset:", num_columns)

discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]
print("\nDiscipline dataframe:")
print(discipline)

sorted_discipline = discipline.sort_values(by=['Red Cards', 'Yellow Cards'], ascending=False)
print("\nTeams sorted by Red Cards, then by Yellow Cards:")
print(sorted_discipline)

mean_yellow_cards = euro12['Yellow Cards'].mean()
print("\nMean Yellow Cards given per Team:", mean_yellow_cards)

teams_gt_6_goals = euro12[euro12['Goals'] > 6]
print("\nTeams that scored more than 6 goals:")
print(teams_gt_6_goals)

teams_start_with_G = euro12[euro12['Team'].str.startswith('G')]
print("\nTeams that start with G:")
print(teams_start_with_G)

first_7_columns = euro12.iloc[:, :7]
print("\nFirst 7 columns:")
print(first_7_columns)

all_except_last_3_columns = euro12.iloc[:, :-3]
print("\nAll columns except the last 3:")
print(all_except_last_3_columns)

accuracy_selected_countries = euro12.loc[euro12['Team'].isin(['England', 'Italy', 'Russia']), ['Team', 'Shooting Accuracy']]
print("\nShooting Accuracy from England, Italy and Russia:")
print(accuracy_selected_countries)


# In[ ]:


#quesion no 3

import pandas as pd
import numpy as np

series1 = pd.Series(np.random.randint(1, 5, size=100))
series2 = pd.Series(np.random.randint(1, 4, size=100))
series3 = pd.Series(np.random.randint(10000, 30001, size=100))

data = {'bedrs': series1, 'bathrs': series2, 'price_sqr_meter': series3}
df = pd.DataFrame(data)

df.columns = ['bedrs', 'bathrs', 'price_sqr_meter']

bigcolumn = pd.concat([series1, series2, series3], axis=0)

print("Is it true that 'bigcolumn' goes only until index 99?", bigcolumn.index.max() == 99)

bigcolumn.reset_index(drop=True, inplace=True)


print("\nDataFrame 'bigcolumn' reindexed:")
print(bigcolumn)


# In[ ]:


#Question no 4

import pandas as pd

url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/Wind_Stats/wind.data"

data = pd.read_csv(url, delim_whitespace=True, parse_dates=[[0, 1, 2]])

def fix_year(x):
    year = x.year - 100 if x.year > 1989 else x.year
    return pd.Timestamp(year, x.month, x.day)

data['Yr_Mo_Dy'] = data['Yr_Mo_Dy'].apply(fix_year)

data.set_index('Yr_Mo_Dy', inplace=True)

missing_values_per_location = data.isnull().sum()

total_non_missing_values = data.notnull().sum().sum()

mean_windspeed = data.mean().mean()

loc_stats = data.describe(percentiles=[])

day_stats = pd.DataFrame()
day_stats['min'] = data.min(axis=1)
day_stats['max'] = data.max(axis=1)
day_stats['mean'] = data.mean(axis=1)
day_stats['std'] = data.std(axis=1)


january_avg_windspeed = data[data.index.month == 1].mean()

yearly_data = data.resample('Y').mean()

monthly_data = data.resample('M').mean()

weekly_data = data.resample('W').mean()

weekly_stats_first_52_weeks = weekly_data.iloc[:52].agg(['min', 'max', 'mean', 'std'])

print("Missing values per location:")
print(missing_values_per_location)
print("\nTotal non-missing values:", total_non_missing_values)
print("\nMean windspeeds of the entire dataset:", mean_windspeed)
print("\nLocation statistics:")
print(loc_stats)
print("\nDay statistics:")
print(day_stats)
print("\nAverage windspeed in January for each location:")
print(january_avg_windspeed)
print("\nYearly frequency data:")
print(yearly_data)
print("\nMonthly frequency data:")
print(monthly_data)
print("\nWeekly frequency data:")
print(weekly_data)
print("\nWeekly statistics for the first 52 weeks:")
print(weekly_stats_first_52_weeks)


# In[ ]:


#Question no 5

import pandas as pd

url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"

chipo = pd.read_csv(url, sep='\t')

print("First 10 entries:\n", chipo.head(10))

print("\nNumber of observations:", len(chipo))

print("\nNumber of columns:", len(chipo.columns))

print("\nColumn names:", chipo.columns.tolist())

print("\nIndex:", chipo.index)

most_ordered_item = chipo.groupby('item_name').sum().sort_values(by='quantity', ascending=False).head(1).index[0]
print("\nMost-ordered item:", most_ordered_item)

most_ordered_item_quantity = chipo.groupby('item_name').sum().loc[most_ordered_item, 'quantity']
print("\nNumber of items ordered for the most-ordered item:", most_ordered_item_quantity)

most_ordered_choice_description = chipo.groupby('choice_description').sum().sort_values(by='quantity', ascending=False).head(1).index[0]
print("\nMost ordered item in the choice_description column:", most_ordered_choice_description)

total_items_ordered = chipo['quantity'].sum()
print("\nTotal items ordered:", total_items_ordered)

chipo['item_price'] = chipo['item_price'].apply(lambda x: float(x[1:]))

print("\nType of item price after conversion:", chipo['item_price'].dtype)

revenue = (chipo['quantity'] * chipo['item_price']).sum()
print("\nRevenue for the period:", revenue)

num_orders = chipo['order_id'].nunique()
print("\nNumber of orders made in the period:", num_orders)

avg_revenue_per_order = revenue / num_orders
print("\nAverage revenue amount per order:", avg_revenue_per_order)

num_unique_items = chipo['item_name'].nunique()
print("\nNumber of different items sold:", num_unique_items)


# In[ ]:


#qn 8
import matplotlib.pyplot as plt

actors = ['John Wick', 'The Terminator', 'Rambo', 'Sarah Connor', 'James Bond']
kill_counts = [300, 200, 150, 120, 100]

sorted_data = sorted(zip(actors, kill_counts), key=lambda x: x[1], reverse=True)
sorted_actors, sorted_kill_counts = zip(*sorted_data)

plt.figure(figsize=(10, 6))
plt.barh(sorted_actors, sorted_kill_counts, color='skyblue')

plt.xlabel('Kill Count')
plt.ylabel('Actor')
plt.title('Deadliest Actors in Hollywood')
plt.xticks(rotation=45) 

for i, (actor, kill_count) in enumerate(zip(sorted_actors, sorted_kill_counts)):
    plt.text(kill_count, i, f' {actor}', va='center', ha='left')

plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# In[ ]:


#qn 9
import matplotlib.pyplot as plt

total_emperors = 50
assassinated_emperors = 12

fraction_assassinated = assassinated_emperors / total_emperors

fraction_survived = 1 - fraction_assassinated

labels = ['Assassinated', 'Survived']

sizes = [fraction_assassinated * 100, fraction_survived * 100]

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)

plt.axis('equal')

plt.title('Fraction of Roman Emperors Assassinated')

plt.show()


# In[ ]:


#Qn10
import matplotlib.pyplot as plt
import numpy as np

years = np.arange(2000, 2010)
revenue_arcades = [1000000, 1100000, 1200000, 1250000, 1300000, 1350000, 1400000, 1450000, 1500000, 1550000]
cs_phds_awarded = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]

plt.figure(figsize=(10, 6))
plt.scatter(revenue_arcades, cs_phds_awarded, c=years, cmap='viridis', alpha=0.8)

cbar = plt.colorbar()
cbar.set_label('Year')

plt.xlabel('Total Revenue Earned by Arcades')
plt.ylabel('Number of Computer Science PhDs Awarded')
plt.title('Relationship between Arcade Revenue and Computer Science PhDs')

plt.grid(True)
plt.show()

