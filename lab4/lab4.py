# %%
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
crimes = pd.read_csv('./crimes.csv')
crimes['Date'] = pd.Series((np.datetime64(d[:10]) for d in crimes.Dates),
index=crimes.index)
print(crimes.dtypes)
crimes.head()

# %%
moon = pd.read_csv('./moon.csv')
moon.date = pd.to_datetime(moon.date)
print(moon.dtypes)
moon.head()
# %%
# merge two tables
crimes_moon = pd.merge(crimes, moon, left_on='Date', right_on='date')
print(crimes_moon.shape)
crimes_moon.head()

# %%
crimes_moon['count'] = pd.Series(1, index=crimes_moon.index)
crimes_moon_grouped = crimes_moon.groupby(['Date', 'phase']).count()
crimes_moon_grouped = crimes_moon_grouped.loc[:, ['count']]

# %%
from matplotlib.dates import date2num

plt.figure(figsize=(14, 10))
# dates = list(date2num(d.date()) for d in crimes_moon_grouped.index.get_level_values(0))
dates_t = list(d.date() for d in crimes_moon_grouped.index.get_level_values(0))
colors = list('red' if phase == 'Full Moon' else 'blue' for phase in crimes_moon_grouped.index.get_level_values(1))
plt.scatter(dates_t, crimes_moon_grouped['count'], c=colors)
plt.plot(dates_t, crimes_moon_grouped['count'], color='grey')
plt.title('Crimes in San-Francisco')
plt.xlabel('Date')
plt.ylabel('Crimes Count')
plt.grid()
plt.show()

# %%
crimes_moon_grouped.reset_index(inplace=True)

# %%
from math import sqrt

from scipy.stats import t

mean_crimes_fm = crimes_moon_grouped.loc[crimes_moon_grouped['phase'] == 'Full Moon', 'count'].mean()
mean_crimes_other = crimes_moon_grouped.loc[crimes_moon_grouped['phase'] != 'Full Moon', 'count'].mean()
n_fm = sum(crimes_moon_grouped['phase'] == 'Full Moon')
std_fm = crimes_moon_grouped.loc[crimes_moon_grouped['phase'] == 'Full Moon', 'count'].std()

alpha = 0.05
t_value = (mean_crimes_fm - mean_crimes_other) / (std_fm / sqrt(n_fm))
p_value = 2* (1 - t.cdf(t_value, df=n_fm - 1))

print(f'Mean crimes during Full Moon {mean_crimes_fm:.4f} +/- {std_fm:.4f}')
print(f'Mean crimes in other time {mean_crimes_other:.4f}')
print(f'Test statistic {t_value:.4f}, p-value {p_value:.4f}, level of significance {alpha:.4f}')

if p_value < alpha:
    print('H0 should be rejected.')
else:
    print('H0 can\'t be rejected')

# %%
# Test impulsive crimes
implusive_crimes = ("OTHER OFFENSES", "LARCENY/THEFT", "VANDALISM", "DRUNKENNESS",
"DRUG/NARCOTIC", "DRIVING UNDER THE INFLUENCE", "SEX OFFENSES FORCIBLE", "RUNAWAY",
"DISORDERLY CONDUCT", "ARSON", "SUICIDE", "SEX OFFENSES NON FORCIBLE", 
"SUSPICIOUS OCC", "ASSAULT", "LIQUOR LAWS", "ROBBERY", "BURGLARY", "VEHICLE THEFT")

crimes_moon['count'] = pd.Series(1, index=crimes_moon.index)
# select impulsive crimes
crimes_moon['impulsive'] = pd.Series((True if c in implusive_crimes else False for c in crimes_moon.Category), index=crimes_moon.index)

# %%
crimes_moon = crimes_moon.loc[crimes_moon.impulsive, :]
crimes_moon_grouped = crimes_moon.groupby(['Date', 'phase']).count()
crimes_moon_grouped = crimes_moon_grouped.loc[:, ['count']]

crimes_moon_grouped.reset_index(inplace=True)
crimes_moon_grouped.head()

mean_crimes_fm = crimes_moon_grouped.loc[crimes_moon_grouped['phase'] == 'Full Moon', 'count'].mean()
mean_crimes_other = crimes_moon_grouped.loc[crimes_moon_grouped['phase'] != 'Full Moon', 'count'].mean()
n_fm = sum(crimes_moon_grouped['phase'] == 'Full Moon')
std_fm = crimes_moon_grouped.loc[crimes_moon_grouped['phase'] == 'Full Moon', 'count'].std()

alpha = 0.05
t_value = (mean_crimes_fm - mean_crimes_other) / (std_fm / sqrt(n_fm))
p_value = 2* (1 - t.cdf(t_value, df=n_fm - 1))

print(f'Mean crimes during Full Moon {mean_crimes_fm:.4f} +/- {std_fm:.4f}')
print(f'Mean crimes in other time {mean_crimes_other:.4f}')
print(f'Test statistic {t_value:.4f}, p-value {p_value:.4f}, level of significance {alpha:.4f}')

if p_value < alpha:
    print('H0 should be rejected.')
else:
    print('H0 can\'t be rejected')

# %%
# Check whether day of the week affects number of crimes
crimes_moon['count'] = pd.Series(1, index=crimes_moon.index)
crimes_moon_grouped = crimes_moon.groupby(['DayOfWeek']).count()
crimes_moon_grouped = crimes_moon_grouped.loc[:, ['count']]

# %%
labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
y = [crimes_moon_grouped.loc[l, 'count'] for l in labels]
x = np.arange(len(labels))  # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, y, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('N')
ax.set_xlabel('Day of Week')
ax.set_xticks(x)
ax.set_xticklabels(labels)
fig.tight_layout()
plt.show()


# %%
# hypothesis testing for Friday
crimes_moon_grouped = crimes_moon.groupby(['Date', 'DayOfWeek']).count()
crimes_moon_grouped = crimes_moon_grouped.loc[:, ['count']]
crimes_moon_grouped.reset_index(inplace=True)
crimes_moon_grouped.head()

# %%
mean_crimes_day = crimes_moon_grouped.loc[crimes_moon_grouped['DayOfWeek'] == 'Friday', 'count'].mean()
mean_crimes_other = crimes_moon_grouped.loc[:, 'count'].mean()
n_day = sum(crimes_moon_grouped['DayOfWeek'] == 'Friday')
std_day = crimes_moon_grouped.loc[crimes_moon_grouped['DayOfWeek'] == 'Friday', 'count'].std()

alpha = 0.05
t_value = (mean_crimes_day - mean_crimes_other) / (std_day / sqrt(n_day))
p_value = 2* (1 - t.cdf(t_value, df=n_day - 1))

Z_score = 2.704 # for the level of significance of 0.01

print(f'Mean crimes during Friday {mean_crimes_day:.4f} +/- {std_day:.4f}')
print(f'Mean crimes in other days of week {mean_crimes_other:.4f}')
print(f'Test statistic {t_value:.4f}, p-value {p_value:.4f}, level of significance {alpha:.4f}')
print(f'95% confidence interval [{mean_crimes_day - Z_score * std_day / sqrt(n_day)}, {mean_crimes_day + Z_score * std_day / sqrt(n_day)}]')

if p_value < alpha:
    print('H0 should be rejected.')
else:
    print('H0 can\'t be rejected')

# %%
