#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %matplotlib inline


film_deaths = pd.read_csv('./filmdeathcounts.csv')

film_deaths['body_per_min'] = film_deaths.Body_Count / film_deaths.Length_Minutes

# %%
plt.plot()
film_deaths.Body_Count.hist(bins=20)
plt.xlabel('Body Count')
plt.ylabel('Film Count')
plt.show()

# %%
print("Top 10 films, where the biggest number of characters were killed")
film_deaths.sort_values(by='Body_Count', ascending=False).loc[:10, :]

# %%
print("Top 10 films, whith highest body per minute values")
film_deaths.sort_values(by='body_per_min', ascending=False).loc[:10, :]

# %%
plt.plot()
film_deaths.IMDB_Rating.hist(bins=10)
plt.xlabel('IMBD_Rating')
plt.ylabel('Films Count')
plt.show()

# %%
imdb_mean = film_deaths.IMDB_Rating.mean()
imdb_std = film_deaths.IMDB_Rating.std()

print(f'IMDB mean = {imdb_mean:.3}, std = {imdb_std:.3}')

# %%
np.set_seed(42)
film_deaths['imdb_simulation'] = np.random.normal(loc=imdb_mean, scale=imdb_std,
    size=len(film_deaths))

plt.plot()

film_deaths.imdb_simulation.hist(bins=10)

plt.xlabel('IMDB_Rating (Simulation)')
plt.ylabel('Films Count (Simulation)')
plt.show()

# %%
from scipy.stats import probplot

axs1 = plt.subplot()
probplot(film_deaths['imdb_simulation'], plot=axs1)
axs1.set_title('IMDB_Rating (Simulation)')
plt.show()

axs2 = plt.subplot()
probplot(film_deaths['IMDB_Rating'], plot=axs2)
axs2.set_title('IMDB_Rating')

plt.show()


# %%
