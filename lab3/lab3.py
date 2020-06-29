# %%
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import probplot
from sklearn.linear_model import LinearRegression

# %%
anscombe_path = './anscombe.csv'
anscombe = pd.read_csv(anscombe_path, index_col=0)
anscombe.head()

# %%
anscombe.describe()

# %%
corr_matrix = anscombe.corr()
corr_matrix
# %%
models = []
fig, axs = plt.subplots(2, 2, figsize=(14, 10)) # default figsize is [8.0, 6.0]
axs = axs.flatten()
fig.suptitle('Anscombe Dataset')

for i in range(1, 5):
    # fit linear regression for each data set
    X = anscombe[f"x{i}"].to_numpy()
    y = anscombe[f"y{i}"].to_numpy()
    reg = LinearRegression().fit(X.reshape(-1, 1), y)
    models.append(reg)
    axs[i - 1].scatter(X, y)
    axs[i - 1].plot(X, reg.predict(X.reshape(-1, 1)))
    axs[i - 1].set(xlabel=f"x{i}", ylabel=f"y{i}")
    print(f"Line equation {reg.coef_[0]} * x + {reg.intercept_}")

plot_path = './anscombe_lin_reg.pdf'
# plt.savefig(plot_path, style='dense')
plt.show()

# %%
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()
fig.suptitle('Residuals')

for i in range(1, 5):
    X = anscombe[f"x{i}"].to_numpy()
    y = anscombe[f"y{i}"].to_numpy()
    residuals = (y - models[i - 1].predict(X.reshape(-1, 1)))
    axs[i - 1].hist(residuals)
    axs[i - 1].set(xlabel=f"x{i}", ylabel=f"y{i}")

# plt.savefig('./anscombe_residuals.pdf', style='dense')
plt.show()

# %%

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()
fig.suptitle('Residuals Q-Q Plot')

for i in range(1, 5):
    X = anscombe[f"x{i}"].to_numpy()
    y = anscombe[f"y{i}"].to_numpy()
    residuals = (y - models[i - 1].predict(X.reshape(-1, 1)))
    probplot(residuals, plot=axs[i - 1])
    axs[i - 1].set(title=f"x{i}, y{i}")

# plt.savefig('./anscombe_qq_plot.pdf', style='dense')
plt.show()


# %%
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()
fig.suptitle('Residuals vs Fitted y')

for i in range(1, 5):
    X = anscombe[f"x{i}"].to_numpy()
    y = anscombe[f"y{i}"].to_numpy()
    residuals = (y - models[i - 1].predict(X.reshape(-1, 1)))
    axs[i - 1].scatter(models[i - 1].predict(X.reshape(-1, 1)), residuals)
    axs[i - 1].set(xlabel="y_predicted", ylabel="residuals", title=f"x{i}, y{i}")

# plt.savefig('./anscombe_pred_vs_res.pdf', style='dense')
plt.show()

# %%
# Diamonds dataset
diamonds = pd.read_csv('./diamonds.csv', index_col=0)
diamonds.head()

# %%
cuts = diamonds.cut.unique()
weight = diamonds.carat.to_numpy()
price = diamonds.price.to_numpy()

cut_cmap_values = np.linspace(start=0, stop=1, num=len(cuts))
cut_cmap_dict = {s: v for s, v in zip(cuts, reversed(cut_cmap_values))}

fig = plt.figure(figsize=(14, 10))
plt.title('Diamonds')
plt.xlabel('Weight (Carat)')
plt.ylabel('Price (US Dollar)')
plt.scatter(weight, price, color=cm.get_cmap('viridis')(diamonds.cut.map(cut_cmap_dict)))
# VERY IMPORTANT OUTCOME
plt.colorbar()
plt.show()

# %%
d_corr_matrix = diamonds.corr()
d_corr_matrix

# %%
figsize = (18, 15)
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=figsize)
axs = axs.flatten()
axs[-1].axis('off') # hide axis which won't be used
fig.suptitle('Diamonds')

fig_res_hist, axs_res_hist = plt.subplots(nrows=3, ncols=2, figsize=figsize)
axs_res_hist = axs_res_hist.flatten()
axs_res_hist[-1].axis('off') # hide axis which won't be used
fig_res_hist.suptitle('Diamonds - Residual Histogram')

fig_qq, axs_qq = plt.subplots(nrows=3, ncols=2, figsize=figsize)
axs_qq = axs_qq.flatten()
axs_qq[-1].axis('off') # hide axis which won't be used
fig_qq.suptitle('Diamonds - Residual Q-Q Plot')

fig_h, axs_h = plt.subplots(nrows=3, ncols=2, figsize=figsize)
axs_h = axs_h.flatten()
axs_h[-1].axis('off') # hide axis which won't be used
fig_h.suptitle('Diamonds - Homoskedasticity')

models_d = []

for i, c in enumerate(cuts):
    d_c = diamonds.loc[diamonds.cut == c, :]
    X = d_c.carat.to_numpy().reshape(-1, 1)
    y = d_c.price.to_numpy()
    reg = LinearRegression().fit(X, y)
    models_d.append(reg)
    predictions = reg.predict(X)
    residuals = y - predictions

    axs[i].scatter(d_c.carat, d_c.price)
    axs[i].plot(d_c.carat, predictions, color='red')
    axs[i].set(xlabel='Weight (Carat)', ylabel='Price (US Dollar)', title=c)

    # hist 
    axs_res_hist[i].hist(residuals)
    axs_res_hist[i].set(xlabel='Residual (US Dollar)', ylabel='Count',
        title=c)
    # Q-Q plot
    probplot(residuals, plot=axs_qq[i])
    axs_qq[i].set(title=c)
    # Homoskedasticity
    axs_h[i].scatter(predictions, residuals)
    axs_h[i].set(xlabel='Predicted Price (US Dollar)', ylabel='Residual (US Dollar)',
        title=c)

fig.savefig('./diamonds.png', style='dense')
fig_res_hist.savefig('./diamonds_res_hist.png', style='dense')
fig_qq.savefig('./diamonds_qq.png', style='dense')
fig_h.savefig('./diamonds_homosked.png', style='dense')


# %%
