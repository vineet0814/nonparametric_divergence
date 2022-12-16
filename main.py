import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
import pandas as pd
import numpy as np
from utils import *

plt.figure(figsize=(10, 10))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
n_samples = 100
n_features = 2 


plt.subplot(421)
X2, Y2 = make_classification(n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1, random_state = 1)
X2_0, X2_1 = split_data(X2, Y2)
output_list, d_bar = compute_dp_hat(X2_0, X2_1, n_samples, n_features)
pars, cov = curve_fit(f=power_law, xdata=output_list, ydata=d_bar, p0=[0, 0, 0], bounds=(-np.inf, np.inf))
plt.title(f'One informative feature, one cluster per class \n Class overlap:{pars[-1]: .2f}', fontsize="small")
plt.scatter(X2[:, 0], X2[:, 1], marker="o", c=Y2, s=25, edgecolor="k")

plt.subplot(422)
X2, Y2 = make_classification(
    n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state = 5
)
X2_0, X2_1 = split_data(X2, Y2)
output_list, d_bar = compute_dp_hat(X2_0, X2_1, n_samples, n_features)
pars, cov = curve_fit(f=power_law, xdata=output_list, ydata=d_bar, p0=[0, 0, 0], bounds=(-np.inf, np.inf))
plt.title(f"Two informative features, one cluster per class \n Class overlap: {pars[-1]: .2f}", fontsize="small")
plt.scatter(X2[:, 0], X2[:, 1], marker="o", c=Y2, s=25, edgecolor="k")

plt.subplot(423)
X2, Y2 = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state = 3)
X2_0, X2_1 = split_data(X2, Y2)
output_list, d_bar = compute_dp_hat(X2_0, X2_1, n_samples, n_features)
pars, cov = curve_fit(f=power_law, xdata=output_list, ydata=d_bar, p0=[0, 0, 0], bounds=(-np.inf, np.inf))
plt.title(f"Two informative features, two clusters per class \n Class overlap: {pars[-1]: .2f}", fontsize="small")
plt.scatter(X2[:, 0], X2[:, 1], marker="o", c=Y2, s=25, edgecolor="k")

plt.subplot(424)
X2, Y2 = make_classification(
    n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=2, random_state = 4)
X2_0, X2_1 = split_data(X2, Y2)
output_list, d_bar = compute_dp_hat(X2_0, X2_1, n_samples, n_features)
pars, cov = curve_fit(f=power_law, xdata=output_list, ydata=d_bar, p0=[0, 0, 0], bounds=(-np.inf, np.inf))
plt.title(f"Multi-class, two informative features, one cluster \n Class overlap: {pars[-1]: .2f}", fontsize="small")
plt.scatter(X2[:, 0], X2[:, 1], marker="o", c=Y2, s=25, edgecolor="k")

plt.subplot(425)
X2, Y2 = make_blobs(n_features=2, centers=2, random_state = 2)
X2_0, X2_1 = split_data(X2, Y2)
output_list, d_bar = compute_dp_hat(X2_0, X2_1, n_samples, n_features)
pars, cov = curve_fit(f=power_law, xdata=output_list, ydata=d_bar, p0=[0, 0, 0], bounds=(-np.inf, np.inf))
plt.title(f"Two blobs \n Class overlap: {pars[-1]: .2f}", fontsize="small")
plt.scatter(X2[:, 0], X2[:, 1], marker="o", c=Y2, s=25, edgecolor="k")

plt.subplot(426)
X2, Y2 = make_gaussian_quantiles(n_features=2, n_classes=2, random_state = 1)
X2_0, X2_1 = split_data(X2, Y2)
output_list, d_bar = compute_dp_hat(X2_0, X2_1, n_samples, n_features)
pars, cov = curve_fit(f=power_law, xdata=output_list, ydata=d_bar, p0=[0, 0, 0], bounds=(-np.inf, np.inf))
plt.title(f"Gaussian divided into two quantiles \n Class overlap: {pars[-1]: .2f}", fontsize="small")
plt.scatter(X2[:, 0], X2[:, 1], marker="o", c=Y2, s=25, edgecolor="k")



plt.subplot(427)
X2, Y2 = make_circles(n_samples=100, noise=0.15, random_state= 1)
X2_0, X2_1 = split_data(X2, Y2)
output_list, d_bar = compute_dp_hat(X2_0, X2_1, n_samples, n_features)
pars, cov = curve_fit(f=power_law, xdata=output_list, ydata=d_bar, p0=[0, 0, 0], bounds=(-np.inf, np.inf))
plt.title(f"Circle \n Class overlap: {pars[-1]: .2f}", fontsize="small")
plt.scatter(X2[:, 0], X2[:, 1], marker="o", c=Y2, s=25, edgecolor="k")

plt.subplot(428)
X2, Y2 = make_moons(n_samples=100, noise=0.3, random_state=0)
X2_0, X2_1 = split_data(X2, Y2)
output_list, d_bar = compute_dp_hat(X2_0, X2_1, n_samples, n_features)
pars, cov = curve_fit(f=power_law, xdata=output_list, ydata=d_bar, p0=[0, 0, 0], bounds=(-np.inf, np.inf))
plt.title(f"Half moon \n Class overlap: {pars[-1]: .2f}", fontsize="small")
plt.scatter(X2[:, 0], X2[:, 1], marker="o", c=Y2, s=25, edgecolor="k")

plt.show()
