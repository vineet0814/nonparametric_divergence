
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix


def compute_neighbors(A, B, k=1, algorithm='auto'):
    '''
    For each sample in A, compute the nearest neighbor in B
    :inputs:
    A and B - both (n_samples x n_features)
    algorithm - look @ scipy NearestNeighbors nocumentation for this (ball_tre or kd_tree)
                dont use kd_tree if n_features>~20 or 30
    :return:
    a list of the closest points to each point in A and B
    '''
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm=algorithm).fit(B)
    nns = nbrs.kneighbors(A)[1]
    nns = nns[:, 1]

    # exit()
    return nns



def dp_div(A, B, method='1nn'):
    '''
    Requires A and B to be the same number of dimensions
    *******
    WARNING!!!
    MST is very slow in this implementation, this is unlike matlab where they ahve comparable speeds
    Overall, MST takes ~25x LONGER!!
    Source of slowdown:
    conversion to and from CSR format adds ~10% of the time diff between 1nn and scipy mst function the remaining 90%
    *******
    '''

    data = np.vstack([A, B])
    N = A.shape[0]
    M = B.shape[0]
    labels = np.vstack([np.zeros([N, 1]), np.ones([M, 1])])

    if method == '1nn':
        nn_indices = compute_neighbors(data, data)
        # import pdb
        # pdb.set_trace()
        errors = np.sum(np.abs(labels[nn_indices] - labels))
        # print('Errors '+str(errors))
        Dp = 1 - ((M + N) / (2 * M * N)) * errors

    # TODO: improve speed for MST, requires a fast mst implementation
    # mst is at least 10x slower than knn approach
    elif method == 'mst':
        dense_eudist = squareform(pdist(data))
        eudist_csr = csr_matrix(dense_eudist)
        mst = minimum_spanning_tree(eudist_csr)
        mst = mst.toarray()
        edgelist = np.transpose(np.nonzero(mst))

        errors = np.sum(labels[edgelist[:, 0]] != labels[edgelist[:, 1]])

        Dp = 1 - ((M + N) / (2 * M * N)) * errors
    # Dp=1
    # errors=0
    Cij = errors

    return Dp, Cij

def compute_dp_hat(X2_0, X2_1, n_samples, dim):
  subsample_size = []
  L = 10
  N = n_samples
  d = dim
  y = np.logspace(d+1, N//2, num= L, base=2 )
  #print(y)
  for i in range(len(y)): 
    subsample_size.append(y[i])
  for i in range(len(y)): 
    yy =  np.logspace(d+1, np.log2(y[i]), num=2*L, base = 2) 
    for j in range(len(yy)): 
      subsample_size.append(yy[j])
  res = [*set(subsample_size)]
  output_list = [int(np.log2(i)) for i in res]    
  output_list.sort()
  output_list = [*set(output_list)]
  output_list = list(filter(lambda x: x <= min(len(X2_0), len(X2_1)), output_list))
  alpha = 75
  M = [int(75*N/i) for i in output_list] 
  d_bar = []
  for j in range(len(output_list)): #J
    d_hat = []
    for k in range(M[j]): 
      x_0 = X2_0[np.random.choice(X2_0.shape[0], output_list[j], replace=False), :]
      x_1 = X2_1[np.random.choice(X2_1.shape[0], output_list[j], replace=False), :]
      #P,  reject, S = FR(x_0, x_1)
      #d_hat.append(1 - (S/output_list[j]))
      d_p, c_ij = dp_div(x_0, x_1, method = '1nn')
      d_hat.append(d_p)
    d_bar.append(sum(d_hat)/M[j])
  return output_list, d_bardef compute_dp_hat(X2_0, X2_1, n_samples, dim):
  subsample_size = []
  L = 10
  N = n_samples
  d = dim
  y = np.logspace(d+1, N//2, num= L, base=2 )
  #print(y)
  for i in range(len(y)): 
    subsample_size.append(y[i])
  for i in range(len(y)): 
    yy =  np.logspace(d+1, np.log2(y[i]), num=2*L, base = 2) 
    for j in range(len(yy)): 
      subsample_size.append(yy[j])
  res = [*set(subsample_size)]
  output_list = [int(np.log2(i)) for i in res]    
  output_list.sort()
  output_list = [*set(output_list)]
  output_list = list(filter(lambda x: x <= min(len(X2_0), len(X2_1)), output_list))
  alpha = 75
  M = [int(75*N/i) for i in output_list] 
  d_bar = []
  for j in range(len(output_list)): #J
    d_hat = []
    for k in range(M[j]): 
      x_0 = X2_0[np.random.choice(X2_0.shape[0], output_list[j], replace=False), :]
      x_1 = X2_1[np.random.choice(X2_1.shape[0], output_list[j], replace=False), :]
      #P,  reject, S = FR(x_0, x_1)
      #d_hat.append(1 - (S/output_list[j]))
      d_p, c_ij = dp_div(x_0, x_1, method = '1nn')
      d_hat.append(d_p)
    d_bar.append(sum(d_hat)/M[j])
  return output_list, d_bar


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
X2, Y2 = make_classification(
    n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1, random_state = 1
)
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
