import numpy as np
from tqdm import tqdm


def dist2prob(D):
    row_sums = D.sum(axis=1)
    D_norm = (D / row_sums[:, np.newaxis])
    S = 1 - D_norm
    row_sums = S.sum(axis=1)
    S_norm = (S / row_sums[:, np.newaxis])		   
    return S_norm     


class MahalanobisClassifier():
    def __init__(self, samples, labels):
        self.clusters={}
        for lbl in np.unique(labels):
            # self.clusters[lbl] = samples.loc[labels == lbl, :]
            indices = np.where(labels == lbl)[0]
            self.clusters[lbl] = samples[indices]

    def mahalanobis(self, x, data, cov=None):
        """Compute the Mahalanobis Distance between each row of x and the data  
        x    : vector or matrix of data with, say, p columns.
        data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
        cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
        """
        # print(data.shape)
        # print(x.shape)
        # print(np.mean(data,axis=0).shape)
        diff = x - np.mean(data,axis=0)
        if not cov:
            cov = np.cov(data.T)
        inv_covmat = np.linalg.inv(cov)
        md=diff.dot(inv_covmat).dot(diff.T)


        return np.sqrt(np.diagonal(md))


    def predict_probability(self, unlabeled_samples,ind2label):
        dists = np.array([])

        # print(unlabeled_samples)

	    #Distance of each sample from all clusters
        for lbl in tqdm(self.clusters,desc='calculate', leave=False):
            tmp_dists=self.mahalanobis(unlabeled_samples, self.clusters[lbl])
            if len(dists)!=0:
                dists = np.column_stack((dists, tmp_dists))
            else:
                dists = tmp_dists

        pred_class=np.array([ind2label[np.argmax(row)] for row in dist2prob(dists)])

        return dist2prob(dists),pred_class