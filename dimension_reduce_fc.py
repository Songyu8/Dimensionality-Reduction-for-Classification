
def ISOMAP():
    from sklearn.manifold import Isomap
    return Isomap  #data_2 = Isomap(n_neighbors = 10, n_components = 2).fit_transform(X)

def LDA():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis

def LLE():
    from sklearn.manifold import LocallyLinearEmbedding
    return LocallyLinearEmbedding

def MDS():
    from sklearn.manifold import MDS
    return MDS

def PCA():
    from sklearn.decomposition import PCA
    return PCA

def KPCA():
    from sklearn.decomposition import KernelPCA
    return KernelPCA



