from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class PostProcess:
    """
    This class performs post-processing on features to cluster them based on the specified method.
    It supports scaling the features using different scaling methods before clustering.

    Parameters:
    - number_of_people (int): The desired number of clusters or people.
    - cluster_method (str): The clustering method to use. Currently, only 'kmeans' is supported.
    - scale_method (str): The scaling method to use. Available options are 'standard', 'minmax', and 'robust'.

    Methods:
    - run(features): Performs clustering on the given features and returns the cluster labels.
    """

    def __init__(self, number_of_people, cluster_method, scale_method):
        self.n = number_of_people

        if cluster_method == 'kmeans':
            self.cluster_method = KMeans(n_clusters=self.n, random_state=0)
        else:
            raise NotImplementedError("Unsupported clustering method. Only 'kmeans' is currently supported.")

        if scale_method == 'standard':
            self.scaler = StandardScaler()
        elif scale_method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif scale_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise NotImplementedError("Unsupported scaling method. Available options are 'standard', 'minmax', and 'robust'.")

    def run(self, features):
        """
        Performs clustering on the given features.

        Parameters:
        - features (array-like): The input features to be clustered.

        Returns:
        - cluster_labels (array): The cluster labels assigned to each feature.
        """

        print('Start Clustering')

        # Scale or normalize the features
        scaled_features = self.scaler.fit_transform(features)

        # Fit the clustering algorithm
        self.cluster_method.fit(scaled_features)

        print('Finish Clustering')

        return self.cluster_method.labels_

