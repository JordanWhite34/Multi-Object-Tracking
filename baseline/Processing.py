from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class postprocess:
    def __init__(self, number_of_people, cluster_method, scale_method):
        self.n = number_of_people
        if cluster_method == 'kmeans':
            self.cluster_method = KMeans(n_clusters=self.n, random_state=0)
        else:
            raise NotImplementedError

        if scale_method == 'standard':
            self.scaler = StandardScaler()
        elif scale_method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif scale_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise NotImplementedError
            
    def run(self, features):
        print('Start Clustering')

        # Scale or normalize the features
        scaled_features = self.scaler.fit_transform(features)

        # Fit the clustering algorithm
        self.cluster_method.fit(scaled_features)

        print('Finish Clustering')

        return self.cluster_method.labels_
