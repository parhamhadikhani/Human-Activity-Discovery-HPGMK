#___________________________________________________________________________________#
#   A Novel Skeleton-Based Human Activity Discovery
#   Technique Using Particle Swarm Optimization with
#   Gaussian Mutation
#
#                                                                                   #
#   Author and programmer: Parham Hadikhani, DTC Lai, WH Ong                             #
#                                                                                   #
#   e-Mail:20h8561@ubd.edu.bn, daphne.lai@ubd.edu.bn, weehong.ong@ubd.edu.bn        #   
#___________________________________________________________________________________#


import numpy as np
import math 
import seaborn as sns; sns.set()



# sum squre error
def calc_sse(centroids: np.ndarray, labels: np.ndarray, data: np.ndarray):
    distances = 0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)
        dist = np.sum((data[idx] - c)**2)
        distances += dist
    return distances

# Calculate distance between data and centroids
def _calc_distance(data: np.ndarray,centroids) -> np.ndarray:
    distances = []
    for c in centroids:
        for i in range(len(data)):
            distances.append(np.linalg.norm(data[i,:,:] - c))
    distances = list(_divide_chunks(distances, len(data))) 
    distances = np.array(distances)
    distances = np.transpose(distances)
    return distances


def _divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n]

# Calculate distance between data and centroids       
def cdist_fast(XA, XB):

    XA_norm = np.sum(XA**2, axis=1)
    XB_norm = np.sum(XB**2, axis=1)
    XA_XB_T = np.dot(XA, XB.T)
    distances = XA_norm.reshape(-1,1) + XB_norm - 2*XA_XB_T
    return distances  
#'''
class Particle:

    def __init__(self,
                 n_cluster: int,
                 data: np.ndarray,
                 use_kmeans: bool = False,
                 wmax: float = 0.9,
                 c1max: float = 2.5,
                 c2max: float = 0,
                 wmin: float = 0.4,
                 c1min: float = 0,
                 c2min: float = 2.5):

        index = np.random.choice(list(range(len(data))), n_cluster)
        self.centroids = data[index].copy()
        self.best_position = self.centroids.copy()
        self.best_sse = calc_sse(self.centroids, self._predict(data), data)
        self.velocity = np.zeros_like(self.centroids)
        self._w = wmax
        self._c1 = c1max
        self._c2 = c2min
        self._wmax = wmax
        self._c1max = c1max
        self._c2max = c2max
        self._wmin = wmin
        self._c1min = c1min
        self._c2min = c2min
        self.sigma=1 

    def _update_parameters(self,t,tmax):
        self._c1=self._c1max-(self._c1max-self._c1min)*(t/tmax)
        self._c2=(self._c2min-self._c2max)*(t/tmax)+self._c2max
        self._w=self._wmax-(t*(self._wmax-self._wmin))/tmax
        if t!=0:
          self.sigma= self.sigma-(1/(tmax))
          
    #Update particle's velocity and centroids
    def update(self, gbest_position: np.ndarray, data: np.ndarray):
        self._update_velocity(gbest_position)
        self._update_centroids(data)

    def _update_velocity(self, gbest_position: np.ndarray):
        v_old = self._w * self.velocity
        cognitive_component = 2 * np.random.random() * (self.best_position - self.centroids)
        social_component = 2 * np.random.random() * (gbest_position - self.centroids)
        self.velocity = v_old + cognitive_component + social_component

    def _update_centroids(self, data: np.ndarray):
        self.centroids = self.centroids + self.velocity
        new_score = calc_sse(self.centroids, self._predict(data), data)
        sse = calc_sse(self.centroids, self._predict(data), data)
        if new_score < self.best_sse:
            self.best_sse = new_score
            self.best_position = self.centroids.copy()
            
    #Predict new data's cluster using minimum distance to centroid
    def _predict(self, data: np.ndarray) -> np.ndarray:
        distance = cdist_fast(data,self.centroids)
        #distance = _calc_distance(data,self.centroids)
        cluster = self._assign_cluster(distance)
        return cluster

    #Assign cluster to data based on minimum distance to centroids
    def _assign_cluster(self, distance: np.ndarray) -> np.ndarray:
        cluster = np.argmin(distance, axis=1)
        return cluster
        
