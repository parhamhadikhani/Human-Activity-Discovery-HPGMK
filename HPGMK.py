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


from particle import Particle, cdist_fast,calc_sse

class ParticleSwarmOptimizedClustering:
    def __init__(self,
                 n_cluster: int,
                 n_particles: int,
                 data: np.ndarray,
                 hybrid: bool = False,
                 max_iter: int = 100,
                 print_debug: int = 10):
        self.n_cluster = n_cluster
        self.n_particles = n_particles
        self.data = data
        self.max_iter = max_iter
        self.particles = []
        self.hybrid = hybrid
        self.label=None
        self.tolerance = 1e-10
        self.print_debug = print_debug
        self.gbest_sse = np.inf
        self.gbest_centroids = None
        self._init_particles()

    def _init_particles(self):
        for i in range(self.n_particles):
            particle = None
            if i == 0 and self.hybrid:
                particle = Particle(self.n_cluster, self.data, use_kmeans=False)
            else:
                particle = Particle(self.n_cluster, self.data, use_kmeans=False)
            if particle.best_sse < self.gbest_sse:
                self.gbest_centroids = particle.centroids.copy()
                self.gbest_sse = particle.best_sse
                self.label=particle._predict(self.data).copy()
            self.particles.append(particle)

    def _update_centroid(self, data: np.ndarray, labels: np.ndarray):
        centroids = []
        for i in range(self.n_cluster):
            idx = np.where(labels == i)
            centroid = np.mean(data[idx], axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        return centroids

    def predict(self,data,new_centroid):
        distance = cdist_fast(data,new_centroid)
        cluster = np.argmin(distance, axis=1)
        return cluster

    def run(self):
        newsse=0
        print('Initial global best score', self.gbest_sse)
        history = []
        for i in range(self.max_iter):
            for particle in self.particles:
                particle.update(self.gbest_centroids, self.data)
                particle._update_parameters(i,self.max_iter)
            for particle in self.particles:
                if particle.best_sse < self.gbest_sse:
                    self.gbest_centroids = particle.centroids.copy()
                    self.gbest_sse = particle.best_sse
                    self.label=particle._predict(self.data).copy()
                    
                    #mutation
                    
                    for i in range(10):
                        velocity=particle.velocity*np.exp(np.random.normal(0, particle.sigma))
                        centroids=np.random.normal(0, particle.sigma)*velocity+particle.centroids
                        distance = cdist_fast(self.data,centroids)
                        labelmu = np.argmin(distance, axis=1)
                        sse=calc_sse(centroids, labelmu, self.data)
                        if sse < self.gbest_sse:
                            particle.best_sse = sse
                            particle.velocity = velocity.copy()
                            particle.centroids = centroids.copy()
                            self.gbest_centroids = centroids.copy()
                            self.gbest_sse = sse
                            self.label = labelmu.copy()
                            for i in range(10):
                                centroids=particle.centroids.copy()
                                point=np.random.randint(len(centroids))
                                velocity = particle.velocity*np.exp(np.random.normal(0, particle.sigma))*(np.max(particle.centroids[point,:])-np.min(particle.centroids[point,:]))
                                centroids[point,:]=np.random.normal(0, particle.sigma)*velocity[point,:]+particle.centroids[point,:]
                                distance = cdist_fast(self.data,centroids)
                                labelmu = np.argmin(distance, axis=1)
                                sse=calc_sse(centroids, labelmu, self.data)
                                if sse < self.gbest_sse:
                                    particle.best_sse = sse
                                    particle.velocity = velocity.copy()
                                    particle.centroids = centroids.copy()
                                    self.gbest_centroids = centroids.copy()
                                    self.gbest_sse = sse
                                    self.label = labelmu.copy()
                    
                    #mutation
                                    
            history.append(self.gbest_sse)
            
        

        
        sigma=1
        for _ in range(self.max_iter):
            new_centroid = self._update_centroid(self.data, self.label)
            label=self.predict(self.data,new_centroid)
            gbest_sse = calc_sse(new_centroid, label, self.data)
            if math.isnan(gbest_sse):
                break
            self.label=self.predict(self.data,new_centroid)
            self.gbest_sse = calc_sse(new_centroid, self.label, self.data)
            diff = np.abs(self.gbest_centroids - new_centroid).mean()
            self.gbest_centroids = new_centroid
            history.append(self.gbest_sse)
            if diff <= self.tolerance:
               break
        
        print('Finish with gbest score {:.18f}'.format(self.gbest_sse))
        return history,self.label,self.gbest_sse
