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

from sklearn.decomposition import PCA

def dimentaion_reduction(data):
    pca = PCA(n_components=0.85)
    pca.fit(data)
    data = pca.fit_transform(data)
    return data


###KEY FRAME###
def Keyframe(ind,peakst):  
    temp=[]
    sum1=0
    for i in range(len(ind)):
        sum1+=ind[i]
        temp.append(sum1)
    j=0
    labe=[]
    count=0
    for i in range(len(peakst)):
        if (peakst[i]<temp[j]):    
            count+=1
        else:
            labe.append(count)
            count=1
            j=j+1
    labe.append(count)
    ind=labe
    return ind
###KEY FRAME###


#####Smapling#######
def Sampling(data, number_of_frame,ind):
    temp=[]
    sum1=0
    for i in range(len(ind)):
        sum1+=ind[i]
        temp.append(sum1)
    a=[]
    k=0
    i=0
    count=0
    label=[]
    j=i+number_of_frame
    while(len(data)>=j):
        a.append(data[i:j])
        if (temp[k]>=j):
            count+=1
        else:
            if (np.abs((temp[k]-j)/number_of_frame)<0.5):
                count+=1
                label.append(count)
                count=0
                k+=1
            else:
                label.append(count)
                count=1
                k+=1
        i=j-1
        j=i+number_of_frame
    label.append(count)
    a=np.array(a)
    return a,label

#####Smapling#######
