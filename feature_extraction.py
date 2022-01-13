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
import statistics
from numba import jit, cuda 
import math 
from scipy.signal import find_peaks
from scipy.stats import mode
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import math as m


def Kinetic_energy(data):
    delta_T=1
    energy_of_joint=np.zeros(len(data)-1)
    for i in range(1,len(data)):
        for j in range(len(data[0,:,0])):
            energy_of_joint[i-1]+=0.5 *pow((np.linalg.norm(data[i,j,:]-data[i-1,j,:])/ delta_T),2)
    peaks, _ = find_peaks(energy_of_joint)
    peaks1, _ = find_peaks(-1*energy_of_joint)

    peakst=np.concatenate((peaks, peaks1), axis=0)
    peakst.sort()
    return peakst,energy_of_joint


def P_Joints(coordinates):
    Head=coordinates[:,0,:]
    LHand=coordinates[:,11,:]
    RHand=coordinates[:,12,:]
    LFoot=coordinates[:,13,:]
    RFoot=coordinates[:,14,:]
    LHip=coordinates[:,7,:]
    RHip=coordinates[:,9,:]
    LShoulder=coordinates[:,3,:]
    RShoulder=coordinates[:,5,:]
    Neck=coordinates[:,1,:]
    Torso=coordinates[:,2,:]
    LElbow=coordinates[:,4,:]
    RElbow=coordinates[:,6,:]
    LKnee=coordinates[:,8,:]
    RKnee=coordinates[:,10,:]
    
    return Head,LHand,RHand,LFoot,RFoot,LHip,RHip,LShoulder,RShoulder,Neck,Torso,LElbow,RElbow,LKnee,RKnee

def spatial_Displacement(x,y):
    distance=np.linalg.norm(x-y)
    return distance

def LHand_Head(x,y):
    features=np.zeros((len(x), 1))
    for i in range(len(x)):
        features[i,0]=spatial_Displacement(x[i,:],y[i,:])
    return features

def RHand_Head(x,y):
    features=np.zeros((len(x), 1))
    for i in range(len(x)):
        features[i,0]=spatial_Displacement(x[i,:],y[i,:])
    return features

    
def LHand_RHand(x,y):
    features=np.zeros((len(x), 1))
    for i in range(len(x)):
        features[i,0]=spatial_Displacement(x[i,:],y[i,:])
    return features

def RHand_RFoot(x,y):
    features=np.zeros((len(x), 1))
    for i in range(len(x)):
        features[i,0]=spatial_Displacement(x[i,:],y[i,:])
    return features

def LHand_LFoot(x,y):
    features=np.zeros((len(x), 1))
    for i in range(len(x)):
        features[i,0]=spatial_Displacement(x[i,:],y[i,:])
    return features

def RShoulder_RFoot(x,y):
    features=np.zeros((len(x), 1))
    for i in range(len(x)):
        features[i,0]=spatial_Displacement(x[i,:],y[i,:])
    return features

    
def LShoulder_LFoot(x,y):
    features=np.zeros((len(x), 1))
    for i in range(len(x)):
        features[i,0]=spatial_Displacement(x[i,:],y[i,:])
    return features


def LHip_LFoot(x,y):
    features=np.zeros((len(x), 1))
    for i in range(len(x)):
        features[i,0]=spatial_Displacement(x[i,:],y[i,:])
    return features

def RHip_RFoot(x,y):
    features=np.zeros((len(x), 1))
    for i in range(len(x)):
        features[i,0]=spatial_Displacement(x[i,:],y[i,:])
    return features

def Temporal_joint_tci(data):
    features=np.zeros((len(data), len(data[0,:])))   
    for f in range(1,len(data)):
        for j in range(len(data[0,:])):
            features[f,j]=((data[f,j]-data[0,j]))#/np.linalg.norm(d[f,i,:]-d[0,i,:]))
    return features
       
def Temporal_joint_tcp(data):
    features=np.zeros((len(data), len(data[0,:])))   
    for f in range(1,len(data)):
        for j in range(len(data[0,:])):
            features[f,j]=((data[f,j]-data[f-1,j]))#/np.linalg.norm(d[f,i,:]-d[0,i,:]))
    return features
#-------------------------------------------------------


#------------------------------------------------------------

#-------------------------------------------------------
#Statistical features

def Mean_difference(joint_selected):
    mean=np.zeros((1,36))
    for i in range(len(joint_selected)):
        mean+=joint_selected[i,:]
    mean=mean/len(joint_selected)
    return mean
    
def Joint_coordinate_mean_difference(joint_selected):
    mean=Mean_difference(joint_selected)
    features=np.zeros((len(joint_selected), 36))
    for i in range(len(joint_selected)): 
        features[i,:]=joint_selected[i]-mean
    return features
        
def Joint_coordinate_variance_difference(joint_selected):
    mean=Mean_difference(joint_selected)
    features=np.zeros((len(joint_selected), 36))
    for i in range(len(joint_selected)): 
        features[i,:]=joint_selected[i]-(((joint_selected[i]-mean)** 2)/len(joint_selected))
    return features

def Joint_coordinate_standard_deviation_difference(joint_selected):
    mean=Mean_difference(joint_selected)
    features=np.zeros((len(joint_selected), 36))
    for i in range(len(joint_selected)): 
        features[i,:]=joint_selected[i]-(np.sqrt(((joint_selected[i]-mean)** 2)/len(joint_selected)))
    return features

def Joint_coordinate_skewness_difference(joint_selected):
    mean=Mean_difference(joint_selected)
    features=np.zeros((len(joint_selected), 36))
    for i in range(len(joint_selected)):
        features[i,:]=joint_selected[i]-((joint_selected[i]-mean)** 3)/(len(joint_selected)-1)*(((np.sqrt(((joint_selected[i]-mean)** 2)/len(joint_selected))))**3)
    return features

def Joint_coordinate_kurtosis_difference(joint_selected):
    mean=Mean_difference(joint_selected)
    features=np.zeros((len(joint_selected), 36))
    for i in range(len(joint_selected)):
        features[i,:]=joint_selected[i]-((joint_selected[i]-mean)** 4)/(len(joint_selected)-1)*(((np.sqrt(((joint_selected[i]-mean)** 2)/len(joint_selected))))**4)
    return features


#------------------------------------------------------------
#angle1

def projection(vector1,vector2):
    a_vec = vector1/np.linalg.norm(vector1)
    b_vec = vector2/np.linalg.norm(vector2)
    cross =  np.linalg.norm(np.cross(a_vec, b_vec))
    return cross

def rejection(vector1,vector2):
    a_vec = vector1/np.linalg.norm(vector1)
    b_vec = vector2/np.linalg.norm(vector2)
    ab_angle = np.arccos(np.dot(a_vec,b_vec))
    return ab_angle

def angle_LHand(LHand,LElbow,LShoulder):
    vector1=LHand-LElbow
    vector2=LElbow-LShoulder
    features=np.zeros((len(vector1), 1))
    for i in range(len(vector1)):    
        features[i,0]=(m.atan2(projection(vector1[i],vector2[i]), rejection(vector1[i],vector2[i])) * 180 / m.pi)+180


    return features

    
def angle_RHand(RHand,RElbow,RShoulder):
    vector1=RHand-RElbow
    vector2=RElbow-RShoulder
    features=np.zeros((len(vector1), 1))
    for i in range(len(vector1)):    
        features[i,0]=(m.atan2(projection(vector1[i],vector2[i]), rejection(vector1[i],vector2[i])) * 180 / m.pi)+180

    return features

def angle_LFoot(LFoot,LKnee,LHip):
    vector1=LFoot-LKnee
    vector2=LKnee-LHip
    features=np.zeros((len(vector1), 1))
    for i in range(len(vector1)):    
        features[i,0]=(m.atan2(projection(vector1[i],vector2[i]), rejection(vector1[i],vector2[i])) * 180 / m.pi)+180
    return features


    
def angle_RFoot(RFoot,RKnee,RHip):
    vector1=RFoot-RKnee
    vector2=RKnee-RHip
    features=np.zeros((len(vector1), 1))
    for i in range(len(vector1)):    
        features[i,0]=(m.atan2(projection(vector1[i],vector2[i]), rejection(vector1[i],vector2[i])) * 180 / m.pi)+180
    return features
#-------------------------------------------------------





#-------------------------------------------------------


#Orientaion
def LHip_LKnee__LKnee_LFoot(LHip,LKnee,LFoot):#1
    vector=LHip-LKnee
    vector2=LKnee-LFoot
    features=np.zeros((len(vector), 3))
    for i in range(len(vector)):
        features[i,:]=Orientation(vector[i],vector2[i])
    return features


def RHip_RKnee__RKnee_RFoot(RHip,RKnee,RFoot):#2
    vector=RHip-RKnee
    vector2=RKnee-RFoot
    features=np.zeros((len(vector), 3))
    for i in range(len(vector)):
        features[i,:]=Orientation(vector[i],vector2[i])
    return features

def LHand_LElbow__LKnee_LFoot(LHand,LElbow,LKnee,LFoot):#3
    vector=LElbow-LHand
    vector2=LKnee-LFoot
    features=np.zeros((len(vector), 3))
    for i in range(len(vector)):
        features[i,:]=Orientation(vector[i],vector2[i])
    return features

def RHand_RElbow__RKnee_RFoot(RHand,RElbow,RKnee,RFoot):#4
    vector=RElbow-RHand
    vector2=RKnee-RFoot
    features=np.zeros((len(vector), 3))
    for i in range(len(vector)):
        features[i,:]=Orientation(vector[i],vector2[i])
    return features

def Head_Neck__RHand_RElbow(Head,Neck,RElbow,RHand):#5
    vector=Head-Neck
    vector2=RElbow-RHand
    features=np.zeros((len(vector), 3))
    for i in range(len(vector)):
        features[i,:]=Orientation(vector[i],vector2[i])
    return features

def Head_Neck__LHand_LElbow(Head,Neck,LElbow,LHand):#6
    vector=Head-Neck
    vector2=LElbow-LHand
    features=np.zeros((len(vector), 3))
    for i in range(len(vector)):
        features[i,:]=Orientation(vector[i],vector2[i])
    return features


def LHand_LElbow__RHand_RElbow(LHand,LElbow,RElbow,RHand):#7
    vector=LElbow-LHand
    vector2=RElbow-RHand
    features=np.zeros((len(vector), 3))
    for i in range(len(vector)):
        features[i,:]=Orientation(vector[i],vector2[i])
    return features


def LElbow_LShoulder__LHand_LElbow(LElbow,LShoulder,LHand):#8
    vector=LShoulder-LElbow
    vector2=LElbow-LHand
    features=np.zeros((len(vector), 3))
    for i in range(len(vector)):
        features[i,:]=Orientation(vector[i],vector2[i])
    return features

def RElbow_RShoulder__RHand_RElbow(RElbow,RShoulder,RHand):#9
    vector=RShoulder-RElbow
    vector2=RElbow-RHand
    features=np.zeros((len(vector), 3))
    for i in range(len(vector)):
        features[i,:]=Orientation(vector[i],vector2[i])
    return features

def Orientation(vector1,vector2):
    a_vec = vector1/np.linalg.norm(vector1)
    b_vec = vector2/np.linalg.norm(vector2)
    
    
    norm=np.linalg.norm(vector1)*np.linalg.norm(vector2)
    C=vector1[2]*vector2[0]-vector1[0]*vector2[2]
    A=vector1[0]*vector2[1]-vector1[1]*vector2[0]
    B=vector1[1]*vector2[2]-vector1[2]*vector2[1]
    
    b=B/norm
    a=A/norm
    c=C/norm
    n2=np.sqrt(a*a+b*b+c*c)
    a=a/n2
    b=b/n2
    c=c/n2    
    x = np.arccos(np.dot(a_vec,b_vec))
    y=np.arccos(c)
    z=np.arcsin(b/np.sin(y))
    return x,y,z




#-------------------------------------------------------

@jit
def Feature_normalisation(features):
    print('Feature_normalisation')

    f_norm=np.zeros((len(features),len(features[0,:])))
    for i in range(len(features)):
        print(i)
        for j in range(len(features[0,:])):
            f_norm[i,j]=(features[i,j]-min(features[:,j]))/(max(features[:,j])-min(features[:,j]))
    return f_norm  

def Extraction(data):
    keyframes,energy=Kinetic_energy(data)
    d=[]
    for i in range(len(keyframes)):
      d.append(data[keyframes[i]])
    d1=np.array(d)
    data=d1
    Head,LHand,RHand,LFoot,RFoot,LHip,RHip,LShoulder,RShoulder,Neck,Torso,LElbow,RElbow,LKnee,RKnee=P_Joints(data)
    joint_selected = np.concatenate((LHand,RHand,LFoot,RFoot,LHip,RHip,LShoulder,RShoulder,LElbow,RElbow,LKnee,RKnee), axis=1)
    features=np.zeros((len(data), 174)) 
    features[:,0]=LHand_Head(LHand,Head).reshape((len(data),))
    features[:,1]=RHand_Head(RHand,Head).reshape((len(data),))
    features[:,2]=LHand_RHand(LHand,RHand).reshape((len(data),))
    features[:,3]=LHip_LFoot(LHip,LFoot).reshape((len(data),))#
    features[:,4]=RHip_RFoot(RHip,RFoot).reshape((len(data),))
    features[:,5:(5+len(joint_selected[0,:]))]=Temporal_joint_tcp(joint_selected)
    features[:,(5+len(joint_selected[0,:])):(len(joint_selected[0,:])+5+len(joint_selected[0,:]))]=Temporal_joint_tci(joint_selected)
    features[:,(len(joint_selected[0,:])+5+len(joint_selected[0,:])):(len(joint_selected[0,:])+len(joint_selected[0,:])+5+len(joint_selected[0,:]))]=Joint_coordinate_mean_difference(joint_selected)
    features[:,(len(joint_selected[0,:])+5+len(joint_selected[0,:])+len(joint_selected[0,:])):(len(joint_selected[0,:])+len(joint_selected[0,:])+len(joint_selected[0,:])+5+len(joint_selected[0,:]))]=Joint_coordinate_variance_difference(joint_selected)
    features[:,0:149]=Feature_normalisation(features[:,0:149])
    i=(len(joint_selected[0,:])+len(joint_selected[0,:])+len(joint_selected[0,:])+5+len(joint_selected[0,:]))
    features[:,i]=angle_LHand(LHand,LElbow,LShoulder).reshape((len(data),))/360
    i+=1
    features[:,i]=angle_RHand(RHand,RElbow,RShoulder).reshape((len(data),))/360
    i+=1
    features[:,i]=angle_LFoot(LFoot,LKnee,LHip).reshape((len(data),))/360
    i+=1
    features[:,i]=angle_RFoot(RFoot,RKnee,RHip).reshape((len(data),))/360
    i=i+1
    features[:,i:i+3]=LHip_LKnee__LKnee_LFoot(LHip,LKnee,LFoot)/m.pi
    i=i+3
    features[:,i:i+3]=RHip_RKnee__RKnee_RFoot(RHip,RKnee,RFoot)/m.pi
    i=i+3
    features[:,i:i+3]=Head_Neck__RHand_RElbow(Head,Neck,RElbow,RHand)/m.pi
    i=i+3
    features[:,i:i+3]=Head_Neck__LHand_LElbow(Head,Neck,LElbow,LHand)/m.pi
    i=i+3
    features[:,i:i+3]=LHand_LElbow__RHand_RElbow(LHand,LElbow,RElbow,RHand)/m.pi
    i=i+3
    features[:,i:i+3]=LElbow_LShoulder__LHand_LElbow(LElbow,LShoulder,LHand)/m.pi
    i=i+3
    features[:,i:i+3]=RElbow_RShoulder__RHand_RElbow(RElbow,RShoulder,RHand)/m.pi

    

    return features,keyframes

