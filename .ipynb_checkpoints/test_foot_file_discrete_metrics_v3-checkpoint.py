# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:16:00 2022

@author: Eric.Honert
"""

#______________________________________________________________________________
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import alphashape
import os
import time
import addcopyfighandler
import plotly.figure_factory as ff

#______________________________________________________________________________
# Functions
def viz3d(PC):
    # Visualize the 3D point cloud. Needs to be updated.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(PC[:,0],PC[:,1],PC[:,2])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
def create_foot_outline(footPC,outline_height):
    idx = point_cloud[:,2] < outline_height
    foot_low = point_cloud[idx,0:2]
    
    test_shape = alphashape.alphashape(foot_low,100)
    
    foot_outline = np.asarray([x for x in test_shape.boundary.coords])
    
    return foot_outline

def AlignBraddock(footPC):
    foot_outline = create_foot_outline(footPC,0.02)
    

    footL = np.max(foot_outline[:,1])
    # Step 1: Find the most medial point less than 20% of the foot length
    idx = foot_outline[:,1] < 0.2*footL
    rear_foot = foot_outline[idx,:]
    min_idx = np.argmin(rear_foot[:,0])
    pt1 = rear_foot[min_idx,:]
        
    # Step 2: Find the Approximate MTP1 Head
    MTP1 = findMTP1(foot_outline)
        
    # Find the rotation angle
    rot_angle = np.arctan2(pt1[0]-MTP1[0],MTP1[1]-pt1[1])
    # Provide the 2D rotation matrix
    R2D = np.asarray([[np.cos(rot_angle),np.sin(rot_angle)],[-np.sin(rot_angle),np.cos(rot_angle)]])
        
    foot_outline = np.transpose(R2D @ np.transpose(foot_outline))
        
    MTP1 = findMTP1(foot_outline)
    R3D = np.asarray([[np.cos(rot_angle),np.sin(rot_angle),0],[-np.sin(rot_angle),np.cos(rot_angle),0],[0,0,1]])
    scan_rot = np.transpose(R3D @ np.transpose(footPC))
    scan_rot[:,0] = scan_rot[:,0] - MTP1[0]
    scan_rot[:,1] = scan_rot[:,1] - np.min(scan_rot[:,1])
    
    return scan_rot
    
    
def findMTP1(foot_outline):
    footL = np.max(foot_outline[:,1])
    idx = np.argmax(foot_outline[:,1])
    
    current_point = np.array([foot_outline[idx,0],0.8*footL+np.min(foot_outline[:,1])])
    
    der_angle = 1
        
    track_st_point = []
    track_angle = []
    
    while der_angle < 90 and current_point[1] > 0.6*footL:
        st_point = current_point
        track_st_point.append(st_point)
        
        # Find the closest point 1mm behind the start point
        idx = np.logical_and(foot_outline[:,1]<st_point[1]-0.001,foot_outline[:,0] < np.mean(foot_outline[:,0]))
        new_pts = foot_outline[idx,:]
        
        min_idx = np.argmin(np.linalg.norm(foot_outline[idx,:] - st_point,axis=1))
        
        current_point = new_pts[min_idx,:]
        # Angle w.r.t. th horizontal: call it the derivative angle
        der_angle = np.arctan2(st_point[1]-current_point[1],st_point[0]-current_point[0])*180/np.pi
        track_angle.append(der_angle)
        
    # If the Met head was not found, pick greatest angle
    if der_angle < 90:
        idx = np.argmax(np.asarray(track_angle))
        MTP1 = track_st_point[idx]
    else: 
        MTP1 = st_point
        
    return MTP1

def FootDiscreteMetrics(footPC):
     foot_outline = create_foot_outline(footPC,0.025)
     
     # Foot Length
     footL = np.max(foot_outline[:,1])-min(foot_outline[:,1])
     
     # Heel Width
     idx = foot_outline[:,1]<(0.2*footL)
     rear_foot = foot_outline[idx,:]
     heel_width = np.max(rear_foot[:,0]) - np.min(rear_foot[:,0])
     # 1st Met Head Distance
     MTP1 = findMTP1(foot_outline)
     MTP1d = MTP1[1]-min(foot_outline[:,1])
     
     # Forefoot Width
     foot_outline = create_foot_outline(footPC,0.025)
     idx = (foot_outline[:,1] < MTP1[1]-0.015)*(foot_outline[:,1]>0.5*np.max(foot_outline[:,1]))
     mid_foot = foot_outline[idx,:]
     MTP5 = max(mid_foot[:,0])
     forefoot_width = MTP5-MTP1[0]
         
     # Instep height: Based on Jurca 2019
     idx = footPC[:,1]>(0.55*footL)
     inH = np.max(footPC[idx,2])
     
     return [footL,heel_width,forefoot_width,inH,MTP1d]
 
def update_faces(new_idx_tf,old_faces):
    # Equvalent to MATLAB find:
    ind = np.argwhere(new_idx_tf==True)
    
    new_faces = []
    for val in old_faces:
        # For debugging purposes
        # new_face_idx_tf = np.concatenate((val[0]==ind,val[1]==ind,val[2]==ind),axis = 1)
        poss = np.argwhere(np.concatenate((val[0]==ind,val[1]==ind,val[2]==ind),axis = 1)==True)
        poss_face = poss[np.argsort(poss[:,-1]),0]
        if poss_face.size == 3:
            new_faces.append(poss_face)
    
    new_faces = np.array(new_faces)
    
    return new_faces

def compute_vol_from_mesh(footPC,faces,normals):
    
    # First make sure that only unique faces are being used
    [faces_up,idx,counts] = np.unique(np.sort(faces,axis=1),axis=0,return_index=True,return_counts=True)
    
    if sum(counts) > len(counts):
        print('Scan with repeated faces, repeats removed')
    
    voxel_vol = [comp_vox_vol(np.mean(np.array([normals[ii[0],:],normals[ii[1],:],normals[ii[2],:]]),axis=0),np.array([footPC[ii[0],:],footPC[ii[1],:],footPC[ii[2],:]])) for ii in faces_up]
    #__________________________________________________________________________
    # Original loop:
    # Pre-allocate the voxel volume variable
    # voxel_vol = []
    # # Index through each face
    # for ii in faces_up:
    #     voxel_vol.append(comp_vox_vol(np.mean(np.array([normals[ii[0],:],normals[ii[1],:],normals[ii[2],:]]),axis=0),np.array([footPC[ii[0],:],footPC[ii[1],:],footPC[ii[2],:]])))
    #__________________________________________________________________________       
        
    tot_vol = sum(voxel_vol)

    return tot_vol

def comp_vox_vol(avg_norm,Voxel):
    prism_height = np.min(Voxel[:,2])
    # Find the lengths of the triangle base for the prism
    a = np.linalg.norm(Voxel[0,0:2]-Voxel[1,0:2])
    b = np.linalg.norm(Voxel[0,0:2]-Voxel[2,0:2])
    c = np.linalg.norm(Voxel[1,0:2]-Voxel[2,0:2])
    # Compute the half perimeter
    s = (a+b+c)/2
    # Heron's formula for triangle area 
    tri_area = np.sqrt(s*(s-a)*(s-b)*(s-c))
    # Compute the volume of the bottom prism of the voxel
    prism_vol = prism_height*tri_area
    # Compute the volume of the top tetrahedron of the voxel
    tetra = Voxel
    tetra[:,2] = tetra[:,2]-prism_height
    tetra_vol = 1/3*tri_area*np.max(tetra[:,2])
    voxel_vol = (prism_vol+tetra_vol)*np.sign(avg_norm[2])
    
    return voxel_vol     

 
def plt_trimesh(vert,face):
    import plotly.io as pio
    pio.renderers.default='browser'
    fig = ff.create_trisurf(vert[:,0],vert[:,1],vert[:,2],face)
    fig.update_layout(scene = dict(aspectmode = 'data'))
    fig.show()
    
def plt_normals(footPC,norms):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(footPC[:,0], footPC[:,1], footPC[:,2], norms[:,0], norms[:,1], norms[:,2], length=0.005, normalize=True)  
    
#______________________________________________________________________________

# Read in files
# only read .asc files for this work
fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Aetrex Object Files\\python_files\\'
fileExt = r".npy"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

# Toggle to save
save_on = 0

# Preallocate variables
store_FL = []
store_FFW = []
store_HW = []
store_inH = []
store_MTP1 = []
store_HAC = []
store_tot_vol = []
store_RF_vol = []
store_MF_vol = []
store_FF_vol = []
store_fname = []
store_lname = []
store_side = []
store_height = []


for ii in range(len(entries)):
    try:
        t = time.time()
        v_f_n = np.load(fPath+entries[ii],allow_pickle=True)
        # obj = pywavefront.Wavefront(fPath+entries[ii],collect_faces=True)
        point_cloud = v_f_n[0]
        faces = v_f_n[1]
        normals = v_f_n[2]
        # Find unique normals
        u_norms = np.array([np.mean(normals[np.where((val[0] == normals[:,3]) & (val[1] == normals[:,4]) & (val[2] == normals[:,5]))],axis=0) for val in point_cloud])
        u_norms = u_norms[:,0:3]
        
        # Make all of the feet right feet
        if entries[ii].find('Left') > 0:
            point_cloud[:,0] = -1*point_cloud[:,0]
            # print(entries[ii])
        
        # Rotate the point cloud and the correspondin unique normals
        zrot = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        point_cloud = point_cloud @ zrot
        u_norms = u_norms @ zrot
        
        # Perform a base alignment of the point cloud
        point_cloud[:,0] = point_cloud[:,0]-np.mean(point_cloud[:,0])
        point_cloud[:,1] = point_cloud[:,1]-np.min(point_cloud[:,1])
        point_cloud[:,2] = point_cloud[:,2]-np.min(point_cloud[:,2])
        #______________________________________________________________________
        # Align the point cloud in the same manor similar to using a braddock device
        point_cloud = AlignBraddock(point_cloud)
        
        # Extract the discrete metrics
        [footL,heel_width,forefoot_width,instepH,MTP1d] = FootDiscreteMetrics(point_cloud)
        #______________________________________________________________________
        # Find the dorsal junction
        test_shape = alphashape.alphashape(point_cloud[:,1:],100)

        foot_outline = np.asarray([x for x in test_shape.boundary.coords])
                
        # Create logical conditions:
        idx = (foot_outline[:,0]>0.05)*(foot_outline[:,1]>0.02)*(foot_outline[:,1]<0.1)
        front_foot = foot_outline[idx,:]
        min_y = np.min(front_foot[:,0])
        min_z = np.min(front_foot[:,1])
        
        front_foot[:,0] = front_foot[:,0]-min_y
        front_foot[:,1] = front_foot[:,1]-min_z
        
        min_idx = np.argmin(np.linalg.norm(front_foot,axis=1))
        
        # plt.scatter(*zip(*foot_outline))
        # plt.plot([0,front_foot[min_idx,0]+min_y],[0,front_foot[min_idx,1]+min_z])
        
        #______________________________________________________________________
        # Crop the foot at the dorsal junction 
        # Boolean opperator
        idx = point_cloud[:,2] < front_foot[min_idx,1]+min_z
        # Compute the foot volume below the dorsal junction
        footPC = point_cloud[idx,:]
        up_faces = update_faces(idx,faces)
        up_unorms = u_norms[idx,:]
        # As there are no longer faces on the top of the foot, the top of the
        # foot will need to lay on the x-y plane where it contributes 0 volume
        dum_PC = footPC
        dum_PC[:,2] = dum_PC[:,2]-np.max(dum_PC[:,2])
        tot_vol = compute_vol_from_mesh(dum_PC,up_faces,up_unorms)
        #______________________________________________________________________
        # Rearfoot Volume
        idx = footPC[:,1]<=front_foot[min_idx,0]+min_y
        rearfootPC = footPC[idx,:]
        rearfootFaces = update_faces(idx,up_faces)
        rearfootNorms = up_unorms[idx,:]
        # As there are no longer faces on the top of the foot, the top of the
        # foot will need to lay on the x-y plane where it contributes 0 volume
        dum_PC = rearfootPC
        dum_PC[:,2] = dum_PC[:,2]-np.max(dum_PC[:,2])
        rearfoot_vol = compute_vol_from_mesh(dum_PC,rearfootFaces,rearfootNorms)
        #______________________________________________________________________
        # Midfoot Volume
        idx = (footPC[:,1]>front_foot[min_idx,0]+min_y)*(footPC[:,1]<=MTP1d)
        midfootPC = footPC[idx,:]
        midfootFaces = update_faces(idx,up_faces)
        midfootNorms = up_unorms[idx,:]
        midfoot_vol = compute_vol_from_mesh(midfootPC,midfootFaces,midfootNorms)
        #______________________________________________________________________
        # Forefoot Volume
        idx = footPC[:,1]>MTP1d
        forefootPC = footPC[idx,:]
        forefootFaces = update_faces(idx,up_faces)
        forefootNorms = up_unorms[idx,:]
        forefoot_vol = compute_vol_from_mesh(forefootPC,forefootFaces,forefootNorms)
        #______________________________________________________________________
        # Heel-Ankle Circumference
        ang = np.arctan2(front_foot[min_idx,0]+min_y,front_foot[min_idx,1]+min_z)
        xrot = np.array([[1,0,0],[0,np.cos(ang),-np.sin(ang)],[0,np.sin(ang),np.cos(ang)]])
        rot_foot = np.transpose(xrot @ np.transpose(point_cloud))
        idx = (rot_foot[:,1]<0.003)*(rot_foot[:,1]>-0.003)
        foot_slice = np.transpose(np.array([rot_foot[idx,0],rot_foot[idx,2]]))
        test_shape = alphashape.alphashape(foot_slice,20)
        HAC = test_shape.length       
        
        #______________________________________________________________________
        # Store variables of interest
        store_FL.append(footL)
        store_FFW.append(forefoot_width)
        store_HW.append(heel_width)
        store_inH.append(instepH)
        store_MTP1.append(MTP1d)
        
        store_tot_vol.append(tot_vol)
        store_RF_vol.append(rearfoot_vol)
        store_MF_vol.append(midfoot_vol)
        store_FF_vol.append(forefoot_vol)
        
        store_HAC.append(HAC)
        
        store_fname.append(entries[ii][:-8])
        elapsed = time.time()-t
        print(elapsed)
        
        # plt.scatter(*zip(*foot_outline))
    except:
            print(entries[ii])


    
        
outcomes = pd.DataFrame({ 'Name_Side':list(store_fname),
                         'FootLength':list(store_FL), 'ForefootWidth':list(store_FFW),
                         'HeelWidth':list(store_HW),'InstepHeight':list(store_inH),
                         'MTP1':list(store_MTP1),'HACirc':list(store_HAC),
                         'TotVol':list(store_tot_vol),'RFVol':list(store_RF_vol),
                         'MFVol':list(store_MF_vol),'FFVol':list(store_FF_vol)})
        

if save_on == 1:         
    outcomes.to_csv('C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Aetrex Object Files\\python_files\\SummaryMetrics.csv',mode='a',header=True)    


# Visualize the point cloud:

# viz3d(point_cloud)
