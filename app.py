# Importing required libraries, obviously
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os

import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
from skimage.morphology import extrema
from skimage.morphology import watershed as skwater
import sys
import datetime, pytz
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt

def detect(img):
    img = np.array(img.convert('RGB'))
    count = 0
    out = [] 
    cnt_lessthan04 = 0
    cnt_lessthan06 = 0
    cnt_lessthan10 = 0
    cell_hsvmin  = (52,88,124)  #Lower end of the HSV range defining the nuclei
    cell_hsvmax  = (150,190,255) #Upper end of the HSV range defining the nuclei

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
    Z = hsv.reshape((-1,3))  #number of chanel, number of row
# print(Z)

#Convert to floating point 32 bit
    Z = np.float32(Z)
# plt.hist(Z),plt.show()
# histr = cv2.calcHist(Z,[0],None,[256],[0,256]) 
  
# # show the plotting graph of an image 
# plt.plot(histr) 
# plt.show()


# plt.hist(Z,256,[0,256]),plt.show()


# plt.scatter(Z[:,0], Z[:,1])
# plt.show()
# dc=DominantColor
#Define the K-means criteria, these are not too important
#whenever 10 iterations of algorithm is ran, or an accuracy of epsilon = 1.0 is reached, stop the algorithm and return the answer.

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) 

#Define the number of clusters to find
#นับจาก histrogram มั้ง มันมี4กลุ่ม >> k=4
    K = 4

#Perform the k-means transformation. What we get back are:
#*Centers: The coordinates at the center of each 3-space cluster
#*Labels: Numeric labels for each cluster
#*Ret: A return code indicating whether the algorithm converged, &c.
    ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

#Produce an image using only the center colours of the clusters
    center = np.uint8(center)
    khsv   = center[label.flatten()]
    khsv   = khsv.reshape((img.shape))
# ShowImage('K-means',khsv,'hsv')

#Reshape labels for masking
    label = label.reshape(img.shape[0:2])
# ShowImage('K-means Labels',label,'gray')

#(Distance,Label) pairs
    nucleus_colour = np.array([139, 106, 192])
    cell_colour    = np.array([130, 41,  207])
# nucleus_colour = np.array([44, 71, 100])
# cell_colour    = np.array([112, 140,  171])

    nuclei_label  = (np.inf,-1)
    cell_label    = (np.inf,-1)
    for l,c in enumerate(center):
  # print(l,c)
        dist_nuc = np.sum(np.square(c-nucleus_colour)) #Euclidean distance between colours
        if dist_nuc<nuclei_label[0]:
            nuclei_label=(dist_nuc,l)
        dist_cell = np.sum(np.square(c-cell_colour)) #Euclidean distance between colours
        if dist_cell<cell_label[0]:
            cell_label=(dist_cell,l)
    nuclei_label = nuclei_label[1]
    cell_label   = cell_label[1]
# print("Nuclei label={0}, cell label={1}".format(nuclei_label,cell_label))

#Multiply by 1 to keep image in an integer format suitable for OpenCV
    thresh = cv2.bitwise_or(1*(label==nuclei_label),1*(label==cell_label))
    thresh = np.uint8(thresh)
# ShowImage('Binary',thresh,'gray')

#Remove noise by eliminating single-pixel patches
    kernel  = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel, iterations = 2)
# ShowImage('Opening',opening,'gray')

#Identify areas which are surely foreground
    fraction_foreground = 0.75
    dist         = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist,fraction_foreground*dist.max(),255,0)
# ShowImage('Distance',dist,'gray')
# ShowImage('Surely Foreground',sure_fg,'gray')

#Identify areas which are surely foreground
    h_fraction = 0.1
    dist     = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    maxima   = extrema.h_maxima(dist, h_fraction*dist.max())
# print("Peaks found: {0}".format(np.sum(maxima))) 
#หาค่าความสูง
#Dilate the maxima so we can see them
    maxima   = cv2.dilate(maxima, kernel, iterations=2)
# ShowImage('Distance',dist,'gray')
# ShowImage('Surely Foreground',maxima,'gray')

# Finding unknown region
    unknown = cv2.subtract(opening,maxima)
# ShowImage('Unknown',unknown,'gray')

# Marker labelling
    ret, markers = cv2.connectedComponents(maxima)
# ShowImage('Connected Components',markers,'rgb')

# Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

# Now, mark the region of unknown with zero
    markers[unknown==np.max(unknown)] = 0

# ShowImage('markers',markers,'rgb')

    dist    = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    markers = skwater(-dist,markers,watershed_line=True)

# ShowImage('Watershed',markers,'rgb')
    imgout = img.copy()
    imgout[markers == 0] = [0,0,255] #Label the watershed_line
    

    #x = datetime.datetime.now()
    #st.success(x.strftime("%x"))
    tz = pytz.timezone('Asia/Bangkok')
    now1 = datetime.datetime.now(tz)
    month_name = 'x มกราคม กุมภาพันธ์ มีนาคม เมษายน พฤษภาคม มิถุนายน กรกฎาคม สิงหาคม กันยายน ตุลาคม พฤศจิกายน ธันวาคม'.split()[now1.month]
    thai_year = now1.year + 543
    time_str = now1.strftime('%H:%M:%S')
    st.success("%d %s %d     %s"%(now1.day, month_name, thai_year, time_str)) 

# img.save(imgout)
# cv2.imwrite('output.jpg', imgout)

    for l in np.unique(markers):
        if l==0:      #Watershed line
            continue
        if l==1:      #Background
            continue
    #For displaying individual cells
    #temp=khsv.copy()
    #temp[markers!=l]=0
    #ShowImage('out',temp,'hsv')
        temp = label.copy()
        temp[markers!=l]=-1
        nucleus_area = np.sum(temp==nuclei_label)
        cell_area    = np.sum(temp==cell_label)
    # print("nucleus_area", nucleus_area)
    # print("cell_area",  cell_area)
    
    # file1.close()
	
        if nucleus_area/(cell_area+nucleus_area) != 0 and nucleus_area/(cell_area+nucleus_area) != 1 and nucleus_area/(cell_area+nucleus_area) != 0.5:
            # output.append(l,nucleus_area/(cell_area+nucleus_area))
            # output.append('\n')
            st.write("N/C ratio is ",nucleus_area/(cell_area+nucleus_area))
            out.append(nucleus_area/(cell_area+nucleus_area))
            count = count + 1
            if nucleus_area/(cell_area+nucleus_area) <=0.40:
                cnt_lessthan04 = cnt_lessthan04 +1
            elif nucleus_area/(cell_area+nucleus_area) >0.40 and nucleus_area/(cell_area+nucleus_area)<=0.6:
                cnt_lessthan06 = cnt_lessthan06 +1
            elif nucleus_area/(cell_area+nucleus_area) >0.60 and nucleus_area/(cell_area+nucleus_area)<=1.0:
                cnt_lessthan10 = cnt_lessthan10 +1
	    
            # st.markdown("Nucleus fraction for cell {0} is {1}".format(l,nucleus_area/(cell_area+nucleus_area)))
            # test2.append(nucleus_area/(cell_area+nucleus_area))
            # output.append("Nucleus fraction for cell {0} is {1}".format(l,nucleus_area/(cell_area+nucleus_area)))
            
            # test1 = "\n".join(output)
            
            # output = "\n".join(output)
            # print("Nucleus fraction for cell {0} is {1} ".format(l,nucleus_area/(cell_area+nucleus_area)))
    # print(*output, sep = "\n")
#     cnt_lessthan04 = 0
#     cnt_lessthan06 = 0
#     cnt_lessthan10 = 0
#     for i in out:
# 	if i<=0.40:
# 		cnt_lessthan04 = cnt_lessthan04 +1
# 	else if i>0.40 && i<=0.6:
# 		cnt_lessthan06 = cnt_lessthan06 +1
# 	else if i>0.6 && i<=1.0:
# 		cnt_lessthan10 = cnt_lessthan10 +1
    st.write("0.00 - 0.40 : ", cnt_lessthan04)
    st.write("0.41 - 0.60 : ", cnt_lessthan06)
    st.write("0.61 - 1.00 : ", cnt_lessthan10)
    st.write("Sum of cell is ", count)
    #histrogram
    # arr = out
    # fig, ax = plt.subplots()
    # ax.hist(arr, bins=3)
    # st.pyplot(fig)
    #bar graph
    features = np.array(['0.00 - 0.40', '0.41 - 0.60', '0.61 - 1.00'])
    features_importances = np.array([cnt_lessthan04, cnt_lessthan06, cnt_lessthan10])

    chart_data = pd.DataFrame()
    chart_data['range'] = features
    chart_data['the number of cells'] = features_importances

    chart_v1 = alt.Chart(chart_data).mark_bar().encode(
    x='range',
    y='the number of cells')
    st.write("", "", chart_v1)
    return img

def about():
	st.write(
		'''
		**About** 

            1. Kanokwan Chasuwan
            2. Chanistha Krikhajornkitti

            The purpose of this paper is to calculate nc ratio from ascites fluid cell to facilitate pathologist instead of reading 
            the result by human eyes via microscope.  This project use image processing technique to find boundary 
            of each cell by using HSV color separation after that calculate the nc ratio then use algorithm k-means clustering 
            to find group of cell that looks similar and transform to grayscale image by using threshold 
            to get the noise image contain point in image and use opening technique to remove noises in the image 
            after that separate the adjacent cell and identify unknown region to identity the center of the cell 
            and use watershed algorithm to find boundary of the cell then calculate Euclidean distance to obtain 
            nuclease and cytoplasm area. Finally calculate n/c ratio by using nuclease are divide by cytoplasm area 
            and display on web application to facilitate pathologist.
            
		''')
    

    


def main():
    st.title("NC Ratio Calculation 👩🏼‍⚕️")
    st.write("**Senior Project**")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", activities)
   
    

    
    if choice == "Home":

    	st.write("----")
        
        # You can specify more file types below if you want
    	image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

    	if image_file is not None:

    		image = Image.open(image_file)

    		if st.button("Calculate"):
                
                # result_img is the image with rectangle drawn on it (in case there are faces detected)
                # result_faces is the array with co-ordinates of bounding box(es)
    			result_img = detect(img=image)
    			st.image(result_img, use_column_width = True)
                # st.markdown(result_out)
                # st.success("Nucleus fraction for cell is {} \n".format(result_out))
    		
                # st.success("Found {} faces\n".format(len(result_faces)))
            

    elif choice == "About":
    	about()




if __name__ == "__main__":
    main()
