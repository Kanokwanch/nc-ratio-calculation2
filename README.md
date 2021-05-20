# Application for N/C ratio calculation from ascites fluid cell
https://share.streamlit.io/aaomsan/nc-ratio-calculation2/main/app.py

The purpose of this project is to calculate the N/C ratio from ascites fluid cells to facilitate
pathologist instead of reading the result by human eyes via microscope. This project uses the image
processing technique to find the boundary of each cell by these processes that will be stated after
this. First, we use HSV color separation. Second, we calculate the N/C ratio then uses algorithm
k-means clustering to find a group of cell that looks similar and transforms to a grayscale image by
using a threshold to get the noise image to contain a point in the image and uses the opening
technique to remove noises in the image after that separate the adjacent cell and identify unknown
region to identify the center of the cell and use watershed algorithm to find the boundary of the
cell then calculate Euclidean distance to obtain nucleus and cytoplasm area. Finally, we calculate
N/C ratio by using nucleus are divided by cytoplasm area and display on a web application to facilitate
pathologist.
