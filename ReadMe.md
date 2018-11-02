# Emergency Vehicle Detector

## Overview
Detect emergency vehciles like Police Cars, Ambulance and Fire trucks using deep learning.

## Build Instructions

1) git clone https://github.com/anirudhtopiwala/NG-Challenge-Emergency-Vehicle-Detection.git 

2) Download a folder named [**Input**](https://drive.google.com/open?id=1StzCPEiY9kks89FHzmRJ4CfvXim1GO8I) into the same Directory. 

3) Download a folder named [**net**](https://drive.google.com/open?id=10QSYrzjFdaItEtfmxRfBpry2lzP69SQx) into the same Directory. 

4) Run ng_main.m

## Annotation Instructions:

1) Load images into an [image datastore variable](https://www.mathworks.com/help/matlab/ref/matlab.io.datastore.imagedatastore.html). An example can be found in the code.
 * This is done so as to define a specific sequence of loading images.
 
2) open Matlabs Image labeler app and import the datastore variable. (Matlab R2018A)

3) Create Labels as ambulance, fireTruck and police. Create negative images under nonemergency vehicle label.

4) Start Drawing Bounding Boxes.

5) Once done export variable at "labelall". Datatype = table


