==========
Data Info
==========
The `clustering` directory contains 2 data files covering a region of 10deg*10deg:
1. 0_10_40_50 ---> 0<l[deg]<10 and 40<b[deg]<50
2. 210_220_-40_-50 ---> follows same format as above
Depending on whether region is above the Galactic disc (i.e. b>0) or below (i.e. b<0), the files are located in the relevant subdirectories.
The HDBSCAN labels and persistences are saved under clus_info. 
The search algorithm partitions the search regions into smaller blocks of side lengths L=1deg, 2deg and 5deg.   
Data is from Gaia DR3.

The `cnn_classification` directory contains supplementary data for, and preliminary results from, the CNN process, e.g.:
- CMDs generated from overdensities for classification. Image size is 128*128 pixels
- Preliminary cluster membership lists
- source_id files for member resolution over multiple instances of the same cluster.
- Tables documenting classification probabilities 
