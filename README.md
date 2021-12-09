# InceptionFCN
This is the InceptionFCN network a modified network of the InceptionTime
The original code was taken from InceptionTime: Finding AlexNet for Time Series Classification (https://github.com/hfawaz/InceptionTime)

## Getting Started
First of all you need to install the enviorenments from enviorenments.yml file to your conda enviorenment by running following command
```shell script
conda env create -f enviorenment.yml
```
and then 
```shell script
conda activate dl4-tsc
```
For network training you have to download UCR archive from here (https://www.cs.ucr.edu/~eamonn/time_series_data/) and create the folder named "archives" and put the UCR archive inside and rename the folder to TSC or you can change your folder names or pathes within the code. 
Go to main.py file and change the directories in line 78.

Also you can go to /utils/ folder and change the constansts 

For network training instructions follow the authors of InceptionTime
We implemented the a different network within the existing network named ICNv2.py in the /classfiers/
