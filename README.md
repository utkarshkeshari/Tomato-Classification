## Tomato-Classification

### How to Use
- Download the project 
- Go to the Project Folder
- Run Python manage.py runserver
- Open a browser 127.0.0.1:8000


#### Technology Used
- Python - A General Purpose Language
- Pandas - A Python Based Data Analysis Tool
- MatplotLib - A Python Based Plotting Library
- Scikit-Learn - A Python Based Machine Learning Tool
- Django - A Python based Web Framework

#### Feature Engineering
- Image Generation 
- Extracting Features from Image
- Creating a CSV File

#### Predictive Algorithms
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbour (KNN)
- Decision Tree
- Random Forest
- XgBoost
- Gradient Boosting

#### Evaluation Metric
- Accuracy Score
- Confusion Matrix
- Learning Curve



### Image Augmentation

From the given set of images we have generated more images by using the Image Generator method of Keras (A Deep Learning Package for Python) 

Techniques used to generate Images
- By rotation 
- By flipping the image horizontally
- By zooming some random pixels 

For a given 5 class, around 1000-1200 images were generated for each class
Total image generated is around 5000

### Feature Extraction

We have removed the borders of each images by leaving the 10 pixels from each side.
Using these images we have extracted features.
There were three channel (Red, Green, Blue) in each image.
Mean of red channel
Mean of green channel
Mean of blue channel

After reading the image, we have converted the pixels in a 3 dimensional array (Here 3rd dimension represents the red,green & blue) .
For each color matrix, 
We have selected the 100th row and columns from 50 to 100 and took the mean
We have selected the 500th row and columns from 50 to 100 and took the mean
We have selected the 100th column and Rows from 50 to 100 and took the mean
We have selected the 500th column and Rows from 50 to 100 and took the mean

So, we have created a total of 15 features from those images for each class.
After taking out these 15 feature we put this record in a CSV file along with label.

### Training

Using the CSV file created above we have trained the 6 Different machine learning algorithms.



