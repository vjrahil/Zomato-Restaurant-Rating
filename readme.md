
# File descriptions:
1. code.ipynb -> contains the main code for our model <br />
2. code.py -> same as code.ipynb <br />
3. report -> report of our project <br />
4. test_X -> testing data for our trained model <br />
5. test_Y -> desired output for test_X <br />
6. zomato.csv -> complete dataset for our model <br />

# Dataset Source
https://www.kaggle.com/shrutimehta/zomato-restaurants-data#zomato.csv

# Analysis of Dataset
Our dataset consists of various attributes such as ‘Restaurant ID’, ‘Restaurant Name’,
‘Country Code’, ‘City’, ‘Address’, ‘Locality’, ‘Locality Verbose’, ‘Longitude’, ‘Latitude’,
‘Cuisines’, ‘Average Cost for two’, ‘Currency’, ‘Has Table booking’, ‘Has Online delivery’, ‘Is
delivering now’, ‘Switch to order menu’, ‘Price range’, ‘Aggregate rating’, ‘Rating color’,
‘Rating text’ and ‘Votes’.
We thought of all the attributes that would be important for deciding the rating of a
restaurant and finally used ‘Average Cost for two’, ‘Price range’ and ‘Votes’ to find the rating
of the restaurants which resulted in good accuracy of the model.

# Accuracy
1)After using random forest using these ‘Average Cost for two' and ‘Votes’ , we get an accuracy of 0.93.<br />

2)After using random forest using these ‘Average Cost for two' ,'Price Range' and ‘Votes’ , we get an accuracy of 0.935.
