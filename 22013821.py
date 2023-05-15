# the Pandas package is a tool to Python that the lets your work with sets of data. It has tools that can help you clean, study, analyse, and change data.
import pandas as pd
# "Numerical Python," which is what NumPy stands for, is a Python tool that is free to use or is used in almost every field of science and business. 
import numpy as np
# Using the sk_learn package to load the kmean function.
from sklearn.cluster import KMeans 
# Matplotlib acts as Python tool that the lets you make visualisations that are still, dynamic, or that you can interact with. Matplotlib makes easy things easy and makes hard things possible.
import matplotlib.pyplot as plt

# bringing in warnings.
import warnings 
warnings.filterwarnings('ignore')

"""# Dataset reading Using Pandas:

In this dataset showing the population in largest cities from 1960 to 2022. In this code, first the code is pre-processed with preprocessing methods. After preprocessing clusters are made using k means technique. In this code multiple graphs are shown for population changes in largest cities. At last a curve fit is shown which is the best fit graph.
"""

# Creates the code that can be used to figure out what the data means.
def Dataset(data_file):
    population_in_large_city_data = pd.read_csv(data_file, skiprows=4) #read data and store into variable.
    population_in_large_city_data_1 = population_in_large_city_data.drop([ 'Unnamed: 67', 'Indicator Code',  'Country Code'],axis=1) # drop some column which is not useful.
    population_in_large_city_data_2 = population_in_large_city_data_1.set_index("Country Name")  # set the country column as index.
    population_in_large_city_data_2 = population_in_large_city_data_2.T # transpose all the data.
    population_in_large_city_data_2.reset_index(inplace=True) # reset index name.
    population_in_large_city_data_2.rename(columns = {'index':'Year'}, inplace = True) # rename the column into year.
    return population_in_large_city_data_1, population_in_large_city_data_2 

# define the path of data.
data_file = 'C:\\Users\\ashin\\Downloads\\API_EN.URB.LCTY_DS2_en_csv_v2_5458076\API_EN.URB.LCTY_DS2_en_csv_v2_5458076.csv'
# callinf the function and store data into variable.
population_in_large_city_Final_Data, population_in_large_city_Transpose_data = Dataset(data_file) 
population_in_large_city_Final_Data_5 = population_in_large_city_Final_Data 
# showing starting rows. 
population_in_large_city_Final_Data.head()

population_in_large_city_Transpose_data.head() # showing the starting 5 rows of dataset.

# check the null values in column.
population_in_large_city_Final_Data.isnull().sum()

# Extracting the data with the help of function.
def population_in_large_city_Final_Data_2(urban_population_Final_Dataset): 
    population_in_large_city_Final_Data_1 = urban_population_Final_Dataset 
    # droped all null values from data.
    population_in_large_city_Final_Data_2 = population_in_large_city_Final_Data_1.dropna() 
    return population_in_large_city_Final_Data_2

# calling the function to extract the data. 
population_in_large_city_Final_Data_3 = population_in_large_city_Final_Data_2(population_in_large_city_Final_Data) 
# storing the country name into variable.
country_name = population_in_large_city_Final_Data_3['Country Name'] 
# shows starting rows from data.
population_in_large_city_Final_Data_3.head(10)

# check shape of data.
population_in_large_city_Final_Data_3.shape

# after droping the null values from data.
population_in_large_city_Final_Data_3.isnull().sum()

# using describe function showing data into dataframe.
population_in_large_city_Final_Data_3.describe().T

"""*** Label Encoder ***"""

# importing the label encoder from sk_learn.
from sklearn.preprocessing import LabelEncoder
# define classifier for encoder.
encoded = LabelEncoder()
# fit classifier with data.
population_in_large_city_Final_Data_3['Country Name'] = encoded.fit_transform(population_in_large_city_Final_Data_3['Country Name']) 
# showing 5 rows from data.
population_in_large_city_Final_Data_3.head(7)

"""*** Define X and Y Variable ***"""

X = population_in_large_city_Final_Data_3.drop(['Country Name','Indicator Name'], axis=1)
y = population_in_large_city_Final_Data_3['Country Name']  

# Bringing in a standard scaler to make the data even.
from sklearn.preprocessing import StandardScaler
# define classifier for scalling.
stand_scaling = StandardScaler()
# fit classifier with data.  
stand_scaling = stand_scaling.fit_transform(X)

"""# Using Elbow Method To getting the Clusters. """

# finding the groups by using the elbow method.
from scipy.spatial.distance import cdist 
Clus_terd = range(10) 
Mean_dist = list()

for S in Clus_terd:
    algo = KMeans(n_clusters=S+1) 
    algo.fit(stand_scaling) 
    Mean_dist.append(sum(np.min(cdist(stand_scaling, algo.cluster_centers_, 'euclidean'), axis=1)) / stand_scaling.shape[0]) 

# Setting all the settings or drawing the line.
# set figure size.
plt.figure(figsize=(8,5))
# set parameter for graph.
plt.plot(Clus_terd, Mean_dist, marker="o", color='b') 
# set xlabel.
plt.xlabel('<<----- Numbers of Clusters ----->>',fontsize=15)
# set ylabel.
plt.ylabel('<<----- Average Distance ----->>', fontsize=15) 
plt.grid()
# set title for graph.
plt.title('Choosing Clusters with Elbow Method', color='r', fontsize=20);

# Get the groups set up for the classification.
c_mean_tech = KMeans(n_clusters=3, max_iter=100, n_init=10, random_state=10)
# putting input into the classifier.  
c_mean_tech.fit(stand_scaling) 
# prediction model to get the name.
geting_prediction = c_mean_tech.predict(stand_scaling)  
geting_prediction

# 3 clusters are made using kmeans clustering approach, 
# The number of points occuring in a cluster are shown here
cluster_label, cluster_points_count = np.unique(geting_prediction, return_counts=True)

# Printing the count of Points occuring in each cluster
for value, count in zip(cluster_label, cluster_points_count):
    print(f"{value}: {count}")

# fixed the colour names for all the groups.
set_the_new_colors = {0 : 'b', 1 : 'g', 2 : 'c'} 
def colors(x):  
    return set_the_new_colors[x]  
new_color = list(map(colors, c_mean_tech.labels_))   

# define figure size.
plt.figure(figsize=(8,5))
# fix parameter for scatter plot.
plt.scatter(x=X.iloc[:,0], y=X.iloc[:,2], c=new_color)  
# fix the xlabel for graph.
plt.xlabel('<<----- 1960 ----->>', fontsize=15)
# fix the ylabel for graph.  
plt.ylabel('<<----- 1962 ----->>', fontsize=15) 
plt.grid()
# fix the title for graph. 
plt.title('Scatter plot for 3 Clusters', color='r', fontsize=20);

# extracting the labels and Centroids.
geting_different_centroids = c_mean_tech.cluster_centers_
get_label = np.unique(geting_prediction) 
geting_different_centroids

# set the figure size for graph.
plt.figure(figsize=(10,7))
for i in get_label:
    plt.scatter(stand_scaling[geting_prediction == i , 0] , stand_scaling[geting_prediction == i , 1] , label = i)  

# set variables for graph such colour, data etc.
plt.scatter(geting_different_centroids[:,0] , geting_different_centroids[:,2] , s = 40, color = 'black') 
# fix xlabel for graph.
plt.xlabel('<<----- 1960 ----->>', fontsize=15)
# fix ylabel for graph.  
plt.ylabel('<<----- 1962 ----->>', fontsize=15) 
plt.grid() 
# fix title for graph.
plt.title('Showing Clusters with their Centroids', color='r', fontsize=20) 
plt.legend()  
plt.show()

# creating the empty list that helps to store different clauster in these list.
cluster_first=[]
cluster_second=[] 
cluster_third=[] 

# Use the loop condition to find out information in each cluster.
for i in range(len(geting_prediction)):
    if geting_prediction[i]==0:
        cluster_first.append(population_in_large_city_Final_Data.loc[i]['Country Name']) 
    elif geting_prediction[i]==1:
        cluster_second.append(population_in_large_city_Final_Data.loc[i]['Country Name'])
    else:
        cluster_third.append(population_in_large_city_Final_Data.loc[i]['Country Name'])

# shows the information that is in the first cluster.
cluster_first_information = np.array(cluster_first)
# shows the information that is in the second cluster.
cluster_second_information = np.array(cluster_second)
# shows the information that is in the third cluster.
cluster_third_information = np.array(cluster_third)   

print(cluster_first_information)

print(cluster_second_information)

print(cluster_third_information)

cluster_1st = cluster_first_information[4] 
print('Cluster_1st_Country_name :', cluster_1st) 

cluster_2nd = cluster_second_information[1] 
print('Cluster_2nd_Country_name :', cluster_2nd) 

cluster_3rd = cluster_third_information[3] 
print('Cluster_3rd_Country_name :', cluster_3rd)

print('Country name :', cluster_1st)
Albania_country = country_name[country_name==cluster_1st]
Albania_country = Albania_country.index.values
Albania_country = population_in_large_city_Final_Data_3[population_in_large_city_Final_Data_3['Country Name']==int(Albania_country)]  
Albania_country = np.array(Albania_country)  
Albania_country = np.delete(Albania_country, np.s_[:3]) 
Albania_country

print('Country name :', cluster_2nd) 
Latvia_country = country_name[country_name==cluster_2nd]
Latvia_country = Latvia_country.index.values
Latvia_country = population_in_large_city_Final_Data_3[population_in_large_city_Final_Data_3['Country Name']==int(Latvia_country)] 
Latvia_country = np.array(Latvia_country)  
Latvia_country = np.delete(Latvia_country,np.s_[:3]) 
Latvia_country

print('Country name :', cluster_3rd) 
Bosnia_and_Herzegovina_country = country_name[country_name==cluster_3rd]
Bosnia_and_Herzegovina_country = Bosnia_and_Herzegovina_country.index.values
Bosnia_and_Herzegovina_country = population_in_large_city_Final_Data_3[population_in_large_city_Final_Data_3['Country Name']==int(Bosnia_and_Herzegovina_country)] 
Bosnia_and_Herzegovina_country= np.array(Bosnia_and_Herzegovina_country)  
Bosnia_and_Herzegovina_country = np.delete(Bosnia_and_Herzegovina_country, np.s_[:3]) 
Bosnia_and_Herzegovina_country

# plotting the line curve for different clusters.
years = list(range(1960,2022))
# define figure size for graph.
plt.figure(figsize=(22,8))

# Define the data for each cluster.
cluster_data = [
    ('Albania Country', Albania_country, 'g'),
    ('Latvia Country', Latvia_country, 'b'),
    ('Bosnia_and_Herzegovina Country', Bosnia_and_Herzegovina_country, 'black')]

# Define figure size for the graph.
fig, axes = plt.subplots(1, len(cluster_data), figsize=(22, 8))

# Iterate over each cluster and plot the line graph.
for i, (title, data, color) in enumerate(cluster_data):
    axis = axes[i]
    axis.plot(years, data, color=color)
    axis.set_xlabel('<<----- Years ----->>', fontsize=15)
    axis.set_ylabel('<<----- Population in largest city ----->>', fontsize=15)
    axis.set_title(title, color='r', fontsize=20)
    axis.grid(True)

plt.tight_layout()
plt.show()

"""# Curve Fitting."""

#Choose all the columns and turn them into a collection.
data = np.array(population_in_large_city_Final_Data_5.columns) 
# dropped some of the columns.
data = np.delete(data,0) 
data = np.delete(data,0) 
# change into data type as int.
data = data.astype(np.int)
print('Year:\n',data)

# Choosing all the facts for India country.
new_data_for_curve_fit = population_in_large_city_Final_Data_5[(population_in_large_city_Final_Data_5['Indicator Name']=='Population in largest city') & (population_in_large_city_Final_Data_5['Country Name']=='India')]   

# change into array.
data_1 = new_data_for_curve_fit.to_numpy()
# dropped some of the columns.
data_1 = np.delete(data_1,0) 
data_1 = np.delete(data_1,0)
# change into data type as int.
data_1 = data_1.astype(np.int) 
print('Population:\n',data_1)

# A portion of the library is being brought in.
import scipy
from scipy.optimize import curve_fit
from scipy import stats 

# Set the equation that has to be fit. For a linear function, this would be y = mx + c.
def linear_func(x, m, c):
    return m*x + c

def curve_fitting_graph(x,y): 

    # Adjust the curves or fitting.
    popt, pcov = curve_fit(linear_func, x, y) 

    # Get the values that were fit or their standard errors
    m, c = popt
    m_err, c_err = np.sqrt(np.diag(pcov)) 

   # Figure out the bottom and top of the confidence range.
    ineterval = 0.95  # Put 95% in the confidence range
    difference = 1.0 - ineterval 
    a_low, a_high = scipy.stats.t.interval(difference, len(x)-2, loc=m, scale=m_err)
    b_low, b_high = scipy.stats.t.interval(difference, len(x)-2, loc=c, scale=c_err)

    # plot the best-fitting function or the range of confidence.
    # fix figure size.
    plt.figure(figsize=(12,6)) 
    #fix data for graph.
    plt.plot(x, y, '+', label='Data',color='black') 
    # set the parameter for curve fit function.
    plt.plot(x, linear_func(x, m, c), 'g', label='Fitted Function')
    plt.fill_between(x, linear_func(x, a_low, b_low), linear_func(x, a_high, b_high), color='b', alpha=0.5, label='Confidence Range') 
    #fix ylabel for the graph.
    plt.title('Curve Fitting',color='r', fontsize=20) #set title for graph.
    #fix xlabel for the graph.
    plt.xlabel('<<----- Years ----->>', fontsize=15) 
    #fix ylabel for the graph.
    plt.ylabel('<<----- Population ----->>', fontsize=15)  
    plt.grid()
    # set legend in graph.
    plt.legend() 
    plt.show() 
    
curve_fitting_graph(data,data_1)

