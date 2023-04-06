# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 18:36:52 2023

@author: ASHIN DEV U A
"""

# importing the required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dataFrame(file_name, years, countries, col, value1):
    '''
    defining a function that returns two dataframes,
    one with countries as columns and other with years
    as columns. Function takes file name and the required
    filters as its attribute
    '''
    # reading the file
    df = pd.read_csv(file_name, skiprows=4)
    
    # cleaning the dataframe
    df1 = df.groupby(col, group_keys= True)
    df1 = df1.get_group(value1)
    df1 = df1.reset_index()
    c = df1['Country Name']
    df1 = df1.iloc[countries, years]
    df1.insert(loc=0, column = 'Country Name', value=c)
    df1 = df1.dropna(axis = 1)
    df2 = df1.set_index('Country Name').T
    return df1,df2


# creating lists with the indices of required years and countries
years = [35,40,45,50,55,60,64]
countries = [251,40,119,55,109,81,77,29]


# creating the required dataframes for plotting
population_c,population_y =dataFrame("API_19_DS2_en_csv_v2_5240184.csv",years,countries,"Indicator Name","Population, total")
co2_c,co2_y = dataFrame("API_19_DS2_en_csv_v2_5240184.csv",years,countries,"Indicator Name","CO2 emissions (kt)")
gdp_c,gdp_y = dataFrame("API_NY.GDP.MKTP.CD_DS2_en_csv_v2_5358352.csv",years,countries,"Indicator Name","GDP (current US$)")
re_c,re_y = dataFrame("API_19_DS2_en_csv_v2_5240184.csv",years,countries,"Indicator Name","Electricity production from renewable sources, excluding hydroelectric (kWh)")
fa_c,fa_y = dataFrame("API_19_DS2_en_csv_v2_5240184.csv",years,countries,"Indicator Name","Forest area (sq. km)")
ec_c,ec_y = dataFrame("API_19_DS2_en_csv_v2_5240184.csv",years,countries,"Indicator Name","Electricity production from coal sources (% of total)")


# using the describe function to explore the data
co2_y.describe()


# using the mean function to find the average CO2 production in different countries over the years and save the data as a csv file
mean = co2_y.mean()
mean.to_csv(r"C:\Users\U.A AJEESH\Desktop\data science\herts\applied ds 1\assignmnt 2/mean.csv")


def plot_p(DataFrame, col, types, name):
    '''
    defining a function that returns a bar plot.
    The function takes the dataframe as its attribute
    and returns a graph over the required columns. It also
    adds a title and label for y axis.
    '''
    
    ax=DataFrame.plot(x=col, rot=45, figsize=(50,25), kind= types, title= name,fontsize=40)
    ax.legend(fontsize=35)
    ax.set_xlabel('Country', fontsize=30)
    ax.set_title(name,pad=20, fontdict={'fontsize':50})
    return

# plotting the required bar plots and saving the plots as a jpg file
plot_p(population_c,"Country Name","bar","Total population")
plt.savefig('Total population.jpg',bbox_inches='tight')

plot_p(gdp_c,"Country Name", "bar","GDP (current US$)")
plt.savefig('GDP.jpg',bbox_inches='tight')

plot_p(co2_c,"Country Name","bar","CO2 emissions (kt)")
plt.savefig('Total Co2 Emission.jpg',bbox_inches='tight')

plot_p(ec_c,"Country Name","bar","Electricity production from coal sources (% of total)")
plt.savefig('Electricity production from coal sources (% of total)n.jpg',bbox_inches='tight')


#plotting the required line plots for Electricity production from renewable sources, excluding hydroelectric (kWh)

legend_properties = {'weight':'bold','size':45}
ax1 = re_y.plot(figsize=(60,30),kind="line",fontsize=45,linewidth=4.0)
ax1.set_title("Electricity production from renewable sources, excluding hydroelectric (kWh)",pad=20, fontdict={'fontsize':60})
ax1.legend(loc=2,prop=legend_properties)
#saving the plot as a jpg file
plt.savefig('Electricity production from renewable sources, excluding hydroelectric (kWh).jpg')

#plotting the required line plots forForest area (sq. km)
legend_properties = {'weight':'bold','size':45}
ax2 = fa_y.plot(figsize=(60,30),kind="line",fontsize=45,linewidth=4.0)
ax2.set_title("Forest area (sq. km)",pad=30, fontdict={'fontsize':55})
ax2.legend(prop=legend_properties)
#saving the plot as a jpg file
plt.savefig('Forest area (sq. km).jpg')
