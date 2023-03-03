# import pandas , numpy and matplotlib modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#line chart

# Read the data from the CSV file into a pandas DataFrame
df=pd.read_csv(r"C:\Users\U.A AJEESH\Desktop\data science\herts\applied ds 1\assignment\top5c-2021.csv")

# Group the dataset by company
grouped_df = df.groupby("company")

# Set the figure size
plt.figure(figsize=(15, 7))

# Loop through each company and create a line plot for their sales by month
for company, data in grouped_df:
    plt.plot(data["month"], data["cars sold"], label=company)

# Set the x-axis label and the size of the label
plt.xlabel("MONTH", size=13)

# Set the y-axis label and the size of the label
plt.ylabel("NO OF CARS SOLD", size=13)

# Set the title of the plot and the size of the title
plt.title("CAR SALE VOLUME OF TOP 5 COMPANIES IN INDIA(2021))", size=15)

# Add a grid to the plot
plt.grid(True)

# Add a legend to the plot
plt.legend()

# Save the chart to a file as a PNG file
plt.savefig("line_chart.png")

# Show the plot
plt.show()

#pie chart

# Read in the dataset
df1=pd.read_csv(r"C:\Users\U.A AJEESH\Desktop\data science\herts\applied ds 1\assignment\car sales annual 2020-2021.csv")

# list of company name
company = list(df1['Company'].unique())

# list of no of car sold by the each companies in 2020
year2020=list(df1['2020'])

# list of no of car sold by the each companies in 2021
year2021=list(df1['2021'])

# create subplots with 1 row and 2 columns and set the size of the figure
fig, ax = plt.subplots(1, 2, figsize=(15, 20))

# plot first pie chart on first subplot
ax[0].pie(year2020, labels=company, autopct='%1.1f%%')
ax[0].set_title('2020', size=15)

# plot second pie chart on second subplot
ax[1].pie(year2021, labels=company, autopct='%1.1f%%')
ax[1].set_title('2021', size=15)

# Add the main title and set the size and position of main title
plt.suptitle("Market Share of Indian Car Companies (2020 & 2021)", y=.7,  size=17)

# Save the chart to a file as a PNG file
plt.savefig("pie_chart.png")

# Display the chart
plt.show()

#bar chart

# set bar width
width = 0.35

# set x coordinates for bars
x = np.arange(len(df1['Company']))

# create bar chart
fig, ax = plt.subplots(figsize=(18, 1.5))
rects1 = ax.bar(x - width/2, year2020, width, label='2020')
rects2 = ax.bar(x + width/2, year2021, width, label='2021')

# add labels, title, and legend
ax.set_ylabel('Cars sold', size=15)
ax.set_xlabel('Companies', size=15)
ax.set_xticks(x)
ax.set_xticklabels(company)
fig.subplots_adjust(top=5)
plt.title("Annual sale figures of Indian Car Companies (2020vs2021)", size=18)
ax.legend()

# Save the chart to a file as a PNG file
plt.savefig("bar_chart.png")

plt.show()