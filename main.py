import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset/user_behavior_dataset.csv')


# Screen time based on age group
screen_time_by_age = df.groupby('Age')['Screen On Time (hours/day)'].mean()

ages = screen_time_by_age.index
values = screen_time_by_age.values
top_values = 3 # number of top values to be displayed
top_indices = np.sort(values)[-top_values]

# Trend Line
trend_line = np.polyfit(ages, values, 1)
p = np.poly1d(trend_line)

plt.figure(figsize=(10, 5))

used_labels = set()

for age, value in zip(ages,values):
    if value >= top_indices:
        table_color='tab:red'
        legend_label='Top 3 highest values'
    else:
        table_color='tab:blue'
        legend_label='Other values'

    if legend_label in used_labels:
        legend_label = '_nolegend_'
    else:
        used_labels.add(legend_label)
    plt.bar(age, value, color=table_color, label=legend_label)

plt.plot(ages, p(ages), "--", color='black', label='Trend Line')
plt.xlabel('Age')
plt.ylabel('Average screen time (hours/day)')
plt.title('Average screen time per age')
plt.legend()
plt.show()

#---------------------------------------------------------------#


