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

# plt.plot(ages, p(ages), "--", color='black', label='Trend Line')
# plt.xlabel('Age')
# plt.ylabel('Average screen time (hours/day)')
# plt.title('Average screen time per age')
# plt.legend()
# plt.show()

#---------------------------------------------------------------#

# How fast battery drains for each Phone model (and Operating System)

battery_by_os_model = (
    df.groupby(['Device Model', 'Operating System'])['Battery Drain (mAh/day)']
      .mean()
      .sort_values(ascending=False)
)

# compare OS and model battery performance while using apps
app_usage_threshold = df['App Usage Time (min/day)'].median()
using_apps = df[df['App Usage Time (min/day)'] > app_usage_threshold]
battery_using_apps_by_os_and_model = (
    using_apps
    .groupby(['Device Model', 'Operating System'])['Battery Drain (mAh/day)']
    .mean()
    .sort_values(ascending=False)
)

low_app_users = df[df['App Usage Time (min/day)'] <= app_usage_threshold]
battery_low = (
    low_app_users.groupby(['Device Model', 'Operating System'])['Battery Drain (mAh/day)'].mean()
)

battery_high = battery_using_apps_by_os_and_model

compare_df = pd.concat(
    [battery_low, battery_high],
    axis=1,
    keys=['Low App Usage', 'High App Usage']
)

compare_df.plot(kind='bar', figsize=(14,6))
plt.ylabel('Battery Drain (mAh/day)')
plt.title('Battery Drain by OS and Model (high vs low app usage)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()