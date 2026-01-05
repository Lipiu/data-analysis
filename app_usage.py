import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

df = pd.read_csv('dataset/user_behavior_dataset.csv')

apps_installed_by_age = df.groupby('Age').agg({
    'Number of Apps Installed' : 'mean',
    'Data Usage (MB/day)' : 'mean'
})

values = apps_installed_by_age['Number of Apps Installed']
top_n = 3
top_values = values.nlargest(top_n)
colors = []
legend_labels = [
    Patch(facecolor='tab:red', label=f'Top {top_n} highest values'),
    Patch(facecolor='tab:blue', label='Other values')
]

for v in values:
    if v in top_values.values:
        colors.append('tab:red')
    else:
        colors.append('tab:blue')

plt.figure(figsize=(10,5))
plt.bar(
    apps_installed_by_age.index,
    values,
    color=colors
)

plt.xlabel('Age')
plt.ylabel('Average Number of Apps Installed')
plt.title('Average Number of Apps Installed by Age')
plt.legend(handles=legend_labels)
plt.tight_layout()
plt.show()

# Younger age groups tend to install more applications on average, 
# while older age groups show a gradual decline in the number of installed apps.

#-----------------------------#
#Data usage related to age

data_values = apps_installed_by_age['Data Usage (MB/day)']
top_data_n = 3
top_data_values = data_values.nlargest(top_data_n)
data_colors = []
legend_labels_data = [
    Patch(facecolor='tab:red', label=f'Top {top_data_n} highest values'),
    Patch(facecolor='tab:blue', label='Other values')
]

for v in data_values:
    if v in top_data_values.values:
        data_colors.append('tab:red')
    else:
        data_colors.append('tab:blue')
plt.figure(figsize=(10,5))
plt.bar(
    apps_installed_by_age.index,
    data_values,
    color=data_colors
)


plt.xlabel('Age')
plt.ylabel('Average Data Usage (MB/day)')
plt.title('Data Usage in relation to age')
plt.legend(handles=legend_labels_data)
plt.tight_layout()
plt.show()