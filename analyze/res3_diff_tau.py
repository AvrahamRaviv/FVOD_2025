import matplotlib.pyplot as plt
import numpy as np


d_us = {
    "0.1": {
        "unsafe-pgd": 100
    },
    "0.075": {
        "unsafe-pgd": 95,
        "unknown": 5
    },
    "0.05": {
        "unknown": 10,
        "unsafe-pgd": 90
    },
    "0.025": {
        "unknown": 51,
        "unsafe-pgd": 49
    },
    "0.01": {
        "unknown": 96,
        "unsafe-pgd": 4
    },
    "0.0075": {
        "unknown": 99,
        "unsafe-pgd": 1
    },
    "0.005": {
        "unknown": 100
    },
    "0.0025": {
        "unknown": 100
    },
    "0.001": {
        "unknown": 100
    },
    "0.00075": {
        "unknown": 100
    },
    "0.0005": {
        "unknown": 99,
        "safe-incomplete": 1
    },
    "0.00025": {
        "safe-incomplete": 52,
        "unknown": 48
    },
    "0.0001": {
        "safe-incomplete": 100
    },
    "7.5e-05": {
        "safe-incomplete": 100
    },
    "5e-05": {
        "safe-incomplete": 100
    },
    "2.5e-05": {
        "safe-incomplete": 100
    },
    "1e-05": {
        "safe-incomplete": 100
    },
    "0.007": {
        "unknown": 99,
        "unsafe-pgd": 1
    },
    "0.006": {
        "unknown": 100
    },
    "0.004": {
        "unknown": 100
    },
    "0.003": {
        "unknown": 100
    },
    "9e-05": {
        "safe-incomplete": 100
    },
    "8e-05": {
        "safe-incomplete": 100
    },
    "7e-05": {
        "safe-incomplete": 100
    },
    "6e-05": {
        "safe-incomplete": 100
    }
}

GTSRB_us_tau_05 = {}
s_keys = sorted([float(i) for i in d_us.keys()], reverse=True)
for i in s_keys:
    GTSRB_us_tau_05[str(i)] = {}
    if 'unsafe-pgd' in d_us[str(i)].keys():
        GTSRB_us_tau_05[str(i)]['unsafe-pgd'] = d_us[str(i)]['unsafe-pgd']
    if 'unknown' in d_us[str(i)].keys():
        GTSRB_us_tau_05[str(i)]['unknown'] = d_us[str(i)]['unknown']
    if 'safe-incomplete' in d_us[str(i)].keys():
        GTSRB_us_tau_05[str(i)]['safe-incomplete'] = d_us[str(i)]['safe-incomplete']

d_us = {
    "0.5": {
        "unsafe-pgd": 99,
        "unknown": 1
    },
    "0.1": {
        "unsafe-pgd": 42,
        "unknown": 58
    },
    "0.075": {
        "unsafe-pgd": 41,
        "unknown": 59
    },
    "0.05": {
        "unknown": 68,
        "unsafe-pgd": 32
    },
    "0.025": {
        "unknown": 90,
        "unsafe-pgd": 10
    },
    "0.01": {
        "unknown": 100
    },
    "0.0075": {
        "unknown": 100
    },
    "0.005": {
        "unknown": 100
    }
}

GTSRB_us_tau_01 = {}
s_keys = sorted([float(i) for i in d_us.keys()], reverse=True)
for i in s_keys:
    GTSRB_us_tau_01[str(i)] = {}
    if 'unsafe-pgd' in d_us[str(i)].keys():
        GTSRB_us_tau_01[str(i)]['unsafe-pgd'] = d_us[str(i)]['unsafe-pgd']
    if 'unknown' in d_us[str(i)].keys():
        GTSRB_us_tau_01[str(i)]['unknown'] = d_us[str(i)]['unknown']
    if 'safe-incomplete' in d_us[str(i)].keys():
        GTSRB_us_tau_01[str(i)]['safe-incomplete'] = d_us[str(i)]['safe-incomplete']

d_us = {
    "0.5": {
        "unsafe-pgd": 100
    },
    "0.1": {
        "unsafe-pgd": 89,
        "unknown": 11
    },
    "0.075": {
        "unknown": 14,
        "unsafe-pgd": 86
    },
    "0.05": {
        "unknown": 38,
        "unsafe-pgd": 62
    },
    "0.025": {
        "unknown": 70,
        "unsafe-pgd": 30
    },
    "0.01": {
        "unknown": 99,
        "unsafe-pgd": 1
    }
}

GTSRB_us_tau_03 = {}
s_keys = sorted([float(i) for i in d_us.keys()], reverse=True)
for i in s_keys:
    GTSRB_us_tau_03[str(i)] = {}
    if 'unsafe-pgd' in d_us[str(i)].keys():
        GTSRB_us_tau_03[str(i)]['unsafe-pgd'] = d_us[str(i)]['unsafe-pgd']
    if 'unknown' in d_us[str(i)].keys():
        GTSRB_us_tau_03[str(i)]['unknown'] = d_us[str(i)]['unknown']
    if 'safe-incomplete' in d_us[str(i)].keys():
        GTSRB_us_tau_03[str(i)]['safe-incomplete'] = d_us[str(i)]['safe-incomplete']

# Epsilon values
tau = [0.5, 0.3, 0.1]

total_height = 100

# Combine dictionaries into a list
all_keys = list(GTSRB_us_tau_05.keys())

# create filtered arrays
cleaned_data = {}
cleaned_data["0.5"] = {0.45: GTSRB_us_tau_05[all_keys[0]], 0.5: GTSRB_us_tau_05[all_keys[1]], 0.55: GTSRB_us_tau_05[all_keys[2]]}
cleaned_data["0.3"] = {0.25: GTSRB_us_tau_03[all_keys[0]], 0.3: GTSRB_us_tau_03[all_keys[1]], 0.35: GTSRB_us_tau_03[all_keys[2]]}
cleaned_data["0.1"] = {0.05: GTSRB_us_tau_01[all_keys[0]], 0.1: GTSRB_us_tau_01[all_keys[1]], 0.15: GTSRB_us_tau_01[all_keys[2]]}


# Define the colors for each category
colors = {
    'unsafe-pgd': 'red',
    'unknown': 'yellow',
    'safe-incomplete': 'green'
}

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# To keep track of categories already added to the legend
legend_labels = set()

# Iterate through the dictionary to plot bars
for t in tau:
    for key in cleaned_data[str(t)].keys():
        bottom = 0
        for category, count in cleaned_data[str(t)][key].items():
            # Compute the height of the segment
            height = count / total_height * 100
            # Check if category label is already in the legend set
            label = category if category not in legend_labels else ""
            # Plot the bar segment, size of each bar 0.005
            ax.bar(key, height, color=colors[category], label=label, bottom=bottom, width=0.035)
            # Update the bottom position
            bottom += height
            # Add category to legend set
            legend_labels.add(category)

# Add labels and title
ax.set_xlabel('Epsilon')
ax.set_ylabel('# samples')
ax.set_title('Verification of GTSRB')
# set x ticks to [0.1, 0.3, 0.5], delete the line between the axis and the ticks
ax.set_xticks([0.1, 0.3, 0.5])
ax.tick_params(axis='x', which='both', bottom=False)

# Display the legend with unique entries
ax.legend(title='Categories')

# Display the plot
plt.show()
plt.savefig('res3_diff_tau_GTSRB.png')
plt.close()
