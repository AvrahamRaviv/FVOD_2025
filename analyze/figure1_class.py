import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

dloc_us = {
    "0.1": {
        "unsafe-pgd": 100,
        "unknown": 0,
        "safe": 0
    },
    "0.075": {
        "unsafe-pgd": 100,
        "unknown": 0,
        "safe": 0
    },
    "0.05": {
        "unsafe-pgd": 100,
        "unknown": 0,
        "safe": 0
    },
    "0.025": {
        "unsafe-pgd": 100,
        "unknown": 0,
        "safe": 0
    },
    "0.01": {
        "unsafe-pgd": 98,
        "unknown": 2,
        "safe": 0
    },
    "0.0075": {
        "unsafe-pgd": 97,
        "unknown": 3,
        "safe": 0
    },
    "0.005": {
        "unsafe-pgd": 80,
        "unknown": 20,
        "safe": 0
    },
    "0.0025": {
        "unsafe-pgd": 39,
        "unknown": 61,
        "safe": 0
    },
    "0.001": {
        "unsafe-pgd": 6,
        "unknown": 90,
        "safe": 4
    },
    "0.00075": {
        "unsafe-pgd": 8,
        "unknown": 64,
        "safe": 28
    },
    "0.0005": {
        "unsafe-pgd": 2,
        "unknown": 39,
        "safe": 59
    },
    "0.00025": {
        "unsafe-pgd": 2,
        "unknown": 9,
        "safe": 89
    },
    "0.0001": {
        "unsafe-pgd": 0,
        "unknown": 6,
        "safe": 94
    },
    "7.5e-05": {
        "unsafe-pgd": 0,
        "unknown": 3,
        "safe": 97
    },
    "5e-05": {
        "unsafe-pgd": 1,
        "unknown": 3,
        "safe": 96
    },
    "2.5e-05": {
        "unsafe-pgd": 0,
        "unknown": 1,
        "safe": 99
    },
    "1e-05": {
        "unsafe-pgd": 0,
        "unknown": 0,
        "safe": 100
    }
}

# sort d by keys
dloc_s = {}
s_keys = sorted([float(i) for i in dloc_us.keys()], reverse=True)
for i in s_keys:
    dloc_s[str(i)] = {}
    if 'unsafe-pgd' in dloc_us[str(i)].keys():
        dloc_s[str(i)]['unsafe-pgd'] = dloc_us[str(i)]['unsafe-pgd']
    if 'unknown' in dloc_us[str(i)].keys():
        dloc_s[str(i)]['unknown'] = dloc_us[str(i)]['unknown']
    if 'safe' in dloc_us[str(i)].keys():
        dloc_s[str(i)]['safe-incomplete'] = dloc_us[str(i)]['safe']

GTSRB_us = {
    "0.1": {
        "unsafe-pgd": 100,
        "unknown": 0,
        "safe": 0
    },
    "0.075": {
        "unsafe-pgd": 100,
        "unknown": 0,
        "safe": 0
    },
    "0.05": {
        "unsafe-pgd": 100,
        "unknown": 0,
        "safe": 0
    },
    "0.025": {
        "unsafe-pgd": 100,
        "unknown": 0,
        "safe": 0
    },
    "0.01": {
        "unsafe-pgd": 91,
        "unknown": 9,
        "safe": 0
    },
    "0.0075": {
        "unsafe-pgd": 78,
        "unknown": 22,
        "safe": 0
    },
    "0.005": {
        "unsafe-pgd": 63,
        "unknown": 37,
        "safe": 0
    },
    "0.0025": {
        "unsafe-pgd": 40,
        "unknown": 50,
        "safe": 10
    },
    "0.001": {
        "unsafe-pgd": 18,
        "unknown": 27,
        "safe": 55
    },
    "0.00075": {
        "unsafe-pgd": 17,
        "unknown": 15,
        "safe": 68
    },
    "0.0005": {
        "unsafe-pgd": 14,
        "unknown": 7,
        "safe": 79
    },
    "0.00025": {
        "unsafe-pgd": 6,
        "unknown": 7,
        "safe": 87
    },
    "0.0001": {
        "unsafe-pgd": 3,
        "unknown": 1,
        "safe": 96
    },
    "7.5e-05": {
        "unsafe-pgd": 3,
        "unknown": 0,
        "safe": 97
    },
    "5e-05": {
        "unsafe-pgd": 3,
        "unknown": 0,
        "safe": 97
    },
    "2.5e-05": {
        "unsafe-pgd": 2,
        "unknown": 0,
        "safe": 98
    },
    "1e-05": {
        "unsafe-pgd": 1,
        "unknown": 0,
        "safe": 99
    }
}

# sort d by keys
GTSRB_s = {}
s_keys = sorted([float(i) for i in GTSRB_us.keys()], reverse=True)
for i in s_keys:
    GTSRB_s[str(i)] = {}
    if 'unsafe-pgd' in GTSRB_us[str(i)].keys():
        GTSRB_s[str(i)]['unsafe-pgd'] = GTSRB_us[str(i)]['unsafe-pgd']
    if 'unknown' in GTSRB_us[str(i)].keys():
        GTSRB_s[str(i)]['unknown'] = GTSRB_us[str(i)]['unknown']
    if 'safe' in GTSRB_us[str(i)].keys():
        GTSRB_s[str(i)]['safe-incomplete'] = GTSRB_us[str(i)]['safe']


# Define the colors for each category
colors = {
    'unsafe-pgd': 'Red',
    'unknown': 'Yellow',
    'safe-incomplete': 'Green'
}


# plot 3x1
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
total_height = 100

# plot dloc
axs[0].set_title('MNIST-OD', fontsize=20)
# Set the total height of each bar

# To keep track of categories already added to the legend
legend_labels = set()

# Iterate through the dictionary to plot bars
for i, (key, value) in enumerate(dloc_s.items()):
    bottom = 0  # Initial bottom position for stacking
    for category, count in value.items():
        # Compute the height of the segment
        height = count / total_height * 100
        # Check if category label is already in the legend set
        label = category if category not in legend_labels else ""
        # Plot the bar segment
        axs[0].bar(key, height, bottom=bottom, color=colors.get(category, 'gray'), label=label)
        # Update the bottom position
        bottom += height
        # Add category to legend set
        legend_labels.add(category)

# Add labels and title
axs[0].set_xlabel('Epsilon', fontsize=30)
axs[0].set_ylabel('# samples', fontsize=30)
# axs[0].set_title('Verification of MNIST-OD')
# rotate x-axis labels by 90. values are the keys of the dictionary

axs[0].set_xticks(range(len(dloc_s)))
axs[0].set_xticklabels(dloc_s.keys(), rotation=90, ha='right')
# Display the legend with unique entries
# axs[0].legend(title='Categories')

# plot GTSRB
axs[1].set_title('GTSRB', fontsize=20)
# Set the total height of each bar
legend_labels = set()

# Iterate through the dictionary to plot bars
for i, (key, value) in enumerate(GTSRB_s.items()):
    bottom = 0  # Initial bottom position for stacking
    for category, count in value.items():
        # Compute the height of the segment
        height = count / total_height * 100
        # Check if category label is already in the legend set
        label = category if category not in legend_labels else ""
        # Plot the bar segment
        axs[1].bar(key, height, bottom=bottom, color=colors.get(category, 'gray'), label=label)
        # Update the bottom position
        bottom += height
        # Add category to legend set
        legend_labels.add(category)

# Add labels and title
axs[1].set_xlabel('Epsilon', fontsize=30)
# axs[1].set_ylabel('# samples')
# axs[1].set_title('Verification of GTSRB')
# rotate x-axis labels by 90. values are the keys of the dictionary

axs[1].set_xticks(range(len(GTSRB_s)))
axs[1].set_xticklabels(GTSRB_s.keys(), rotation=90, ha='right')

# save and close
plt.savefig('res2.png')
plt.savefig('res2.pdf')
plt.close()