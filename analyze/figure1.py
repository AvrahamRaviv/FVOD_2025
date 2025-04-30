import matplotlib.pyplot as plt

# set times new roman
plt.rcParams['font.family'] = 'Times New Roman'


gtsrb_us = {
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

dloc_us = {
    "0.1": {
        "unsafe-pgd": 100
    },
    "0.075": {
        "unsafe-pgd": 100
    },
    "0.05": {
        "unsafe-pgd": 100
    },
    "0.025": {
        "unsafe-pgd": 100
    },
    "0.01": {
        "unsafe-pgd": 97,
        "unknown": 3
    },
    "0.0075": {
        "unsafe-pgd": 82,
        "unknown": 18
    },
    "0.005": {
        "unsafe-pgd": 29,
        "unknown": 71
    },
    "0.0025": {
        "unsafe-pgd": 3,
        "unknown": 97
    },
    "0.001": {
        "unknown": 100
    },
    "0.00075": {
        "unknown": 100
    },
    "0.0005": {
        "unknown": 100
    },
    "0.00025": {
        "unknown": 100
    },
    "0.0002": {
        "unknown": 100
    },
    "0.00015": {
        "unknown": 98,
        "safe-incomplete": 2
    },
    "0.0001": {
        "unknown": 48,
        "safe-incomplete": 52
    },
    "9e-05": {
        "unknown": 34,
        "safe-incomplete": 66
    },
    "8e-05": {
        "unknown": 1,
        "safe-incomplete": 99
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
    }
}

lard_us = {
    "0.1": {
        "unsafe-pgd": 100
    },
    "0.075": {
        "unsafe-pgd": 100
    },
    "0.05": {
        "unsafe-pgd": 100
    },
    "0.025": {
        "unsafe-pgd": 100
    },
    "0.01": {
        "unsafe-pgd": 100
    },
    "0.0075": {
        "unsafe-pgd": 100
    },
    "0.005": {
        "unknown": 47,
        "unsafe-pgd": 53
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
        "unknown": 100
    },
    "0.00025": {
        "unknown": 100
    },
    "0.0001": {
        "unknown": 96,
        "safe-incomplete": 4
    },
    "7.5e-05": {
        "safe-incomplete": 46,
        "unknown": 54
    },
    "5e-05": {
        "safe-incomplete": 96,
        "unknown": 4
    },
    "2.5e-05": {
        "safe-incomplete": 100
    },
    "1e-05": {
        "safe-incomplete": 100
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
    if 'safe-incomplete' in dloc_us[str(i)].keys():
        dloc_s[str(i)]['safe-incomplete'] = dloc_us[str(i)]['safe-incomplete']

gtsrb_s = {}
s_keys = sorted([float(i) for i in gtsrb_us.keys()], reverse=True)
for i in s_keys:
    gtsrb_s[str(i)] = {}
    if 'unsafe-pgd' in gtsrb_us[str(i)].keys():
        gtsrb_s[str(i)]['unsafe-pgd'] = gtsrb_us[str(i)]['unsafe-pgd']
    if 'unknown' in gtsrb_us[str(i)].keys():
        gtsrb_s[str(i)]['unknown'] = gtsrb_us[str(i)]['unknown']
    if 'safe-incomplete' in gtsrb_us[str(i)].keys():
        gtsrb_s[str(i)]['safe-incomplete'] = gtsrb_us[str(i)]['safe-incomplete']

lard_s = {}
s_keys = sorted([float(i) for i in lard_us.keys()], reverse=True)
for i in s_keys:
    lard_s[str(i)] = {}
    if 'unsafe-pgd' in lard_us[str(i)].keys():
        lard_s[str(i)]['unsafe-pgd'] = lard_us[str(i)]['unsafe-pgd']
    if 'unknown' in lard_us[str(i)].keys():
        lard_s[str(i)]['unknown'] = lard_us[str(i)]['unknown']
    if 'safe-incomplete' in lard_us[str(i)].keys():
        lard_s[str(i)]['safe-incomplete'] = lard_us[str(i)]['safe-incomplete']


# Define the colors for each category
colors = {
    'unsafe-pgd': 'Red',
    'unknown': 'Yellow',
    'safe-incomplete': 'Green'
}


# plot 3x1
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
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
axs[0].set_xlabel('Epsilon', fontsize=20)
axs[0].set_ylabel('# samples', fontsize=30)
# axs[0].set_title('Verification of MNIST-OD')
# rotate x-axis labels by 90. values are the keys of the dictionary

axs[0].set_xticks(range(len(dloc_s)))
axs[0].set_xticklabels(dloc_s.keys(), rotation=90, ha='right')
# Display the legend with unique entries
# axs[0].legend(title='Categories')

# plot gtsrb
# To keep track of categories already added to the legend
# To keep track of categories already added to the legend
legend_labels = set()
axs[1].set_title('GTSRB', fontsize=20)
# Iterate through the dictionary to plot bars
for i, (key, value) in enumerate(gtsrb_s.items()):
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

axs[1].set_xlabel('Epsilon')
# axs[1].set_ylabel('# samples', fontsize=20)
# axs[1].set_title('Verification of GTSRB')
axs[1].set_xticks(range(len(gtsrb_s)))
axs[1].set_xticklabels(gtsrb_s.keys(), rotation=90, ha='right')
# axs[1].legend(title='Categories')

# plot lard
axs[2].set_title('LARD', fontsize=20)
# To keep track of categories already added to the legend
legend_labels = set()
# Iterate through the dictionary to plot bars
for i, (key, value) in enumerate(lard_s.items()):
    bottom = 0  # Initial bottom position for stacking
    for category, count in value.items():
        # Compute the height of the segment
        height = count / total_height * 100
        # Check if category label is already in the legend set
        label = category if category not in legend_labels else ""
        # Plot the bar segment
        axs[2].bar(key, height, bottom=bottom, color=colors.get(category, 'gray'), label=label)
        # Update the bottom position
        bottom += height
        # Add category to legend set
        legend_labels.add(category)

# Add labels and title
axs[2].set_xlabel('Epsilon')
# axs[2].set_ylabel('# samples')
axs[2].set_xticks(range(len(lard_s)))
axs[2].set_xticklabels(lard_s.keys(), rotation=90, ha='right')
# axs[2].legend(title='Categories')

# add legend in bottom of all subplots (in the middle, means in 1, 1)
# plot it instead of the 1,1 subplot, and keep 1,0 and 1,2 empty (delete all their content)
# axs[1][0].axis('off')
# axs[1][1].axis('off')
# axs[1][2].axis('off')
# # Display the legend with unique entries
# axs[1][1].legend(title='Categories', loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

# save and close
plt.savefig('res1.png')
plt.savefig('res1.pdf')
plt.close()