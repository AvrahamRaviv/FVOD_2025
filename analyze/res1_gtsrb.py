import matplotlib.pyplot as plt

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

# sort d by keys
d = {}
s_keys = sorted([float(i) for i in d_us.keys()], reverse=True)
for i in s_keys:
    d[str(i)] = {}
    if 'unsafe-pgd' in d_us[str(i)].keys():
        d[str(i)]['unsafe-pgd'] = d_us[str(i)]['unsafe-pgd']
    if 'unknown' in d_us[str(i)].keys():
        d[str(i)]['unknown'] = d_us[str(i)]['unknown']
    if 'safe-incomplete' in d_us[str(i)].keys():
        d[str(i)]['safe-incomplete'] = d_us[str(i)]['safe-incomplete']

# Define the colors for each category
colors = {
    'unsafe-pgd': 'Red',
    'unknown': 'Yellow',
    'safe-incomplete': 'Green'
}

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set the total height of each bar
total_height = 100

# To keep track of categories already added to the legend
legend_labels = set()

# Iterate through the dictionary to plot bars
for i, (key, value) in enumerate(d.items()):
    bottom = 0  # Initial bottom position for stacking
    for category, count in value.items():
        # Compute the height of the segment
        height = count / total_height * 100
        # Check if category label is already in the legend set
        label = category if category not in legend_labels else ""
        # Plot the bar segment
        ax.bar(key, height, bottom=bottom, color=colors.get(category, 'gray'), label=label)
        # Update the bottom position
        bottom += height
        # Add category to legend set
        legend_labels.add(category)

# Add labels and title
ax.set_xlabel('Epsilon')
ax.set_ylabel('# samples')
ax.set_title('Verification of GTSRB')
plt.xticks(rotation=90, ha='right')

# Display the legend with unique entries
ax.legend(title='Categories')

# Display the plot
plt.show()
plt.savefig('res1_gtsrb.pdf')
plt.close()
