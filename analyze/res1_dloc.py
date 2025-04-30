import matplotlib.pyplot as plt
d = {
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
ax.set_title('Verification of MNIST-OD')
plt.xticks(rotation=90, ha='right')


# Display the legend with unique entries
ax.legend(title='Categories')

# Display the plot
plt.show()
plt.savefig('res1_dloc.png')
plt.close()