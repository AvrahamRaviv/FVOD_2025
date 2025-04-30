import matplotlib.pyplot as plt

# For each dataset we take 3 different tau values - [0.5, 0.6, 0.7]
# For each tua we take 3 different epsilon values - [1e-2, 1e-3, 1e-4]
# For each epsilon we have 3 different categories - ['unsafe-pgd', 'unknown', 'safe-incomplete']
# For each epsilon we give new x value - tau + [-0.05, 0, 0.05]
GTSRB_dict = {'0.5': {0.01: {"unknown": 96, "unsafe-pgd": 4}, 0.001: {"unknown": 100}, 0.0001: {"safe-incomplete": 100}}, '0.6': {0.01: {"unknown": 72, "unsafe-pgd": 28}, 0.001: {"unknown": 100}, 0.0001: {"unknown": 16, "safe-incomplete": 84}}, '0.7': {0.01: {"unknown": 42, "unsafe-pgd": 58}, 0.001: {"unknown": 100}, 0.0001: {"safe-incomplete": 55, "unknown": 45}}}
DLOC_dict = {'0.5': {0.01: {"unknown": 97, "unsafe-pgd": 3}, 0.001: {"unknown": 100}, 0.0001: {"unknown": 48, "safe-incomplete": 52}}, '0.6': {0.01: {"unknown": 1, "unsafe-pgd": 99}, 0.001: {"unknown": 98, "unsafe-pgd": 2}, 0.0001: {"unknown": 91, "safe-incomplete": 8, "unsafe-pgd": 1}}, '0.7': {0.01: {"unsafe-pgd": 100}, 0.001: {"unknown": 87, "unsafe-pgd": 13}, 0.0001: {"unsafe-pgd": 3, "unknown": 97}}}
LARD_dict = {'0.5': {0.01: {"unsafe-pgd": 100}, 0.001: {"unknown": 100}, 0.0001: {"unknown": 96, "safe-incomplete": 4}}, '0.6': {0.01: {"unsafe-pgd": 100}, 0.001: {"unknown": 100}, 0.0001: {"unknown": 100}}, '0.7': {0.01: {"unknown": 100}, 0.001: {"unknown": 100}, 0.0001: {"safe-incomplete": 100}}}

# For each dataset, for each tau, for each epsilon, sort by the order of: 'unsafe-pgd', 'unknown', 'safe-incomplete'
for dataset in [GTSRB_dict, DLOC_dict, LARD_dict]:
    for tau in ['0.5', '0.6', '0.7']:
        for epsilon in [0.01, 0.001, 0.0001]:
            dataset[tau][epsilon] = dict(sorted(dataset[tau][epsilon].items(), key=lambda item: ['unsafe-pgd', 'unknown', 'safe-incomplete'].index(item[0])))

# Tau values
tau = [0.5, 0.6, 0.7]
epsilons = ["1e-2", "1e-3", "1e-4"]
total_height = 100
# Define the colors for each category
colors = {
    'unsafe-pgd': 'red',
    'unknown': 'yellow',
    'safe-incomplete': 'green'
}
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# plot GTSRB
axs[0].set_title('GTSRB', fontsize=14)

legend_labels = set()

# Iterate through the dictionary to plot bars
for t in tau:
    for key in GTSRB_dict[str(t)].keys():
        bottom = 0
        for category, count in GTSRB_dict[str(t)][key].items():
            # Compute the height of the segment
            height = count / total_height * 100
            # Check if category label is already in the legend set
            label = category if category not in legend_labels else ""
            # Plot the bar segment, size of each bar 0.005
            x_val = t + [-0.025, 0, 0.025][list(GTSRB_dict[str(t)].keys()).index(key)]
            axs[0].bar(x_val, height, color=colors[category], label=label, bottom=bottom, width=0.02)
            # Update the bottom position
            bottom += height
            # Add category to legend set
            legend_labels.add(category)
        # write above the bar the value of epsilon (current key)
        eps2write = epsilons[list(GTSRB_dict[str(t)].keys()).index(key)]
        x_val = t + [-0.025, 0, 0.025][list(GTSRB_dict[str(t)].keys()).index(key)]
        axs[0].text(x_val, 75, str(eps2write), ha='center', va='bottom', fontsize=8)

# Add labels and title
axs[0].set_xlabel('Tau', fontsize=20)
axs[0].set_ylabel('# samples', fontsize=20)
axs[0].set_title('Verification of GTSRB', fontsize=14)
# set x ticks to [0.5, 0.6, 0.7], delete the line between the axis and the ticks
axs[0].set_xticks([0.5, 0.6, 0.7])
axs[0].tick_params(axis='x', which='both', bottom=False)

# plot DLOC
axs[1].set_title('DLOC', fontsize=14)

legend_labels = set()

# Iterate through the dictionary to plot bars
for t in tau:
    for key in DLOC_dict[str(t)].keys():
        bottom = 0
        for category, count in DLOC_dict[str(t)][key].items():
            # Compute the height of the segment
            height = count / total_height * 100
            # Check if category label is already in the legend set
            label = category if category not in legend_labels else ""
            # Plot the bar segment, size of each bar 0.005
            x_val = t + [-0.025, 0, 0.025][list(GTSRB_dict[str(t)].keys()).index(key)]
            axs[1].bar(x_val, height, color=colors[category], label=label, bottom=bottom, width=0.02)
            # Update the bottom position
            bottom += height
            # Add category to legend set
            legend_labels.add(category)
        # write above the bar the value of epsilon (current key)
        eps2write = epsilons[list(GTSRB_dict[str(t)].keys()).index(key)]
        x_val = t + [-0.025, 0, 0.025][list(GTSRB_dict[str(t)].keys()).index(key)]
        axs[1].text(x_val, 75, str(eps2write), ha='center', va='bottom', fontsize=8)

# Add labels and title
axs[1].set_xlabel('Tau', fontsize=20)
axs[1].set_ylabel('# samples', fontsize=20)
axs[1].set_title('Verification of DLOC', fontsize=14)
# set x ticks to [0.5, 0.6, 0.7], delete the line between the axis and the ticks
axs[1].set_xticks([0.5, 0.6, 0.7])
axs[1].tick_params(axis='x', which='both', bottom=False)

# Display the legend with unique entries
# axs[1].legend(title='Categories')

# plot LARD
axs[2].set_title('LARD', fontsize=14)

legend_labels = set()

# Iterate through the dictionary to plot bars
for t in tau:
    for key in LARD_dict[str(t)].keys():
        bottom = 0
        for category, count in LARD_dict[str(t)][key].items():
            # Compute the height of the segment
            height = count / total_height * 100
            # Check if category label is already in the legend set
            label = category if category not in legend_labels else ""
            # Plot the bar segment, size of each bar 0.005
            x_val = t + [-0.025, 0, 0.025][list(GTSRB_dict[str(t)].keys()).index(key)]
            axs[2].bar(x_val, height, color=colors[category], label=label, bottom=bottom, width=0.02)
            # Update the bottom position
            bottom += height
            # Add category to legend set
            legend_labels.add(category)
        # write above the bar the value of epsilon (current key)
        eps2write = epsilons[list(GTSRB_dict[str(t)].keys()).index(key)]
        x_val = t + [-0.025, 0, 0.025][list(GTSRB_dict[str(t)].keys()).index(key)]
        axs[2].text(x_val, 75, str(eps2write), ha='center', va='bottom', fontsize=8)
        
# Add labels and title
axs[2].set_xlabel('Tau', fontsize=20)
axs[2].set_ylabel('# samples', fontsize=20)
axs[2].set_title('Verification of LARD', fontsize=14)
# set x ticks to [0.5, 0.6, 0.7], delete the line between the axis and the ticks
axs[2].set_xticks([0.5, 0.6, 0.7])
axs[2].tick_params(axis='x', which='both', bottom=False)

# Display the plot
plt.show()
plt.savefig('res3_diff_tau.pdf')
plt.savefig('res3_diff_tau.png')
plt.close()