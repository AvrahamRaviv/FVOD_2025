import matplotlib.pyplot as plt

# set times new roman
plt.rcParams['font.family'] = 'Times New Roman'

# For each dataset we take 3 different tau values - [0.5, 0.6, 0.7]
# For each tua we take 3 different epsilon values - [1e-2, 1e-3, 1e-4]
# For each epsilon we have 3 different categories - ['unsafe-pgd', 'unknown', 'safe-incomplete']
# For each epsilon we give new x value - tau + [-0.05, 0, 0.05]
GTSRB_dict = {
    '0.5': {0.01: {"unknown": 96, "unsafe-pgd": 4}, 0.001: {"unknown": 100}, 0.0001: {"safe-incomplete": 100}},
    '0.6': {0.01: {"unknown": 72, "unsafe-pgd": 28}, 0.001: {"unknown": 100},
            0.0001: {"unknown": 16, "safe-incomplete": 84}},
    '0.7': {0.01: {"unknown": 42, "unsafe-pgd": 58}, 0.001: {"unknown": 100},
            0.0001: {"safe-incomplete": 55, "unknown": 45}}}
DLOC_dict = {'0.5': {0.01: {"unknown": 97, "unsafe-pgd": 3}, 0.001: {"unknown": 100},
                     0.0001: {"unknown": 48, "safe-incomplete": 52}},
             '0.6': {0.01: {"unknown": 1, "unsafe-pgd": 99}, 0.001: {"unknown": 98, "unsafe-pgd": 2},
                     0.0001: {"unknown": 91, "safe-incomplete": 8, "unsafe-pgd": 1}},
             '0.7': {0.01: {"unsafe-pgd": 100}, 0.001: {"unknown": 87, "unsafe-pgd": 13},
                     0.0001: {"unsafe-pgd": 3, "unknown": 97}}}
LARD_dict = {'0.5': {0.01: {"unsafe-pgd": 100}, 0.001: {"unknown": 100}, 0.0001: {"unknown": 96, "safe-incomplete": 4}},
             '0.6': {0.01: {"unsafe-pgd": 100}, 0.001: {"unknown": 100}, 0.0001: {"unknown": 100}},
             '0.7': {0.01: {"unknown": 100}, 0.001: {"unknown": 100}, 0.0001: {"safe-incomplete": 100}}}

# For each dataset, for each tau, for each epsilon, sort by the order of: 'unsafe-pgd', 'unknown', 'safe-incomplete'
for dataset in [GTSRB_dict, DLOC_dict, LARD_dict]:
    for tau in ['0.5', '0.6', '0.7']:
        for epsilon in [0.01, 0.001, 0.0001]:
            dataset[tau][epsilon] = dict(sorted(dataset[tau][epsilon].items(),
                                                key=lambda item: ['unsafe-pgd', 'unknown', 'safe-incomplete'].index(
                                                    item[0])))

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
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# plot GTSRB
axs[0].set_title('GTSRB')

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
        # eps2write = \epsilon = eps2write
        eps2write = r'$\epsilon = $' + str(eps2write)
        x_val = t + [-0.025, 0, 0.025][list(GTSRB_dict[str(t)].keys()).index(key)]
        # write the eps2write in the bar, with rotation of 90
        axs[0].text(x_val, 75, eps2write, ha='center', va='bottom', fontsize=10, rotation=90)

# Add labels and title
axs[0].set_xlabel('Tau', fontsize=30)
axs[0].set_ylabel('# samples', fontsize=30)
axs[0].set_title('Verification of GTSRB', fontsize=20)
# set x ticks to [0.5, 0.6, 0.7], delete the line between the axis and the ticks
axs[0].set_xticks([0.5, 0.6, 0.7])
axs[0].tick_params(axis='x', which='both', bottom=False)

# plot DLOC
axs[1].set_title('DLOC')

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
        # eps2write = \epsilon = eps2write
        eps2write = r'$\epsilon = $' + str(eps2write)
        x_val = t + [-0.025, 0, 0.025][list(GTSRB_dict[str(t)].keys()).index(key)]
        # write the eps2write in the bar, with rotation of 90
        axs[1].text(x_val, 75, eps2write, ha='center', va='bottom', fontsize=10, rotation=90)

# Add labels and title
axs[1].set_xlabel('Tau', fontsize=30)
# axs[1].set_ylabel('# samples')
axs[1].set_title('Verification of DLOC', fontsize=20)
# set x ticks to [0.5, 0.6, 0.7], delete the line between the axis and the ticks
axs[1].set_xticks([0.5, 0.6, 0.7])
axs[1].tick_params(axis='x', which='both', bottom=False)

# Display the legend with unique entries
# axs[1].legend(title='Categories')


# Display the plot
plt.savefig('res3_diff_tau.pdf')
plt.savefig('res3_diff_tau.png')
plt.close()


# init pandas df with 40 rows (0-39) and 17 columns (for 17 epsilons)
eps = [0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.00075, 0.0005, 0.00025, 0.0001, 7.5e-05, 5e-05, 2.5e-05, 1e-05]
image_ids = [f"image_{i}" for i in range(40)]
import pandas as pd
# create list with length 2*eps, for each eps it will be eps_start and eps_time
eps_rows = [f"{eps[i]}_status" for i in range(len(eps))] + [f"{eps[i]}_time" for i in range(len(eps))]
df = pd.DataFrame(index=image_ids, columns=eps)

# assume we have specific epsilon (eps) and dict with 3 optional categories
# each category contains list of images id
# I want run on all over values, and update df in cell eps, image id with the key value
# for example:
# eps = 0.1, res = {'unsafe-pgd': [0, 1], 'safe': [2]}
res = {'unsafe-pgd': [0, 1], 'safe': [2]}
eps = 0.1
for key, value in res.items():
    for image_id in value:
        image_name = f"image_{image_id}"
        df.loc[image_name, eps] = key

# convert all None to 0
df = df.fillna(0)
# save df
df.to_csv('res.csv')