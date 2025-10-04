import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data = "MoDataV1_5-4_tiles_MoDataV1.csv_sums_CSV.csv"
df = pd.read_csv(data)

# bin avg sum in intervals of size 5
bin_size = 5
min_bin = np.floor(df['AverageSum'].min() / bin_size) * bin_size
max_bin = np.ceil(df['AverageSum'].max() / bin_size) * bin_size + bin_size
bin_edges = np.arange(min_bin, max_bin, bin_size)
df['AvgSum_bin'] = pd.cut(df['AverageSum'], bins=bin_edges, right=False)

# fcn to compute 95% CI !!
def ci95(series):
    n = series.count()
    if n < 2:
        return 0
    return 1.96 * series.std(ddof=1) / np.sqrt(n)

# group by bins and calculate mean and CI for each group
grouped = df.groupby('AvgSum_bin').agg(
    PrimarySum_mean=('PrimarySum', 'mean'),
    PrimarySum_ci=('PrimarySum', ci95),
    SecondarySum_mean=('SecondarySum', 'mean'),
    SecondarySum_ci=('SecondarySum', ci95),
    BotSum_mean=('BotSum', 'mean'),
    BotSum_ci=('BotSum', ci95),
    bin_center=('AverageSum', 'mean')
).reset_index()

# === STATISTICAL SIGNIFICANCE TESTING ===
# for each bin, compare BotSum vs average of human counts
print("Statistical Significance Analysis")
print("=" * 80)
print(f"{'Bin':<15} {'Bot Mean':<12} {'Human Mean':<12} {'t-stat':<10} {'p-value':<10} {'Sig?':<8}")
print("-" * 80)

significant_bins = []

for idx, row in grouped.iterrows():
    bin_label = row['AvgSum_bin']
   
    # get data for this bin
    bin_data = df[df['AvgSum_bin'] == bin_label]
   
    if len(bin_data) < 2:
        print(f"{str(bin_label):<15} {'N/A (insufficient data)'}")
        continue
   
    bot_values = bin_data['BotSum'].values
    # avg of the two human counts for each observation
    human_avg = (bin_data['PrimarySum'] + bin_data['SecondarySum']) / 2
   
    # paired t-test (since bot and human counts are for the same tiles/full caps)
    t_stat, p_value = stats.ttest_rel(bot_values, human_avg)
   
    is_significant = p_value < 0.05
    if is_significant:
        significant_bins.append(str(bin_label))
   
    sig_marker = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))
   
    print(f"{str(bin_label):<15} {row['BotSum_mean']:>10.2f}  {row['PrimarySum_mean']/2 + row['SecondarySum_mean']/2:>10.2f}  "
          f"{t_stat:>9.3f}  {p_value:>9.4f}  {sig_marker:<8}")

print("=" * 80)
print(f"\nSignificance levels: * p<0.05, ** p<0.01, *** p<0.001")
print(f"\nBins with significant differences (p < 0.05): {len(significant_bins)}")
if significant_bins:
    print(f"Bins: {', '.join(significant_bins)}")

# plot with significance markers
plt.figure(figsize=(12, 5))
plt.rc('font', size=14)

plt.errorbar(grouped['bin_center'], grouped['PrimarySum_mean'], yerr=grouped['PrimarySum_ci'],
             fmt='-', linewidth=1.5, capsize=7, capthick=1.5, elinewidth=1,
             color='fuchsia', label='Human 1 Count')
plt.errorbar(grouped['bin_center'], grouped['SecondarySum_mean'], yerr=grouped['SecondarySum_ci'],
             fmt='-', linewidth=1.5, capsize=7, capthick=1.5, elinewidth=1,
             color='deeppink', label='Human 2 Count')
plt.errorbar(grouped['bin_center'], grouped['BotSum_mean'], yerr=grouped['BotSum_ci'],
             fmt='-', linewidth=1.5, capsize=7, capthick=1.5, elinewidth=1,
             color='limegreen', label='Model Predicted Count')

plt.text(1, 55, "R² value = 0.925")
plt.xlabel("Humans' Average Count")
plt.ylabel('Egg Count')
plt.title('Human and Model Counts with 95% Confidence Interval')
plt.legend()
plt.xticks(np.arange(0, 70, 7.5))
plt.tight_layout()
plt.savefig("ci3.png")
plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# data = "MoDataV1_5-4_tiles_MoDataV1.csv_sums_CSV.csv"

# df = pd.read_csv(data)

# # bin avg sum in intervals of size 5
# bin_size = 5
# min_bin = np.floor(df['AverageSum'].min() / bin_size) * bin_size
# max_bin = np.ceil(df['AverageSum'].max() / bin_size) * bin_size + bin_size
# bin_edges = np.arange(min_bin, max_bin, bin_size)
# df['AvgSum_bin'] = pd.cut(df['AverageSum'], bins=bin_edges, right=False)

# # fcn to compute 95% CI !!
# def ci95(series):
#     n = series.count()
#     if n < 2:
#         return 0
#     return 1.96 * series.std(ddof=1) / np.sqrt(n)

# # group by bins and calculate mean and CI for each group (which is what we graph)
# grouped = df.groupby('AvgSum_bin').agg(
#     PrimarySum_mean=('PrimarySum', 'mean'),
#     PrimarySum_ci=('PrimarySum', ci95),
#     SecondarySum_mean=('SecondarySum', 'mean'),
#     SecondarySum_ci=('SecondarySum', ci95),
#     BotSum_mean=('BotSum', 'mean'),
#     BotSum_ci=('BotSum', ci95),
#     bin_center=('AverageSum', 'mean')  # use the mean of AverageSum in each bin as x
# ).reset_index()

# # plot
# plt.figure(figsize=(12, 5))
# plt.rc('font', size=14)

# plt.errorbar(grouped['bin_center'], grouped['PrimarySum_mean'], yerr=grouped['PrimarySum_ci'],
#              fmt='-', linewidth=1.5, capsize=7, capthick=1.5, elinewidth=1, 
#              color='fuchsia', label='Human 1 Count')
# plt.errorbar(grouped['bin_center'], grouped['SecondarySum_mean'], yerr=grouped['SecondarySum_ci'],
#              fmt='-', linewidth=1.5, capsize=7, capthick=1.5, elinewidth=1, 
#              color='deeppink', label='Human 2 Count')
# plt.errorbar(grouped['bin_center'], grouped['BotSum_mean'], yerr=grouped['BotSum_ci'],
#              fmt='-', linewidth=1.5, capsize=7, capthick=1.5, elinewidth=1, 
#              color='limegreen', label='Model Predicted Count')
# plt.text(1, 55, "R² value = 0.925")
# plt.xlabel("Humans' Average Count")
# plt.ylabel('Egg Count')
# plt.title('Human and Model Counts with 95% Confidence Interval')
# plt.legend()
# plt.xticks(np.arange(0, 70, 7.5))
# plt.tight_layout()
# plt.savefig("ci3.png")