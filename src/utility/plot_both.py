import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# matplotlib.rcParams['text.usetex'] = True

data_all = pd.read_csv('../../results/log.csv')

data_MAP = data_all.loc[data_all['algorithm'] == 'MAP']
data_RICH = data_all.loc[data_all['algorithm'] == 'RICH']
data_ACBC = data_all.loc[data_all['algorithm'] == 'ACBC']
data_dsrb = data_all.loc[data_all['algorithm'] == 'distributed']

x_axis = np.arange(15, 34)

hit_MAP = [data_MAP.loc[data_MAP['speed_mu'] == i].iloc[0]['avg_hit'] for i in x_axis]
# hit_RICH = [data_RICH.loc[data_RICH['speed_mu'] == i].iloc[0]['avg_hit'] for i in x_axis]
hit_RICH = [data_RICH.loc[(data_RICH['speed_mu'] == i) & (data_RICH['tot_user'] == 1000)].iloc[0]['avg_hit'] for i in x_axis]
hit_ACBC = [data_ACBC.loc[data_ACBC['speed_mu'] == i].iloc[0]['avg_hit'] for i in x_axis]
hit_dsrb_0 = [data_dsrb.loc[(data_dsrb['speed_mu'] == i) & (data_dsrb['overflow'] == 0)].iloc[0]['avg_hit']
              for i in x_axis]
hit_dsrb_5 = [data_dsrb.loc[(data_dsrb['speed_mu'] == i) & (data_dsrb['overflow'] == 5)].iloc[0]['avg_hit']
              for i in x_axis]
hit_dsrb_10 = [data_dsrb.loc[(data_dsrb['speed_mu'] == i) & (data_dsrb['overflow'] == 10)].iloc[0]['avg_hit']
               for i in x_axis]

hit_MAP = np.asarray(hit_MAP).astype(np.float32) / 625
hit_RICH = np.asarray(hit_RICH).astype(np.float32) / 625
hit_ACBC = np.asarray(hit_ACBC).astype(np.float32) / 625
hit_dsrb_0 = np.asarray(hit_dsrb_0).astype(np.float32) / 625
hit_dsrb_5 = np.asarray(hit_dsrb_5).astype(np.float32) / 625
hit_dsrb_10 = np.asarray(hit_dsrb_10).astype(np.float32) / 625

delay_EN = 1.5
delay_AC = 3
delay_CS = 10

delay_MAP = hit_MAP * delay_EN + (1-hit_MAP) * delay_CS
delay_MAP_AC = hit_MAP * delay_EN + (1-hit_MAP) * delay_AC
delay_RICH = hit_RICH * delay_EN + (1-hit_RICH) * delay_CS
delay_RICH_AC = hit_RICH * delay_EN + (1-hit_RICH) * delay_AC
delay_ACBC = hit_ACBC * delay_EN + (1-hit_ACBC) * delay_CS
delay_ACBC_AC = hit_ACBC * delay_EN + (1-hit_ACBC) * delay_AC
delay_dsrb_0 = hit_dsrb_0 * delay_EN + (1-hit_dsrb_0) * delay_CS
delay_dsrb_5 = hit_dsrb_5 * delay_EN + (1-hit_dsrb_5) * delay_CS
delay_dsrb_10 = hit_dsrb_10 * delay_EN + (1-hit_dsrb_10) * delay_CS


PoF_MAP = [data_MAP.loc[data_MAP['speed_mu'] == i].iloc[0]['avg_PoF'] for i in x_axis]
PoF_RICH = [data_RICH.loc[data_RICH['speed_mu'] == i].iloc[0]['avg_PoF'] for i in x_axis]
PoF_ACBC = [data_ACBC.loc[data_ACBC['speed_mu'] == i].iloc[0]['avg_PoF'] for i in x_axis]
PoF_dsrb_0 = [data_dsrb.loc[(data_dsrb['speed_mu'] == i) & (data_dsrb['overflow'] == 0)].iloc[0]['avg_PoF']
              for i in x_axis]
PoF_dsrb_5 = [data_dsrb.loc[(data_dsrb['speed_mu'] == i) & (data_dsrb['overflow'] == 5)].iloc[0]['avg_PoF']
              for i in x_axis]
PoF_dsrb_10 = [data_dsrb.loc[(data_dsrb['speed_mu'] == i) & (data_dsrb['overflow'] == 10)].iloc[0]['avg_PoF']
               for i in x_axis]

PoF_RICH_AC = np.asarray(PoF_RICH) + 1
PoF_ACBC_AC = np.asarray(PoF_ACBC) + 1
PoF_MAP_AC = np.asarray(PoF_MAP) + 1


# -- Plot setup
SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, axs = plt.subplots(2, 1, figsize=(10, 11), dpi=300)
# -------------------------------
axs[0].set_xlim(15, 33)
axs[0].set_xticks(range(15, 34, 2))

# axs[0].plot(x_axis, [625]*19, color='#34495E', linewidth=4.0, label='Tot. chunks', linestyle='--', zorder=1)

axs[0].plot(x_axis, delay_ACBC_AC, color='#E74C3C', linewidth=4.0, label='ACBC-AC (ours)', zorder=0)
axs[0].plot(x_axis, delay_ACBC, color='#E74C3C', linewidth=4.0, label='ACBC (ours)', linestyle=':', zorder=0)
axs[0].plot(x_axis, delay_MAP_AC, color='#884EA0', linewidth=3.0, label='MAP-AC', zorder=0)
axs[0].plot(x_axis, delay_MAP, color='#884EA0', linewidth=3.0, label='MAP', linestyle=':', zorder=0)
axs[0].plot(x_axis, delay_RICH_AC, color='#2471A3', linewidth=3.0, label='RICH-AC', zorder=0)
axs[0].plot(x_axis, delay_RICH, color='#2471A3', linewidth=3.0, label='RICH', linestyle=':', zorder=0)
axs[0].plot(x_axis, delay_dsrb_0, color='#1ABC9C', linewidth=3.0, label=r"Distr $\xi=0$", zorder=0)
axs[0].plot(x_axis, delay_dsrb_10, color='#F39C12', linewidth=3.0, label=r"Distr $\xi=10$", zorder=0)

axs[0].legend(ncol=4, bbox_to_anchor=(0, 1, 1, 1), mode='expand', loc=3)

axs[0].set_xlabel('speed $\mu$ (m/s)')
axs[0].set_ylabel('average access delay (ms)')

axs[0].grid()
axs[0].set_facecolor('#E8EAF6')

# -------------------------------
axs[1].set_xlim(15, 33)
axs[1].set_xticks(range(15, 34, 2))
axs[1].set_ylim(0, 12)
axs[1].set_yticks(np.arange(0, 12, 2))

axs[1].plot(x_axis, PoF_ACBC_AC, color='#E74C3C', linewidth=4.0, label='ACBC-AC (ours)', zorder=0)
axs[1].plot(x_axis, PoF_ACBC, color='#E74C3C', linewidth=4.0, label='ACBC (ours)', linestyle=':', zorder=0)
axs[1].plot(x_axis, PoF_MAP_AC, color='#884EA0', linewidth=3.0, label='MAP-AC', zorder=0)
axs[1].plot(x_axis, PoF_MAP, color='#884EA0', linewidth=3.0, label='MAP', linestyle=':', zorder=0)
axs[1].plot(x_axis, PoF_RICH_AC, color='#2471A3', linewidth=3.0, label='RICH-AC', zorder=0)
axs[1].plot(x_axis, PoF_RICH, color='#2471A3', linewidth=3.0, label='RICH', linestyle=':', zorder=0)
axs[1].plot(x_axis, PoF_dsrb_0, color='#1ABC9C', linewidth=3.0, label=r"Distr $\xi=0$", zorder=0)
axs[1].plot(x_axis, PoF_dsrb_10, color='#F39C12', linewidth=3.0, label=r"Distr $\xi=10$", zorder=0)

axs[1].set_xlabel('speed $\mu$ (m/s)')
axs[1].set_ylabel('Price of fog (PoF)')

axs[1].grid()
axs[1].set_facecolor('#E8EAF6')

# -------------------------------
# Picture in picture
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
axins = zoomed_inset_axes(axs[1], 2.5, loc='upper right')
axins.plot(x_axis, PoF_ACBC_AC, color='#E74C3C', linewidth=4.0, label='ACBC-AC (ours)', zorder=0)
axins.plot(x_axis, PoF_ACBC, color='#E74C3C', linewidth=4.0, label='ACBC (ours)', linestyle=':', zorder=0)
axins.plot(x_axis, PoF_MAP_AC, color='#884EA0', linewidth=3.0, label='MAP-AC', zorder=0)
axins.plot(x_axis, PoF_MAP, color='#884EA0', linewidth=3.0, label='MAP', linestyle=':', zorder=0)
axins.plot(x_axis, PoF_RICH_AC, color='#2471A3', linewidth=3.0, label='RICH-AC', zorder=0)
axins.plot(x_axis, PoF_RICH, color='#2471A3', linewidth=3.0, label='RICH', linestyle=':', zorder=0)
axins.plot(x_axis, PoF_dsrb_0, color='#1ABC9C', linewidth=3.0, label=r"Distr $\xi=0$", zorder=0)
axins.plot(x_axis, PoF_dsrb_10, color='#F39C12', linewidth=3.0, label=r"Distr $\xi=10$", zorder=0)
axins.set_xlim(30, 33)
axins.set_ylim(2, 4.5)

axins.yaxis.set_visible(False)
axins.xaxis.set_visible(False)
axins.set_facecolor('#ECEEF9')

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mark_inset(axs[1], axins, loc1=3, loc2=4, fc="none", ec="0.5", linewidth=1.5)

plt.show()
