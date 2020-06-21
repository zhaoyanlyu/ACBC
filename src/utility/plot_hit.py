import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

data_all = pd.read_csv('../../results/log.csv')

data_MAP = data_all.loc[data_all['algorithm'] == 'MAP']
data_RICH = data_all.loc[data_all['algorithm'] == 'RICH']
data_ACBC = data_all.loc[data_all['algorithm'] == 'ACBC']
data_dsrb = data_all.loc[data_all['algorithm'] == 'distributed']

x_axis = np.arange(15, 34)

hit_MAP = [data_MAP.loc[data_MAP['speed_mu'] == i].iloc[0]['avg_hit'] for i in x_axis]
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

delay_EN = 3
delay_AC = 10
delay_CS = 40

delay_MAP = hit_MAP * delay_EN + (1-hit_MAP) * delay_CS
delay_RICH = hit_RICH * delay_EN + (1-hit_RICH) * delay_CS
delay_ACBC = hit_ACBC * delay_EN + (1-hit_ACBC) * delay_AC
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


# -- Plot setup
SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rcParams['font.family'] = 'Times New Roman'

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
plt.xlim(15, 33)
plt.xticks(range(15, 34, 2))

# plt.plot(x_axis, [625]*19, color='#34495E', linewidth=2.0, label='Tot. chunks', linestyle='--', zorder=1)
plt.plot(x_axis, hit_ACBC, color='#E74C3C', linewidth=4.0, label='ACBC (ours)', zorder=0)
plt.plot(x_axis, hit_MAP, color='#884EA0', linewidth=2.0, label='MAP', zorder=0)
plt.plot(x_axis, hit_RICH, color='#2471A3', linewidth=2.0, label='RICH', zorder=0)
plt.plot(x_axis, hit_dsrb_0, color='#1ABC9C', linewidth=2.0, label='Distr-0', zorder=0)
plt.plot(x_axis, hit_dsrb_5, color='#9E9D24', linewidth=2.0, label='Distr-5', zorder=0)
plt.plot(x_axis, hit_dsrb_10, color='#F39C12', linewidth=2.0, label='Distr-10', zorder=0)

plt.legend(ncol=3, bbox_to_anchor=(0, 1, 1, 1), mode='expand', loc=3)

plt.xlabel('speed (m/s)')
plt.ylabel('hit ratio (%)')

plt.grid()
ax1.set_facecolor('#E8EAF6')

plt.show()
