#!/usr/bin/env python3

import pickle
import matplotlib.pyplot as plt
import matplotlib

accuracy = pickle.load(open("../sample_accuracy.pickle", "rb"))
latency = pickle.load(open("../sample_latency.pickle", "rb"))

plt.rcParams["font.family"] = "Linux Biolinum O"
plt.rcParams.update({'font.size': 33,'font.weight':'bold','pdf.fonttype':42})
plt.rcParams["legend.handlelength"]=1.5

fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_axes([0.174, 0.235, 0.779, 0.68])
ax1.yaxis.set_label_coords(-0.142, 0.48)
ax1.ticklabel_format(style='sci', axis='x')


ax1.scatter(latency, accuracy, c='#adadad', s=10, zorder=6)


#ax1.set_xticks(xtick)
#ax1.set_xlim(xlim)
#ax1.set_yticks([i for i in range(72, 79, 2)])
#ax1.set_ylim([72, 78])

#ax1.set_xlabel(xname +  ' Lat. (ms)',weight='bold')
ax1.set_xlabel('Predicted Latency (ms)',weight='bold')
ax1.set_ylabel('Predicted Acc. (%)', weight='bold')

ax1.yaxis.grid(zorder=-1,color='lightgray',dashes=(5,10),linewidth=1,linestyle='--')
ax1.xaxis.grid(zorder=-1,color='lightgray',dashes=(5,10),linewidth=1,linestyle='--')

#ax1.yaxis.get_major_formatter().set_powerlimits((0,1))

ax1.tick_params(axis='both',direction='out', length=10, width=3, colors='k', grid_alpha=1)
for axis in ['top','bottom','left','right']:
	ax1.spines[axis].set_linewidth(3.0)
	
#legend=fig.legend(scatterpoints=1, frameon=True,ncol=1,bbox_to_anchor=(0.95,0.52),fontsize='small',facecolor='w',edgecolor='k',handletextpad=0.1,labelspacing=0,columnspacing = 0.1)
#legend.get_frame().set_alpha(0.5)


plt.show()

fig.savefig('./sample_model.pdf')