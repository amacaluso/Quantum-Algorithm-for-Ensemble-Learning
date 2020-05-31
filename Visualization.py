from Utils import *

data_out = pd.read_csv('output/data_out.csv')

x = np.arange(len(data_out))

fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10,3))

p0_avg = data_out.qAVG_sim
p0_ens = data_out.qEns_sim
p0_clas = data_out.cAVG

#ax = plt.subplot(221)

ax.plot(x, p0_ens, color='orange', label='qEnsemble', zorder=1, linewidth=5)
ax.plot(x, p0_avg, color='steelblue', label='qAVG')
ax.scatter(x, p0_clas, label='cAVG', color='sienna', zorder=2, linewidth=.5)

#ax.set_xlim(-1.1, 1.1)
# ax.set_ylim(-.2, 1.05)
ax.grid(alpha=0.3)
#ax.set_xticks(np.round(np.arange(-1, 1.1, .4), 1).tolist())
#ax.set_title('Comparison', size=14)
ax.tick_params(labelsize=12)


avg = data_out.qAVG_real
ens = data_out.qEns_real
clas = data_out.cAVG

#ax1 = plt.subplot(222)

ax1.plot(x, ens, color='orange', label='qEnsemble', zorder=1, linewidth=5)
ax1.plot(x, avg, color='steelblue', label='qAVG')
ax1.scatter(x, clas, label='cAVG', color='sienna', zorder=2, linewidth=.5)

#ax.set_xlim(-1.1, 1.1)
# ax.set_ylim(-.2, 1.05)
ax1.grid(alpha=0.3)
#ax.set_xticks(np.round(np.arange(-1, 1.1, .4), 1).tolist())
#ax.set_title('Comparison', size=14)
ax1.tick_params(labelsize=12)
# plt.legend(loc='lower center',# bbox_to_anchor=(0.5, -.15),
#           ncol=3, fancybox=True, shadow=True, fontsize = 12)
# handles, labels = ax.get_legend_handles_labels()
# lgd=fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.4, -.025))

plt.savefig('output/experiments.png', dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()
plt.close()


fig = plt.figure(figsize=(6, 1))
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='center', ncol=3)
plt.savefig('output/legend.png', dpi=300, bbox_inches='tight')
plt.show()


