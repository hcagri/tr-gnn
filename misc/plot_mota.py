import matplotlib.pyplot as plt 
import os 
import os.path as osp 
import pandas as pd 
import matplotlib.gridspec as gridspec


experiment_path = 'outputs/experiments/mot17_private_train_03-26_09:30:02.056639'

exp_files = os.listdir(experiment_path)
epochs = [s for s in exp_files if "Epoch" in s]
epochs = sorted(epochs, key = lambda x: int(x[5:]))


results = {'Epoch': [], 'HOTA': [], 'MOTA': [], 'IDF1': [], 'IDSW': [], 'IDs': [], 'GT_IDs': []}

for epoch in epochs:
    if not osp.exists(osp.join(experiment_path, epoch,'mot_files', 'pedestrian_summary.txt')):
        continue
    with open(osp.join(experiment_path, epoch,'mot_files', 'pedestrian_summary.txt'), 'r') as f:
        data = f.readlines()
    names = data[0].replace('\n', '').split(" ")
    values = data[1].split(" ")
    dict_ = {names[i]: float(values[i]) for i in range(len(names))}

    results['Epoch'].append(int(epoch[5:]))
    results['HOTA'].append(dict_['HOTA'])
    results['MOTA'].append(dict_['MOTA'])
    results['IDF1'].append(dict_['IDF1'])
    results['IDSW'].append(dict_['IDSW'])
    results['IDs'].append(dict_['IDs'])
    results['GT_IDs'].append(dict_['GT_IDs'])


plt.style.use('ggplot')
colors = ['#636EFA','#EF553B','#00CC96','#AB63FA','#FFA15A','#19D3F3','#FF6692','#B6E880','#FF97FF','#FECB52']

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.2)

fig.suptitle(f"\nMOT Metrics\n{experiment_path.split('/')[-1]}", fontsize=22)

ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1, :])
ax1.sharex(ax2)
ax2.sharex(ax1)

ax1.plot(results['Epoch'], results['HOTA'], label='HOTA', color=colors[0], alpha=0.8)
max_value = max(results['HOTA'])
max_index = results['HOTA'].index(max_value)
ax1.scatter(results['Epoch'][max_index], max_value, color='red', s=100)
ax1.annotate(f'HOTA: {max_value:.2f}', (results['Epoch'][max_index], max_value),
             textcoords="offset points", xytext=(0,10), ha='center')


ax1.plot(results['Epoch'], results['MOTA'], label='MOTA', color=colors[1], alpha=0.8)
max_value = max(results['MOTA'])
max_index = results['MOTA'].index(max_value)
ax1.scatter(results['Epoch'][max_index], max_value, color='red', s=100)
ax1.annotate(f'MOTA: {max_value:.2f}', (results['Epoch'][max_index], max_value),
             textcoords="offset points", xytext=(0,10), ha='center')


ax1.plot(results['Epoch'], results['IDF1'], label='IDF1', color=colors[2], alpha=0.8)
max_value = max(results['IDF1'])
max_index = results['IDF1'].index(max_value)
ax1.scatter(results['Epoch'][max_index], max_value, color='red', s=100, label='MAX')
ax1.annotate(f'IDF1: {max_value:.2f}', (results['Epoch'][max_index], max_value),
             textcoords="offset points", xytext=(0,10), ha='center')

ax1.legend()


ax2.plot(results['Epoch'], results['IDSW'], label='IDSW', color=colors[4], alpha=0.8)
min_value = min(results['IDSW'])
min_index = results['IDSW'].index(min_value)
ax2.scatter(results['Epoch'][min_index], min_value, color='red', s=100)
ax2.annotate(f'IDSW: {min_value}', (results['Epoch'][min_index], min_value),
             textcoords="offset points", xytext=(0,10), ha='center')

ax2.plot(results['Epoch'], results['IDs'], label='IDs', color=colors[5], alpha=0.8)
min_value = min(results['IDs'])
min_index = results['IDs'].index(min_value)
ax2.scatter(results['Epoch'][min_index], min_value, color='red', s=100)
ax2.annotate(f'IDs: {min_value}', (results['Epoch'][min_index], min_value),
             textcoords="offset points", xytext=(0,10), ha='center')

ax2.plot(results['Epoch'], results['GT_IDs'], label='GT_IDs', color=colors[6], alpha=0.8, linestyle='--')
ax2.set_xlabel('Epoch')
ax2.legend()


plt.savefig(osp.join(experiment_path, 'mot_metrics.png'), dpi=300)



# fig = plt.figure(figsize=(18, 10))
# gs = gridspec.GridSpec(1, 5)
# gs.update(wspace=0.2)

# ax1 = plt.subplot(gs[0, :3])
# ax2 = plt.subplot(gs[0, 3:])


# ax1.plot(results['Epoch'], results['HOTA'], label='HOTA', color=colors[0], alpha=0.8)
# max_value = max(results['HOTA'])
# max_index = results['HOTA'].index(max_value)
# ax1.scatter(results['Epoch'][max_index], max_value, color='red', s=100)
# ax1.annotate(f'HOTA: {max_value:.2f}', (results['Epoch'][max_index], max_value),
#              textcoords="offset points", xytext=(0,10), ha='center')


# ax1.plot(results['Epoch'], results['MOTA'], label='MOTA', color=colors[1], alpha=0.8)
# max_value = max(results['MOTA'])
# max_index = results['MOTA'].index(max_value)
# ax1.scatter(results['Epoch'][max_index], max_value, color='red', s=100)
# ax1.annotate(f'MOTA: {max_value:.2f}', (results['Epoch'][max_index], max_value),
#              textcoords="offset points", xytext=(0,10), ha='center')


# ax1.plot(results['Epoch'], results['IDF1'], label='IDF1', color=colors[2], alpha=0.8)
# max_value = max(results['IDF1'])
# max_index = results['IDF1'].index(max_value)
# ax1.scatter(results['Epoch'][max_index], max_value, color='red', s=100, label='MAX')
# ax1.annotate(f'IDF1: {max_value:.2f}', (results['Epoch'][max_index], max_value),
#              textcoords="offset points", xytext=(0,10), ha='center')

# ax1.legend()
# ax1.set_xlabel('Epoch')



# ax2.plot(results['Epoch'], results['IDSW'])
# ax2.plot(results['Epoch'], results['IDs'])
# ax2.plot(results['Epoch'], results['GT_IDs'], linestyle='-')

# plt.savefig('test.png', dpi=300)