#Train networks on multiple problems of increasing top complexity.
from QuotientHomology import *
from torch.utils.data import Dataset, TensorDataset
import torch.optim as optim
from tqdm import tqdm
import persim

def optimize_nn(model, dat_loader, optim, crit, num_epochs = 100):
    loss_hist = []
    for epoch in tqdm(range(num_epochs)):
        for inputs, targets in dat_loader:
            # Forward pass
            outputs = model(inputs)[-1]
            loss = crit(outputs, targets)
    
            # Backward pass and optimization
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_hist.append(loss.item())
        if (epoch+1) % 10 == 0 and epoch<300: #No need to compute performance at every step early on
            class_predictions = torch.argmax(model(dat_loader.dataset.tensors[0])[-1],1)==dat_loader.dataset.tensors[1]
            class_performance = class_predictions.sum()/len(dat_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Classification performance: {class_performance:.4f}')
            # if class_performance==1:
                # return outputs, loss_hist
        elif epoch>300:
            class_predictions = torch.argmax(model(dat_loader.dataset.tensors[0])[-1],1)==dat_loader.dataset.tensors[1]
            class_performance = class_predictions.sum()/len(dat_loader)
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Classification performance: {class_performance:.4f}')
            # if class_performance==1:
                # return outputs, loss_hist
    return outputs, loss_hist
#%%

Mfld = ManifoldGenerator()

def construct_data(n_points, n_shells=2, dim=1):
    r_1 = np.arange(1,1+n_shells)
    r_2 = np.arange(1,1+n_shells)+0.5
    mfld_1 = np.hstack([r*Mfld.Sn(n_points,[dim+1]) for r in r_1])
    mfld_2 = np.hstack([r*Mfld.Sn(n_points,[dim+1]) for r in r_2])
    labels = np.array([0]*len(mfld_1.T) + [1]*len(mfld_2.T))
    X = np.hstack([mfld_1,mfld_2])
    shuff_perm = np.random.permutation(len(X.T))
    return torch.Tensor(X)[:,shuff_perm], torch.Tensor(labels)[shuff_perm]


#%%
Hidden_layers = [25,25,25,25]
n_models = 10

n_samples = 500
dims = np.arange(1,4)

init_models = [[] for d in range(len(dims))]
trained_models = [[] for d in range(len(dims))]
decompositions_init = [[] for d in range(len(dims))]
decompositions_trained = [[] for d in range(len(dims))]

init_representations = []
trained_representations = []

n_samples_in_poly_init = np.empty([n_models,len(dims),len(Hidden_layers)],dtype=object)
n_samples_in_poly_trained = np.empty([n_models,len(dims),len(Hidden_layers)],dtype=object)

for d in dims:
    X,y = construct_data(n_samples,dim=d)
    if d==1:
        X_points = X.clone()
    data_train = TensorDataset(X.T, y.long())
    trainloader = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    for n in range(n_models):
        model = FeedforwardNetwork(d+1, Hidden_layers, 
                                   out_layer_sz=2, init_type='none',activation=nn.ReLU)
        
        decomposer = NetworkDecompositions(model)
        decomposer.compute_overlap_decomp(X.T)
        decompositions_init[d-1].append(decomposer)
        for l in range(len(Hidden_layers)):
            n_samp_reg = [len(reg) for reg in decomposer.global_decomposition[0][l]]
            n_samples_in_poly_init[n,d-1,l] = n_samp_reg
        init_representations.append(model(X.T))
        
        
        optimizer = optim.Adam(model.parameters(), lr=0.00002)
        outputs, loss_hist = optimize_nn(model, trainloader, optimizer, criterion, num_epochs=1000)
        
        decomposer = NetworkDecompositions(model)
        decomposer.compute_overlap_decomp(X.T)
        trained_models[d-1].append(model)    
        decompositions_trained[d-1].append(decomposer)
        for l in range(len(Hidden_layers)):
            n_samp_reg = [len(reg) for reg in decomposer.global_decomposition[0][l]]
            n_samples_in_poly_trained[n,d-1,l] = n_samp_reg
        trained_representations.append(model(X.T))

#%%
import pickle
mode = 'save'
if mode=='save':
    for i,dec in enumerate(decompositions_init):
        for n, dec_i in enumerate(dec):
            with open("../models/Decompositions/Decomp_model_init"+f"_{i}_{n}.pkl", 'wb') as outp:
                pickle.dump(dec, outp, pickle.HIGHEST_PROTOCOL)
if mode=='save':
    for i,dec in enumerate(decompositions_trained):
        for n, dec_i in enumerate(dec):
            with open("../models/Decompositions/Decomp_model_trained"+f"_{i}_{n}.pkl", 'wb') as outp:
                pickle.dump(dec, outp, pickle.HIGHEST_PROTOCOL)
else:
    decompositions_init = [[] for d in range(len(dims))]
    decompositions_trained = [[] for d in range(len(dims))]
    for n in range(len(dims)):
        for i in range(n_models):
            with open("../models/Decompositions/Decomp_model_init"+f"_{i}_{n}.pkl", 'rb') as outp:
                decompositions_init[n].append(pickle.load(outp))
            with open("../models/Decompositions/Decomp_model_trained"+f"_{i}_{n}.pkl", 'rb') as outp:
                decompositions_trained[n].append(pickle.load(outp))
#%%Plot of decompositions for circle case
# X = construct_data(n_samples,dim=1)[0]
fig,ax = plt.subplots(2,5,figsize=(12,4))
for l in range(5):
    pc.Region(decompositions_init[0][0].polyhedral_decomposition[l]).plot(ax=ax[0,l],color='red',linewidth=1,linestyle='-')
    ax[0,l].scatter(X_points[0],X_points[1],color='gray',alpha=0.2, s=2)
    [ax[0,l].scatter(X_points[0,clss],X_points[1,clss],cmap='gist_ncar', vmin=0, vmax=1, s=5,
              c=(n*np.ones(len(clss)))/len(decompositions_init[0][0].overlap_decomposition[l])) for n, clss in 
     enumerate(decompositions_init[0][0].overlap_decomposition[l])]
    ax[0,l].set_xlim(-2.75,2.75)
    ax[0,l].set_ylim(-2.75,2.75)
    ax[0,l].axis('off')
    pc.Region(decompositions_trained[0][0].polyhedral_decomposition[l]).plot(ax=ax[1,l], color='cyan',linewidth=1,linestyle='-')
    ax[1,l].scatter(X_points[0],X_points[1],color='gray',alpha=0.2, s=2)
    [ax[1,l].scatter(X_points[0,clss],X_points[1,clss],cmap='gist_ncar', vmin=0, vmax=1, s=5,
              c=(n*np.ones(len(clss)))/len(decompositions_trained[0][0].overlap_decomposition[l])) for n, clss in 
     enumerate(decompositions_trained[0][0].overlap_decomposition[l])]
    ax[1,l].set_xlim(-2.75,2.75)
    ax[1,l].set_ylim(-2.75,2.75)
    ax[1,l].axis('off')
fig.savefig(f'../figures/polyhedra_layers.png',dpi=500, transparent=True)

#%%Plot of volume of regions before and after training
volume_init = np.empty((len(dims), 5), dtype=object)
for d in range(len(dims)):
    for l in range(5):
        volume_init[d,l] = []
        for i in range(n_models):
            print(i)
            for poly in decompositions_init[d][i].polyhedral_decomposition[l]:
                volume_init[d,l].append(poly.volume)
                
volume_trained = np.empty((len(dims), 5), dtype=object)
for d in range(len(dims)):
    for l in range(5):
        volume_trained[d,l] = []
        for i in range(n_models):
            print(i)
            for poly in decompositions_trained[d][i].polyhedral_decomposition[l]:
                volume_trained[d,l].append(poly.volume)
   
#%%
volume_overlap_init = np.empty((len(dims)), dtype=object)
volume_overlap_trained = np.empty((len(dims)), dtype=object)
for d in range(len(dims)):
    volume_overlap_init[d] = []
    for i in range(n_models):
        for n, poly in enumerate(decompositions_init[d][i].polyhedral_decomposition[-1]):
            for ovr in decompositions_init[d][i].overlap_decomposition[-1]:
                if len(np.intersect1d(decompositions_init[d][i].global_decomposition[0][-1][n],ovr))>0:
                    volume_overlap_init[d].append(poly.volume)
                    break

for d in range(len(dims)):
    volume_overlap_trained[d] = []
    for i in range(n_models):
        for n, poly in enumerate(decompositions_trained[d][i].polyhedral_decomposition[-1]):
            for ovr in decompositions_trained[d][i].overlap_decomposition[-1]:
                if len(np.intersect1d(decompositions_trained[d][i].global_decomposition[0][-1][n],ovr))>0:
                    volume_overlap_trained[d].append(poly.volume)
                    break
#%%
init_vol_means = np.array([[np.median(volume_init[d,l]) for l in range(5)] for d in range(len(dims))])
trained_vol_means = np.array([[np.median(volume_trained[d,l]) for l in range(5)] for d in range(len(dims))])
init_vol_low = np.array([[np.percentile(volume_init[d,l],18) for l in range(5)] for d in range(len(dims))])
init_vol_high = np.array([[np.percentile(volume_init[d,l],82) for l in range(5)] for d in range(len(dims))])
trained_vol_low = np.array([[np.percentile(volume_trained[d,l],18) for l in range(5)] for d in range(len(dims))])
trained_vol_high = np.array([[np.percentile(volume_trained[d,l],82) for l in range(5)] for d in range(len(dims))])

clrs = ['orange','green', 'purple']

fig = plt.figure(figsize=(3.5,2.5))
for d in range(len(dims)):
    plt.errorbar(np.arange(0, 5),init_vol_means[d],
                 yerr=(init_vol_means[d]-init_vol_low[d], init_vol_means[d]+init_vol_high[d]),color=clrs[d],
                 alpha=0.7)
    plt.plot(np.arange(0, 5),init_vol_means[d],'.',color = 'red', zorder=10000)
    
    # plt.fill_between(np.arange(0, 5), init_vol_means[d]-init_vol_low[d], init_vol_means[d]+init_vol_high[d],alpha=0.2)
    plt.errorbar(np.arange(0.1, 5.1),trained_vol_means[d],
                 yerr=(trained_vol_means[d]-trained_vol_low[d], trained_vol_means[d]+trained_vol_high[d]),color=clrs[d],
                 linestyle='--')
    plt.plot(np.arange(0.1, 5.1),trained_vol_means[d],'*', color='cyan', zorder=10000)
plt.yscale('log')
plt.xticks([0,1,2,3,4])
plt.savefig(f'../figures/volumes.png',dpi=500, transparent=True)
#volume_init = [poly.vol for poly in decompositions_init[d][i].polyhedral_decomposition[l] for l in range(5) for i in range(n_models) for d in range(len(dims))]
# volume_trained = 


volume_overlap = []
for d in range(len(dims)):
    volume_overlap.append(volume_overlap_init[d])
    volume_overlap.append(volume_overlap_trained[d])
    
fig,ax = plt.subplots(figsize=(3.5,2.5))

boxplt = ax.boxplot(volume_overlap,showfliers=False, patch_artist=True,
            medianprops={'linewidth':0})#[np.median(volume_overlap_init[d]) for d in range(len(dims))])
for n,patch in enumerate(boxplt['boxes']):
    if n%2==0:
        patch.set_facecolor([1,0,0,1])
    else:
        patch.set_facecolor([0,1,1,0.4])
    patch.set_edgecolor('black')
    patch.set_linewidth(2)

    
    
[plt.plot([1+2*i,2*i+2],[np.median(volume_overlap_init[i]),np.median(volume_overlap_trained[i])],'-^',color=clrs[i]) for i in range(len(dims))]
plt.yscale('log')
plt.ylim(10e-7, 10e0)
plt.xticks([])
fig.savefig(f'../figures/volumes_overlap.png',dpi=500, transparent=True)

#%% Plot individual neuron functions with hypersampling
X_sample = []
for polytope in decompositions_trained[0][0].polyhedral_decomposition[-1]:
    H_rep = (polytope.A, polytope.b)
    X_sample.append(decompositions_trained[0][0].populate_polyhedra(H_rep, 100))
X_sample = torch.Tensor(np.vstack(X_sample)) 
out = model(X_sample)

plt.figure()
# plt.scatter(X_sample[:,0],X_sample[:,1],c=out[-1][:,0].detach().numpy(),vmin=-1,vmax=1,s=4)
plt.quiver(X_sample[:,0],X_sample[:,1],out[-1][:,0].detach().numpy(),out[-1][:,1].detach().numpy(),width=0.001)
plt.scatter(X_points[0],X_points[1],color='gray',alpha=0.2, s=2)

#%%
mean_overlaps_init = [[len(decompositions_init[d][i].overlap_decomposition[-1]) 
                      for i in range(n_models)] for d in range(len(dims))]
mean_overlaps_trained = [[len(decompositions_trained[d][i].overlap_decomposition[-1]) 
                      for i in range(n_models)] for d in range(len(dims))]
mean_overlaps = []
for i in range(len(dims)): #sort overlaps for plotting
    mean_overlaps.append(mean_overlaps_init[i])
    mean_overlaps.append(mean_overlaps_trained[i])

fig,ax = plt.subplots(figsize=(3.5,2.5))
# plt.errorbar([0,1],[len(decompositions_init[0][0].overlap_decomposition[-1]),
#                 len(decompositions_trained[0][0].overlap_decomposition[-1])],yerr=1,color='brown', marker='s')
boxplt = ax.boxplot(mean_overlaps,medianprops={'linewidth':0}, patch_artist=True)
for n,patch in enumerate(boxplt['boxes']):
    if n%2==0:
        patch.set_facecolor([1,0,0,1])
    else:
        patch.set_facecolor([0,1,1,0.4])
    patch.set_edgecolor('black')
    patch.set_linewidth(2)

[plt.plot([1+2*i,2*i+2],[np.median(mean_overlaps_init[i]),np.median(mean_overlaps_trained[i])],'-^',color=clrs[i]) for i in range(len(dims))]
# plt.legend(['$S^1$','$S^2$','$S^3$'])
plt.plot([1,3,5],np.median(mean_overlaps_init,1),'red',linestyle='--',alpha=0.5)
plt.xticks([])#np.arange(1,1+2*len(dims)),['initialization','trained']+['']*4,rotation=-15)
plt.savefig(f'../figures/n_regions.png',dpi=500, transparent=True)


mean_overlap_reg_init = [[len(np.concatenate(decompositions_init[d][i].overlap_decomposition[-1])) 
                      for i in range(n_models)] for d in range(len(dims))]
mean_overlap_reg_trained = [[len(np.concatenate(decompositions_trained[d][i].overlap_decomposition[-1])) 
                      for i in range(n_models)] for d in range(len(dims))]
mean_overlap_reg = []
for i in range(len(dims)): #sort overlaps for plotting
    mean_overlap_reg.append(mean_overlap_reg_init[i])
    mean_overlap_reg.append(mean_overlap_reg_trained[i])

plt.figure(figsize=(3,2))
# plt.errorbar([0,1],[len(np.concatenate(decompositions_init[0][0].overlap_decomposition[-1])),
#                 len(np.concatenate(decompositions_trained[0][0].overlap_decomposition[-1]))],yerr=1,color='purple', marker='s')
plt.boxplot(mean_overlap_reg)
plt.xticks([1,2],['initialization','trained'])
plt.savefig(f'../figures/n_points_region.png',dpi=300, transparent=True)

#%%Test statistics with Kruskal Wallis test
from scipy.stats import kruskal
pvals_overlap_volume = [kruskal(volume_overlap[2*i],volume_overlap[2*i+1])[1] for i in range(len(dims))]
pvals_n_overlaps = [kruskal(mean_overlaps[2*i],mean_overlaps[2*i+1])[1] for i in range(len(dims))]

#%% Plot statistics for other initializations 
decompositions_init2 = [[] for d in range(len(dims))]

for d in dims:
    X,y = construct_data(n_samples,dim=d)
    if d==1:
        X_points = X.clone()
    data_train = TensorDataset(X.T, y.long())
    trainloader = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    for n in range(n_models):
        model = FeedforwardNetwork(d+1, Hidden_layers, 
                                   out_layer_sz=2, init_type='orthogonal',activation=nn.ReLU)
        
        decomposer = NetworkDecompositions(model)
        decomposer.compute_overlap_decomp(X.T)
        decompositions_init2[d-1].append(decomposer)

#%%
volume_overlap_init2 = np.empty((len(dims)), dtype=object)
for d in range(len(dims)):
    volume_overlap_init2[d] = []
    for i in range(n_models):
        for n, poly in enumerate(decompositions_init2[d][i].polyhedral_decomposition[-1]):
            for ovr in decompositions_init2[d][i].overlap_decomposition[-1]:
                if len(np.intersect1d(decompositions_init2[d][i].global_decomposition[0][-1][n],ovr))>0:
                    volume_overlap_init2[d].append(poly.volume)
                    break

volume_overlap2 = []
for d in range(len(dims)):
    volume_overlap2.append(volume_overlap_init2[d])
    volume_overlap2.append(volume_overlap_trained[d])
    
fig,ax = plt.subplots(figsize=(3.5,2.5))

boxplt = ax.boxplot(volume_overlap2,showfliers=False, patch_artist=True,
            medianprops={'linewidth':0})#[np.median(volume_overlap_init[d]) for d in range(len(dims))])
for n,patch in enumerate(boxplt['boxes']):
    if n%2==0:
        patch.set_facecolor([1,0,1,0.4])
    else:
        patch.set_facecolor([0,1,1,0.4])
    patch.set_edgecolor('black')
    patch.set_linewidth(2)

    
    
[plt.plot([1+2*i,2*i+2],[np.median(volume_overlap_init2[i]),np.median(volume_overlap_trained[i])],'-^',color=clrs[i]) for i in range(len(dims))]
plt.yscale('log')
plt.ylim(10e-7, 10e0)
plt.xticks([])
fig.savefig(f'../figures/volumes_overlap_orthogonal.png',dpi=500, transparent=True)

#%%
mean_overlaps_init2 = [[len(decompositions_init2[d][i].overlap_decomposition[-1]) 
                      for i in range(n_models)] for d in range(len(dims))]
mean_overlaps2 = []
for i in range(len(dims)): #sort overlaps for plotting
    mean_overlaps2.append(mean_overlaps_init2[i])
    mean_overlaps2.append(mean_overlaps_trained[i])


fig,ax = plt.subplots(figsize=(3.5,2.5))
# plt.errorbar([0,1],[len(decompositions_init[0][0].overlap_decomposition[-1]),
#                 len(decompositions_trained[0][0].overlap_decomposition[-1])],yerr=1,color='brown', marker='s')
boxplt = ax.boxplot(mean_overlaps2,medianprops={'linewidth':0}, patch_artist=True)
for n,patch in enumerate(boxplt['boxes']):
    if n%2==0:
        patch.set_facecolor([1,0,1,0.4])
    else:
        patch.set_facecolor([0,1,1,0.4])
    patch.set_edgecolor('black')
    patch.set_linewidth(2)

[plt.plot([1+2*i,2*i+2],[np.median(mean_overlaps_init2[i]),np.median(mean_overlaps_trained[i])],'-^',color=clrs[i]) for i in range(len(dims))]
# plt.legend(['$S^1$','$S^2$','$S^3$'])
plt.plot([1,3,5],np.median(mean_overlaps_init2,1),'magenta',linestyle='--',alpha=0.2)
plt.xticks([])#np.arange(1,1+2*len(dims)),['initialization','trained']+['']*4,rotation=-15)
plt.savefig(f'../figures/n_points_region_orthogonal.png',dpi=300, transparent=True)

#%%
pvals_overlap_volume2 = [kruskal(volume_overlap2[2*i],volume_overlap2[2*i+1])[1] for i in range(len(dims))]
pvals_n_overlaps2 = [kruskal(mean_overlaps2[2*i],mean_overlaps2[2*i+1])[1] for i in range(len(dims))]

#%%save trained models
for i, m in enumerate(trained_models[0]):
    torch.save(m.state_dict(), f"../models/toy_model_{i}.pth")
    
#%%
mean_samples_curve_init = np.zeros([len(dims),len(Hidden_layers)])
mean_samples_curve_trained = np.zeros([len(dims),len(Hidden_layers)])

for d in range(len(dims)):
    for l in range(len(Hidden_layers)):
        mean_samples_curve_init[d,l] = np.mean(n_samples_in_poly_init[0,d,l])
        mean_samples_curve_trained[d,l] = np.mean(n_samples_in_poly_trained[0,d,l])
       
plt.plot(mean_samples_curve_init.T,'o-')
plt.plot(mean_samples_curve_trained.T,'o-')

#%% Overlap regions as a function of depth and width
widths = [5,10,15,20,25]
depths = [1,2,3,4]
decompositions_init_wd = [[[] for l in range(len(depths))] for d in range(n_models)]
overlaps_wd = np.zeros([n_models,len(widths), len(depths)])


for n in range(n_models):
    X,y = construct_data(n_samples,dim=1)
    if d==1:
        X_points = X.clone()

    for i,l in enumerate(depths):
        for j,w in enumerate(widths):        
            Hidden_layers = [w]*l
            model = FeedforwardNetwork(2, Hidden_layers, 
                                       out_layer_sz=2, init_type='none',activation=nn.ReLU)
            
            decomposer = NetworkDecompositions(model)
            decomposer.compute_overlap_decomp(X.T)
            decompositions_init_wd[n][i].append(decomposer)
            
            overlaps_wd[n,j,i] = len(decomposer.overlap_decomposition[-1])

#%%
poly_wd = np.zeros([n_models,len(widths), len(depths)])

for n in range(n_models):
    for i,l in enumerate(depths):
        for j,w in enumerate(widths):
            poly_wd[n,j,i] = len(decompositions_init_wd[n][i][j].polyhedral_decomposition[-1])
#%%
mean_poly = np.mean(poly_wd,0)
std_poly = np.std(poly_wd,0)


mean_overlaps = np.mean(overlaps_wd,0)
std_overlaps = np.std(overlaps_wd,0)

fig,ax  = plt.subplots(1,2,figsize=(9,6))

ax[0].imshow(mean_poly, cmap='coolwarm',origin='lower', aspect='auto')
ax[0].spines[:].set_visible(False)

# Set minor ticks at the grid line positions (but without labels)
ax[0].set_xticks(np.arange(len(depths) + 1) - 0.5, minor=True)  # Offset by -0.5 to align with edges
ax[0].set_yticks(np.arange(len(widths) + 1) - 0.5, minor=True)

# Set major ticks (optional, for labels)
ax[0].set_xticks(np.arange(len(depths)), labels=depths)
ax[0].set_yticks(np.arange(len(widths)), labels=widths)
for i in range(len(widths)):
    for j in range(len(depths)):
        ax[0].text(j,i,f"{int(round(mean_poly[i,j],0))} ± {int(round(std_poly[i,j],0))}",
                fontsize=13, ha='center',va='center')
        
ax[0].set_xlabel('depth', fontsize=14)
ax[0].set_ylabel('width', fontsize=14)

# ax.minorticks_on()
ax[0].grid(color='white', linestyle='-', linewidth=3, which='minor')
ax[0].tick_params(which="minor", bottom=False, left=False, labelbottom=False, labelleft=False)


ax[1].imshow(mean_overlaps, cmap='coolwarm',origin='lower', aspect='auto')
ax[1].spines[:].set_visible(False)

# Set minor ticks at the grid line positions (but without labels)
ax[1].set_xticks(np.arange(len(depths) + 1) - 0.5, minor=True)  # Offset by -0.5 to align with edges
ax[1].set_yticks(np.arange(len(widths) + 1) - 0.5, minor=True)

# Set major ticks (optional, for labels)
ax[1].set_xticks(np.arange(len(depths)), labels=depths)
ax[1].set_yticks(np.arange(len(widths)), labels=[])
for i in range(len(widths)):
    for j in range(len(depths)):
        ax[1].text(j,i,f"{int(round(mean_overlaps[i,j],0))} ± {int(round(std_overlaps[i,j],0))}",
                fontsize=13, ha='center',va='center')
        
ax[1].set_xlabel('depth', fontsize=14)
# ax[1].set_ylabel('width')

# ax.minorticks_on()
ax[1].grid(color='white', linestyle='-', linewidth=3, which='minor')
ax[1].tick_params(which="minor", bottom=False, left=False, labelbottom=False, labelleft=False)
ax[1].tick_params(which="major",  left=False, labelleft=False)
fig.tight_layout()
fig.savefig(f'../figures/depth_width_overlaps.png',dpi=300, transparent=True)
# [plt.errorbar(depths,mean_overlaps[i],yerr = std_overlaps[i]) for i in range(len(depths))]
