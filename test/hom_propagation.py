import sys
sys.path.append("/Users/kosio/Repos/network-relative-homology/src/")
from NetRelHom import *
from TopologicalMethods import *
from torch.utils.data import Dataset, TensorDataset
import torch.optim as optim
from tqdm.auto import tqdm
import persim
import pickle
#%% Network parameters from Naitzat

Hidden_layers = {"D-I": [15]*9,
                 "D-II": [15]*9,
                 "D-II-2": [25]*9,
                 "D-III": [15]*9,
                 "D-III-2": [50]*9}

n_models = 30
dataset = "D-II"
model_type = "D-II"
Phom = PersistentHomology()

def count_betti(diag, epsilon, dim=0):
    betti = sum(np.logical_and(diag[dim][:,0]<epsilon, diag[dim][:,1]>epsilon))
    return betti

#%% Use the same metric as in Naitzat
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
def naitzat_metric(X,n_neighbors=0):
    #Code adapted from https://github.com/topnn/topnn_framework/blob/master/transformers/betti_calc.py
    if dataset=='D-I' and n_neighbors==0:
        n_neighbors = 14
    # elif dataset=='D-III' and n_neighbors==0:
    #     n_neighbors = 17
    elif n_neighbors==0:
        n_neighbors = 19
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                                    algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    A = nbrs.kneighbors_graph(X).toarray()
    d = shortest_path(A, method='auto', directed=False)
    
    #implementation of distance_mat_verb fixing disconnected parts
    d[d==np.inf] = 1000
    return d
#%% Load data
X_train = torch.Tensor(np.load("../data/" + dataset+"-train.npy"))

X_test = torch.Tensor(np.load("../data/" + dataset+"-test.npy"))

if dataset!="D-I":
    labels_train = torch.Tensor(np.load("../data/" + dataset+"-train-labels.npy"))
    y_train = torch.zeros([len(labels_train),2])
    y_train[labels_train==0,0] = 1
    y_train[labels_train==1,1] = 1
    
    labels_test = torch.Tensor(np.load("../data/" + dataset+"-test-labels.npy"))
    y_test = torch.zeros([len(labels_test),2])
    y_test[labels_test==0,0] = 1
    y_test[labels_test==1,1] = 1
else:
    y_train = torch.Tensor(np.load("../data/" + dataset+"-train-labels.npy"))
    y_test = torch.Tensor(np.load("../data/" + dataset+"-test-labels.npy"))

#The indices of the topology in the first dataset are flipped, different dataset need different parameters
if dataset=="D-I": 
    inp_dim = 2
    hom_dim  = 0
    n_perms = None
    a=1
    epsilon = 2.5
elif dataset=="D-II":
    inp_dim = 3
    hom_dim = 1
    n_perms = None
    a=1
    epsilon = 2.5
elif dataset=="D-III": #Need to sample more for this
    inp_dim = 3
    hom_dim = 2
    n_perms = None #subsample point cloud for improved speed in 3d
    a=0
    epsilon = 2.5
#%% Optimize k and epsilon to reproduce the results in Naitzat for smaller datasets
ks = np.arange(5,30)
epsilons = np.linspace(0.2,5,30)
bettis_0 = np.zeros([len(ks),len(epsilons)])
bettis_1 = np.zeros([len(ks),len(epsilons)])
bettis_2 = np.zeros([len(ks),len(epsilons)])

for i,k in enumerate(ks):
    d = naitzat_metric(X_test[np.where(y_test[:,a]==1)[0]],k)
    for j,e in enumerate(epsilons):
        phom = tda(d, distance_matrix=True, maxdim=hom_dim, thresh=e, n_perm=n_perms)['dgms']
        bettis_0[i,j] = count_betti(phom,e)
        if hom_dim>0:
            bettis_1[i,j] = count_betti(phom,e,1)
            if hom_dim==2:
                bettis_2[i,j] = count_betti(phom,e,2)
    print(k)
    
if dataset=='D-I':
    valid_condition = bettis_0==9 #k=14 and 2.5 work well (as in Naitzat et al.)
elif dataset=='D-II':
    valid_condition =np.logical_and(bettis_0==9,bettis_1==9) #k=19 and epsilon=2.5 is a good choice
elif dataset=='D-III':
    valid_condition =np.logical_and(bettis_0==9,np.logical_and(bettis_1==0,bettis_2==9)) #k=19, epislon=2.5 is a good choice
    
valid_params = np.where(valid_condition)

#%%
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(valid_condition, cmap='Greens', interpolation='none', )
if dataset=="D-I":
    rect = patches.Rectangle(
        (14 - 0.5, 9 - 0.5),  # Bottom-left corner of the square
        1, 1,  # Width and height of the square
        linewidth=2, edgecolor='red', facecolor='none'  # Styling
    )
else:
    rect = patches.Rectangle(
        (14 - 0.5, 14 - 0.5),  # Bottom-left corner of the square
        1, 1,  # Width and height of the square
        linewidth=2, edgecolor='red', facecolor='none'  # Styling
    )

ax.add_patch(rect)

ax.set_xticks(np.arange(0,30))
ax.set_xticklabels([round(epsilons[i],1) for i in range(len(epsilons))], rotation=60)
ax.set(yticks=np.arange(0,25), yticklabels=ks)
ax.set_xlabel('$\epsilon$')
ax.set_ylabel('k')
ax.set_title(f'Valid parameters for {dataset}')
# plt.savefig(f'../figures/valid_param_regimes_{dataset}.png',dpi=500,transparent=True)
#%%main decomposition and homology loop
models = []
decompositions = []

    
_, phom_data, _ = Phom.homology_analysis(X_test[np.where(y_test[:,a]==1)[0]], naitzat_metric, False, [hom_dim,n_perms],thresh=epsilon)

for i in tqdm(range(n_models)):
    model = FeedforwardNetwork(inp_dim, Hidden_layers[model_type], 
                               out_layer_sz=2, init_type='none',activation=nn.ReLU)
    model.load_state_dict(torch.load("../models/" + model_type+f"_model_{i}.pth"))
    model.eval()
    models.append(model)
    
    #Exclude missclassified points as they can induce incorrect gluings
    out = torch.argmax(model(X_test)[-1],1)
    print('Performance = ' + str(round(sum(out==np.where(y_test==1)[1]).item()/len(y_test),4)))
    X_correct = X_test[out==np.where(y_test==1)[1]]
    y_correct = y_test[out==np.where(y_test==1)[1]]
    print(X_correct.shape)
    
    #select Ma submanifold for analysis
    Ma = np.where(y_correct[:,a]==1)[0]
    X_correct = X_correct[Ma]
    y_correct = y_correct[Ma]
    
    out_corr = model(X_correct)
    
    
    decomposer = NetworkDecompositions(model)
    decomps = decomposer.compute_overlap_decomp(X_correct, sensitivity=10)
    decompositions.append(decomposer)
    


#%% Save/load decompositions
mode = 'load'
if mode=='save':
    for i,dec in enumerate(decompositions):
        with open("../models/Decompositions/Decomp_model"+model_type+f"_{i}.pkl", 'wb') as outp:
            pickle.dump(dec, outp, pickle.HIGHEST_PROTOCOL)
else:
    decompositions = []
    for i in range(n_models):
        with open("../models/Decompositions/Decomp_model"+model_type+f"_{i}.pkl", 'rb') as outp:
            decompositions.append(pickle.load(outp))
            
#%%Check whether the rank decomposition matters
ranks = []
for i in tqdm(range(n_models)):
    model = FeedforwardNetwork(inp_dim, Hidden_layers[model_type], 
                               out_layer_sz=2, init_type='none',activation=nn.ReLU)
    model.load_state_dict(torch.load("../models/" + model_type+f"_model_{i}.pth"))
    model.eval()
    
    #Exclude missclassified points as they can induce incorrect gluings
    out = torch.argmax(model(X_test)[-1],1)
    print('Performance = ' + str(round(sum(out==np.where(y_test==1)[1]).item()/len(y_test),4)))
    X_correct = X_test[out==np.where(y_test==1)[1]]
    y_correct = y_test[out==np.where(y_test==1)[1]]
    

    ranks.append(np.concatenate(decompositions[i].compute_rank_classes(X_correct)[1]))
    print(ranks[-1].std())
    
mean_ranks = np.array([np.mean(ranks[i]) for i in range(30)])
std_ranks = np.array([np.std(ranks[i]) for i in range(30)])
upper_std = np.minimum(std_ranks, len(X_correct.T)-mean_ranks)
#%%histogram of ranks
fig = plt.figure(figsize=(1.67,3))
plt.errorbar(mean_ranks,np.arange(0,30),xerr=[std_ranks,upper_std],markersize=2,marker='s',color='black',ls='none')
# plt.xlim(2.7,3.25)
plt.yticks([])
fig.savefig(f'../figures/ranks_{dataset}_model_{model_type}.png',dpi=300, transparent=True)
#%%Compute relative homology groups
phom_layerwise = [[[] for i in range(len(Hidden_layers[model_type])+1)] for j in range(n_models)]
phom_layer_naitz = [[[] for i in range(len(Hidden_layers[model_type])+1)] for j in range(n_models)]
for i in range(n_models):
    model = FeedforwardNetwork(inp_dim, Hidden_layers[model_type], 
                               out_layer_sz=2, init_type='none',activation=nn.ReLU)
    model.load_state_dict(torch.load("../models/" + model_type+f"_model_{i}.pth"))
    model.eval()

    #Exclude missclassified points as they can induce incorrect gluings
    out = torch.argmax(model(X_test)[-1],1)
    print('Performance = ' + str(round(sum(out==np.where(y_test==1)[1]).item()/len(y_test),4)))
    X_correct = X_test[out==np.where(y_test==1)[1]]
    y_correct = y_test[out==np.where(y_test==1)[1]]
    
    #select Ma submanifold for analysis
    Ma = np.where(y_correct[:,a]==1)[0]
    X_correct = X_correct[Ma]
    y_correct = y_correct[Ma]
    
    out_corr = model(X_correct)


    for l in range(len(Hidden_layers[model_type])+1):
        _, diag_naitz, _ = Phom.homology_analysis(out_corr[l].detach().numpy(), naitzat_metric,False, [hom_dim,n_perms],thresh=epsilon)
        phom_layer_naitz[i][l].append(diag_naitz)
        if len(decompositions[i].overlap_decomposition[l])>0:
            print('Found non-trivial homology in model ' + str(i) + ', layer ' + str(l))
            try:
                _, diag, _ = Phom.relative_homology(X_correct, decompositions[i].overlap_decomposition[l],naitzat_metric,False,[hom_dim,n_perms],thresh=epsilon)
            except:
                _, diag, _ = Phom.relative_homology(X_correct, decompositions[i].overlap_decomposition[l],naitzat_metric,False,[hom_dim,None],thresh=epsilon)
            phom_layerwise[i][l].append(diag)
        else:
            phom_layerwise[i][l].append(phom_data)

#%%
poly_decomp = decompositions[1].polyhedral_decomposition[-1]#decomposer.polyhedral_decomposition[-1]
region = pc.Region(poly_decomp)
region.plot(color='gray')
_ = [plt.scatter(X_correct[clss,0],X_correct[clss,1], 
             c=np.linspace(0,1,len(decomps[-1]))[n]*np.ones(len(clss)),
             cmap='gist_ncar',vmin=0,vmax=1) 
             for n,clss in enumerate(decompositions[1].overlap_decomposition[-1])]

#%% Extract Betti numbers from each diagram using the parameters used in Naitzat

betti_num = 0
Betti_numbers = np.zeros([n_models,len(Hidden_layers[model_type])+1])
Betti_numbers_naitz = np.zeros([n_models,len(Hidden_layers[model_type])+1])
for i in range(n_models):
    for j in range(len(Hidden_layers[model_type])+1):
        Betti_numbers[i,j] = count_betti(phom_layerwise[i][j][0],epsilon,dim=betti_num)
        Betti_numbers_naitz[i,j] = count_betti(phom_layer_naitz[i][j][0],epsilon,dim=betti_num)
#%%
average_betti = np.mean(Betti_numbers,0)
std_betti = np.std(Betti_numbers,0)/2


average_betti_naitz = np.mean(Betti_numbers_naitz,0)
std_betti_naitz = np.std(Betti_numbers_naitz,0)/2

plt.figure(figsize=(5,3))
plt.plot(average_betti,'k-*')
plt.plot(average_betti_naitz,'g-o')
plt.legend(['Relative Homology', 'Naitzat et al. 2020'],  loc=3, prop={'size': 8})# plt.plot(Betti_numbers.T,'g',alpha=0.1)
plt.fill_between(np.arange(0,10), average_betti+std_betti, 
                average_betti-std_betti, color='black',alpha=0.1)
plt.fill_between(np.arange(0,10), average_betti_naitz + std_betti_naitz, 
                average_betti_naitz - std_betti_naitz, color='green',alpha=0.1)

plt.grid('on')
plt.ylim(0, 10)

plt.savefig(f'../figures/betti_numbers{dataset}_model_{model_type}_{betti_num}.png',dpi=500, transparent=True)

#%% Plot curves counting the number of overlapping regions
num_overlap_regions = np.zeros([len(decompositions),len(Hidden_layers[model_type])+1])
for i in range(n_models):
    for j in range(len(Hidden_layers[model_type])+1):
        num_overlap_regions[i,j] = len(decompositions[i].overlap_decomposition[j])

mean_overlap_num = np.mean(num_overlap_regions,0)
std_overlap_num = np.std(num_overlap_regions,0)/2

plt.figure(figsize=(5,3))
plt.plot(mean_overlap_num,'o-',color='red')
plt.fill_between(np.arange(0,10), mean_overlap_num+std_overlap_num, mean_overlap_num-std_overlap_num,
                 color='red',alpha=0.4)
plt.grid('on')
plt.savefig(f'../figures/number_overlaps_{dataset}_model_{model_type}_{betti_num}.png',dpi=500, transparent=True)

#%%Plot dimensionality reduced gluing manifolds
if dataset=="D-I":
    reducer = Isomap(n_components = 2, n_neighbors = 14, metric='precomputed')
else:
    reducer = Isomap(n_components = 3, n_neighbors = 19, metric='precomputed')

d_inp = naitzat_metric(X_correct)
manifolds = []
for l in range(len(Hidden_layers[model_type])+1):
    if len(decompositions[-1].overlap_decomposition[l])>0:
        d, _, _ = Phom.relative_homology(X_correct, decompositions[-1].overlap_decomposition[l],naitzat_metric,False,[hom_dim,n_perms],thresh=epsilon)
    else:
        d = d_inp
    mfld = reducer.fit_transform(d)
    manifolds.append(mfld)

fig = plt.figure(figsize=(7,2))
for i in range(len(manifolds)):
    mfld = manifolds[i].T
    if dataset=='D-I':
        ax = fig.add_subplot(1,len(manifolds),i+1)
        ax.scatter(mfld[0],mfld[1], s=1)
    else:
        ax = fig.add_subplot(1,len(manifolds),i+1, projection='3d')
        ax.scatter(mfld[0],mfld[1],mfld[2], s=1)
    ax.axis('off')
fig.tight_layout()