import sys
sys.path.append("/Users/kosio/Repos/network-relative-homology/src/")
from NetRelHom import *
from TopologicalMethods import *
from torch.utils.data import Dataset, TensorDataset
import torch.optim as optim
from tqdm.auto import tqdm
import persim

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

    
#%%Perform homology analysis 
models = []
decompositions = []
phom_layerwise = [[[] for i in range(len(Hidden_layers[model_type])+1)] for j in range(n_models)]
_, phom_data, _ = Phom.homology_analysis(X_test[np.where(y_test[:,1]==1)[0]], pairwise_distances, False, [1,None])
for i in tqdm(range(n_models)):
    model = FeedforwardNetwork(3, Hidden_layers[model_type], 
                               out_layer_sz=2, init_type='none',activation=nn.ReLU)
    model.load_state_dict(torch.load("../models/" + model_type+f"_model_{i}.pth"))
    model.eval()
    models.append(model)
    
    #Exclude missclassified points as they can induce incorrect gluings
    out = torch.argmax(model(X_test)[-1],1)
    print('Performance = ' + str(round(sum(out==np.where(y_test==1)[1]).item()/len(y_test),4)))
    X_correct = X_test[out==np.where(y_test==1)[1]]
    y_correct = y_test[out==np.where(y_test==1)[1]]
    
    #select Ma submanifold for analysis
    Ma = np.where(y_correct[:,1]==1)[0]
    X_correct = X_correct[Ma]
    y_correct = y_correct[Ma]
    

    
    decomposer = NetworkDecompositions(model)
    decomps = decomposer.compute_overlap_decomp(X_correct, sensitivity=0.5)
    decompositions.append(decomposer)
    
    for l in range(len(Hidden_layers[model_type])+1):
        if len(decomps[l])>0:
            print('Found non-trivial homology in model ' + str(i) + ', layer ' + str(l))
            _, diag, _ = Phom.relative_homology(X_correct, decomposer.overlap_decomposition[l],pairwise_distances,False,[1,None])
            phom_layerwise[i][l].append(diag)
        else:
            phom_layerwise[i][l].append(phom_data)


#%%
poly_decomp = decomposer.polyhedral_decomposition[-1]
region = pc.Region(poly_decomp)
region.plot()
_ = [plt.scatter(X_correct[clss,0],X_correct[clss,1], 
             c=np.linspace(0,1,len(decomps[-1]))[n]*np.ones(len(clss)),
             cmap='gist_ncar',vmin=0,vmax=1) 
             for n,clss in enumerate(decomps[-1])]

#%% Extract Betti numbers from each diagram using the parameters used in Naitzat

epsilon = 2.5 #This is the value used in Naitzat
def count_betti(diag, epsilon, dim=0):
    betti = sum(np.logical_and(diag[dim][:,0]<epsilon, diag[dim][:,1]>epsilon))
    return betti
Betti_numbers = np.zeros([n_models,len(Hidden_layers[model_type])+1])
for i in range(n_models):
    for j in range(len(Hidden_layers[model_type])+1):
        Betti_numbers[i,j] = count_betti(phom_layerwise[i][j][0],epsilon,dim=0)
#%%
average_betti = np.mean(Betti_numbers,0)
std_betti = np.std(Betti_numbers,0)
up_betti = average_betti+std_betti
down_betti = average_betti-std_betti
up_betti[up_betti>9] = 9

plt.figure()
plt.plot(Betti_numbers.T,'g',alpha=0.1)
plt.plot(average_betti,'g-o')
plt.fill_between(np.arange(0,10), down_betti, 
                up_betti, color='green',alpha=0.4)
plt.grid('on')
plt.savefig('../figures/betti_numbers' + str(dataset)+'model_'+str(model_type)+'png',dpi=500)