import sys
sys.path.append("/Users/kosio/Repos/network-relative-homology/src/")
from NetRelHom import *
from TopologicalMethods import *
from torch.utils.data import Dataset, TensorDataset
import torch.optim as optim
from tqdm import tqdm

def optimize_nn(model, dat_loader, optim, crit, num_epochs = 100):
    loss_hist = []
    for epoch in tqdm(range(num_epochs)):
        for inputs, targets in dat_loader:
            # Forward pass
            outputs = model(inputs)[-1]
            loss = crit(outputs, targets.unsqueeze(1))
    
            # Backward pass and optimization
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_hist.append(loss.item())
    
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.5f}')
    return outputs, loss_hist
#%% Test performance by fitting a neural network on torus
Mfld = ManifoldGenerator()
n_samples = 400
X_train = torch.meshgrid([torch.linspace(0,1,int(np.sqrt(n_samples))),#torch.linspace(0,1,n_samples)[:,None]#
                         torch.linspace(0,1,int(np.sqrt(n_samples)))])#2*(torch.randn(n_samples,2)-0.5)
neighbor_scale = np.sqrt(2*torch.linspace(0,1,int(np.sqrt(n_samples)))[1].item()**2)+1e-2

pcloud = np.column_stack([X_train[0].ravel(),X_train[1].ravel()])
sc = SimpComp(threshold=neighbor_scale,dim=3,method='Rips')
tree = sc.assign_complex(pcloud)
sc.plot_simplicial_complex(pcloud,tree,[[0,2,32,71],[1,9]])

#X_train = torch.meshgrid([torch.linspace(0,1,int(np.cbrt(n_samples))),
#                          torch.linspace(0,1,int(np.cbrt(n_samples))),
#                         torch.linspace(0,1,int(np.cbrt(n_samples))),])#2*(torch.randn(n_samples,2)-0.5)
X_train = torch.vstack([X_train[0].flatten(),X_train[1].flatten()]).T#,X_train[2].flatten()]).T
X_train = X_train+0.0001*torch.randn(X_train.shape)
#X_train = torch.linspace(0,1,n_samples)[None].T+0.01*torch.randn(n_samples)[None].T
y_train = torch.Tensor(Mfld.TN(int(np.sqrt(n_samples)),1,2)).T#torch.Tensor(Mfld.S1(n_samples,1)).T#torch.Tensor(Mfld.T2(int(np.sqrt(n_samples)),1,0.6)).T#t
y_train = y_train+0.001*torch.randn(*y_train.shape)
reg_data_train = TensorDataset(X_train, y_train)
trainloader = torch.utils.data.DataLoader(reg_data_train, batch_size=1, shuffle=True)

hidden_sizes=[50,50,50,5]
model = FeedforwardNetwork(input_size=2,hidden_sizes=hidden_sizes,#8*f_dim,4*f_dim,2*f_dim], 
                               out_layer_sz=4, init_type='none',activation=nn.ReLU)

#%%
decomposer = NetworkDecompositions(model)
decomps = decomposer.compute_all_decomps(X_train)
polyh_decomp = decomposer.polyhedral_decomposition#weights_to_inpolyhedra(X_train,layer=len(hidden_sizes),bbox=[-100,100])
#%%
for l in range(len(hidden_sizes)+1):
    region = pc.Region(polyh_decomp[l])
    region.plot()
    plt.scatter(X_train[:,0],X_train[:,1],c='k')
    plt.xlim(-0.01,1.01)
    plt.ylim(-0.01,1.01)
    # [plt.scatter(X_train[clss,0],X_train[clss,1]) for clss in decomposer.global_decomposition[0][l]]
    [plt.scatter(X_train[clss,0],X_train[clss,1]) for clss in decomposer.overlap_decomposition[l]]
#%%
optimizer = optim.Adam( model.parameters(), lr=0.00001)
criterion = nn.MSELoss()
outputs, loss_hist = optimize_nn(model, trainloader, optimizer, criterion, num_epochs=1000)

#%%
X_test = torch.meshgrid([torch.linspace(-0.4,1.4,int(1.5*np.sqrt(n_samples))),#torch.linspace(-0.5,1.5,n_samples)[:,None]#
                                   torch.linspace(-0.4,1.4,int(1.5*np.sqrt(n_samples)))])#,torch.linspace(-0.2,1.2,20)])#
X_test = torch.vstack([X_test[0].flatten(),X_test[1].flatten()]).T#,X_test[2].flatten()]).T

decomposer = NetworkDecompositions(model)
decomps = decomposer.compute_all_decomps(X_test)
polyh_decomp = decomposer.polyhedral_decomposition#weights_to_inpolyhedra(X_train,layer=len(hidden_sizes),bbox=[-100,100])
#%%
for l in range(len(hidden_sizes)+1):
    region = pc.Region(polyh_decomp[l])
    region.plot()
    plt.scatter(X_test[:,0],X_test[:,1],c='blue',s=0.5)
    plt.xlim(-0.21,1.21)
    plt.ylim(-0.21,1.21)
    # [plt.scatter(X_train[clss,0],X_train[clss,1]) for clss in decomposer.global_decomposition[0][l]]
    [plt.scatter(X_test[clss,0],X_test[clss,1]) for clss in decomposer.overlap_decomposition[l]]
    
    
#%% Superresolved sampling
new_pcloud = np.vstack([decomposer.populate_polyhedra((H.A,H.b),100) for H in polyh_decomp[-1]])
#%%
MPH =  MapHomology(model, X_test.detach().numpy(), scomplex_analyzer=sc, gluing_threshold=0, neural_model=model)
map_homology, fs, ovrlap_complex = MPH.map_homology_sequence(len(hidden_sizes))
print(map_homology)
get_all_decomps = MPH.decomps.compute_all_decomps(X_test,sensitivity=0)


# [plt.plot(fs[i],'.') for i in range(len(fs))]
reduced_mfld = []
reducer = Isomap(n_components=4)
for i in range(len(fs)):
    pcl = reducer.fit_transform(fs[i])
    plt.scatter(pcl[:,0], pcl[:,1],s=1)   
    reduced_mfld.append(pcl)     

for i in range(len(ovrlap_complex)):
    plt.figure()
    # sc.plot_simplicial_complex(pcloud, tree, ovrlap_complex[i])
    sc.plot_simplicial_complex(X_test.detach().numpy(), tree, MPH.decomps.overlap_decomopsition[i],MPH.decomps.global_decomposition[0][i])
    # sc.plot_simplicial_complex(reduced_mfld[i], tree, MPH.decomps.overlap_decomopsition[i],MPH.decomps.rank_decomposition[0][i-1])
#%%
out = model(X_train)[-1].detach().numpy().T
if out.shape[0]>2:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the scatter plot
    # [ax.scatter(out[0,clss],out[1,clss],out[2,clss]) for clss in decomposer.global_decomposition[0][-1]]
    [ax.scatter(out[0,clss],out[1,clss],out[2,clss]) for clss in decomposer.overlap_decomposition[-1]]
else:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # [ax.scatter(out[0,clss],out[1,clss])for clss in decomposer.local_decomposition[0][-1]]
    [ax.scatter(out[0,clss],out[1,clss],s=5)for clss in decomposer.overlap_decomposition[-1]]