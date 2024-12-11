import sys
sys.path.append("/Users/kosio/Repos/network-relative-homology/src/")
from NetRelHom import *
from TopologicalMethods import *
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
            loss = crit(outputs, targets.unsqueeze(1))
    
            # Backward pass and optimization
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_hist.append(loss.item())
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.5f}')
    return outputs, loss_hist
#%%
#Test random knots
n_samples = 750
n_knots = 1
Mfld = ManifoldGenerator()
Phom = PersistentHomology()
rand_params = 2*(np.random.rand(n_knots,4)-1)
random_knots = [Mfld.ParametricKnot(n_samples, params[0], 
                                    params[1], params[0], 
                                    params[1], [1]) for params in rand_params]

plt.figure(figsize=(2*n_knots,2))
for i in range(n_knots):
    plt.subplot(1,n_knots,i+1)
    plt.plot(random_knots[i][0], random_knots[i][1])
    plt.axis('off')

#%%
criterion = nn.MSELoss()

X_train = torch.linspace(-torch.pi,torch.pi,n_samples)#[:,None]#torch.Tensor(Mfld.S1(n_samples,1)).T#
X_train = torch.vstack([X_train,0*X_train]).T
X_test = torch.linspace(-torch.pi,torch.pi,2*n_samples)#torch.Tensor(Mfld.S1(n_samples,1)).T#
X_test = torch.vstack([X_test,0*X_test]).T

#%%
hidden_sizes=[30]*4
out_knots = []
decompositions = []
homs = []
out_homs = []
bottlenecks = []

for i,knot in enumerate(random_knots):
    reg_data_train = TensorDataset(X_train, torch.Tensor(knot.T+1))
    trainloader = torch.utils.data.DataLoader(reg_data_train, batch_size=1, shuffle=True)
    
    model = FeedforwardNetwork(input_size=2,hidden_sizes=hidden_sizes,
                                   out_layer_sz=2, init_type='none',activation=nn.ReLU)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    outputs, loss_hist = optimize_nn(model, trainloader, optimizer, criterion, num_epochs=300)
    
    out_knots.append(model(X_test)[-1].detach().numpy().T)


    decomposer = NetworkDecompositions(model)
    decomps = decomposer.compute_overlap_decomp(X_test,1)
    decompositions.append(decomposer)
    
    d, diag, cycles = Phom.relative_homology(X_test, decomps[-1],pairwise_distances,False,[1,None])
    d_out, diag_out, _ = Phom.homology_analysis(out_knots[-1].T, pairwise_distances, False, [1,None])
    
    homs.append(diag)
    out_homs.append(Phom.normalize(diag_out))
    
    # distance_bottleneck, matching = persim.bottleneck(diag[1],diag_out[1], matching=True)
    # bottlenecks.append(matching)

#%%
plt.figure(figsize=(2*n_knots,3))
for i in range(len(random_knots)): 
    plt.subplot(3,n_knots,i+1)
    plt.plot(1+random_knots[i][0], 1+random_knots[i][1])
    plt.plot(out_knots[i][0], out_knots[i][1],'.')
    for n, clss in enumerate(decompositions[i].overlap_decomposition[-1]):
        plt.scatter(out_knots[i][0,clss],out_knots[i][1,clss],cmap='tab20b',
                 c=(n*np.ones(len(clss)))/len(decompositions[i].overlap_decomposition[-1]),
                 vmin=0,vmax=1,zorder=1000)
        plt.plot(out_knots[i][0,clss],out_knots[i][1,clss],'k')
    plt.axis('off')
    
    plt.subplot(3,n_knots,n_knots+i+1)
    for n, clss in enumerate(decompositions[i].overlap_decomposition[-1]):
        plt.scatter(X_test[clss,0],n+X_test[clss,1],cmap='tab20b',
                 c=(n*np.ones(len(clss)))/len(decompositions[i].overlap_decomposition[-1]),
                 vmin=0,vmax=1,zorder=1000,s=10)
        plt.plot(X_test[clss,0], n + X_test[clss,1],'k',lw=5)
    plt.axis('off')
    
    plt.subplot(3,n_knots,2*n_knots + i + 1)
    # persim.bottleneck_matching(homs[i][1], out_homs[i][1], bottlenecks[i])
    plot_diagrams(homs[i],legend=False)
    # plot_diagrams(out_homs[i],legend=False)
    plt.axis('off')

