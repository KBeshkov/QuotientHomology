from QuotientHomology import *
from torch.utils.data import Dataset, TensorDataset
import torch.optim as optim
from tqdm import tqdm
import persim

def optimize_nn(model, dat_loader, val_loader, optim, crit, num_epochs = 100):
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
        if (epoch+1) % 20 == 0:
            out_val = model(val_loader.dataset.tensors[0])[-1]
            loss_val = crit(out_val,val_loader.dataset.tensors[1])
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_val.item():.5f}')
            if loss_val.item()<0.00002:
                return outputs, loss_hist
    return outputs, loss_hist
#%%
#Test random knots
n_samples = 500
n_knots = 1
Mfld = ManifoldGenerator()
Phom = PersistentHomology()
rand_params = 2*(np.random.rand(n_knots,4)-1)
random_knots = [Mfld.ParametricKnot(n_samples, params[0], 
                                    params[1], params[0], 
                                    params[1], [1]) for params in rand_params]
random_knots_test = [Mfld.ParametricKnot(2*n_samples, params[0], 
                                    params[1], params[0], 
                                    params[1], [1]) for params in rand_params]

plt.figure(figsize=(2*n_knots,2))
for i in range(n_knots):
    plt.subplot(1,n_knots,i+1)
    plt.plot(random_knots[i][0], random_knots[i][1])
    plt.axis('off')

#%%
criterion = nn.MSELoss()

X_train = torch.linspace(-torch.pi,torch.pi,n_samples)[:,None]#torch.Tensor(Mfld.S1(n_samples,1)).T#
# X_train = torch.vstack([X_train,0*X_train]).T
X_test = torch.linspace(-torch.pi,torch.pi,2*n_samples)[:,None]#torch.Tensor(Mfld.S1(n_samples,1)).T#
# X_test = torch.vstack([X_test,0*X_test]).T

#%%
hidden_sizes=[8]*3
out_knots = []
decompositions = []
homs = []
out_homs = []
bottlenecks = []

for i,knot in enumerate(random_knots):
    reg_data_train = TensorDataset(X_train, torch.Tensor(knot.T+1))
    trainloader = torch.utils.data.DataLoader(reg_data_train, batch_size=1, shuffle=True)
    
    reg_data_val = TensorDataset(X_test, torch.Tensor(random_knots_test[i].T+1))
    valloader = torch.utils.data.DataLoader(reg_data_val, batch_size=1, shuffle=True)
    
    model = FeedforwardNetwork(input_size=1,hidden_sizes=hidden_sizes,
                                   out_layer_sz=2, init_type='none',activation=nn.ReLU)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    outputs, loss_hist = optimize_nn(model, trainloader, valloader, optimizer, criterion, num_epochs=1000)
    
    out_knots.append(model(X_test)[-1].detach().numpy().T)


    decomposer = NetworkDecompositions(model)
    decomps = decomposer.compute_overlap_decomp(X_test,sensitivity=10000)
    decompositions.append(decomposer)
    
    d, diag, cycles = Phom.relative_homology(X_test, decomps[-1],pairwise_distances,False,[1,None])
    d_out, diag_out, _ = Phom.homology_analysis(out_knots[-1].T, pairwise_distances, False, [1,None])
    
    homs.append(diag)
    out_homs.append(Phom.normalize(diag_out))
    
    # distance_bottleneck, matching = persim.bottleneck(diag[1],diag_out[1], matching=True)
    # bottlenecks.append(matching)

#%%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(2*n_knots,4))
gs = gridspec.GridSpec(4, 2*n_knots, figure=fig)
for i in range(len(random_knots)): 
    ax = fig.add_subplot(gs[:2,2*i:2*i+2])
    ax.plot(1+random_knots[i][0], 1+random_knots[i][1])
    ax.plot(out_knots[i][0], out_knots[i][1],'k',lw=1)
    for n, clss in enumerate(decompositions[i].overlap_decomposition[-1]):
        ax.scatter(out_knots[i][0,clss],out_knots[i][1,clss],cmap='gist_ncar', s=7,
                 c=(n*np.ones(len(clss)))/len(decompositions[i].overlap_decomposition[-1]),
                 vmin=0,vmax=1,zorder=1000)
        # ax[0,i].plot(out_knots[i][0,clss],out_knots[i][1,clss],'k')
    ax.axis('off')
    
    # plt.subplot(3,n_knots,n_knots+i+1)
    # for n, clss in enumerate(decompositions[i].overlap_decomposition[-1]):
    #     plt.scatter(X_test[clss,0],n+X_test[clss,1],cmap='gist_ncar',
    #              c=(n*np.ones(len(clss)))/len(decompositions[i].overlap_decomposition[-1]),
    #              vmin=0,vmax=1,zorder=1000,s=10)
    #     plt.plot(X_test[clss,0], n + X_test[clss,1],'k',lw=5)
    # plt.axis('off')
    
    ax = fig.add_subplot(gs[2:,2*i:2*i+1])
    ax.plot(X_test[:,1],X_test[:,0],lw=5,color='gray')
    for n, clss in enumerate(decompositions[i].overlap_decomposition[-1]):
        ax.scatter(X_test[clss,1],X_test[clss,0],cmap='gist_ncar',
                  c=(n*np.ones(len(clss)))/len(decompositions[i].overlap_decomposition[-1]),
                  vmin=0,vmax=1,zorder=1000,s=10)
    ax.axis('off')
    
    
    # sub_ax = inset_axes(parent_axes=ax[0,i],borderpad=1, width="50%",height="5%")
    # sub_ax.plot(X_test[:,0],X_test[:,1])
    # for n, clss in enumerate(decompositions[i].overlap_decomposition[-1]):
    #     sub_ax.scatter(X_test[clss,0],X_test[clss,1],cmap='gist_ncar',
    #               c=(n*np.ones(len(clss)))/len(decompositions[i].overlap_decomposition[-1]),
    #               vmin=0,vmax=1,zorder=1000,s=10)

    # sub_ax.axis('off')

    # persim.bottleneck_matching(homs[i][1], out_homs[i][1], bottlenecks[i])
    ax = fig.add_subplot(gs[2,2*i+1:2*i+2])
    for n,interval in enumerate(out_homs[i][1]):
        ax.plot([interval[0],interval[1]],[n,n],'g-s',clip_on=False,markersize=0.01)
    # plot_diagrams(homs[i],legend=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax = fig.add_subplot(gs[3,2*i+1:2*i+2])
    for n,interval in enumerate(homs[i][1]):
        ax.plot([interval[0],interval[1]],[n,n],'k-s',clip_on=False,markersize=0.01)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


fig.tight_layout()
# fig.savefig(f'../figures/knots_many.pdf',dpi=500, transparent=True)