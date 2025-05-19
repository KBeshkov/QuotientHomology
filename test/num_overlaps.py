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
    overlap_hist = []
    performance_hist = []
    for epoch in tqdm(range(num_epochs)):
        if epoch % 200 == 0:
            performance = (torch.argmax(model(dat_loader.dataset.tensors[0])[-1],axis=1)==dat_loader.dataset.tensors[1]).sum()/len(dat_loader.dataset.tensors[1])
            print(f'Epoch [{epoch+1}/{num_epochs}], Performance: {performance:.5f}')
            performance_hist.append(performance.item())

            decomposer = NetworkDecompositions(model)
            decomps = decomposer.compute_overlap_decomp(dat_loader.dataset.tensors[0],sensitivity=100)
            overlap_hist.append(decomps)

        for dat in dat_loader:
            inputs, targets = dat
            # Forward pass
            outputs = model(inputs.reshape(len(inputs),-1))[-1]
            loss = crit(outputs, targets) #+ 0.001*torch.abs(outputs).sum() #require sparse outputs?
    
            # Backward pass and optimization
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_hist.append(loss.item())
    return outputs, loss_hist, overlap_hist, performance_hist

#%%Generate random 2d functions
from numpy.polynomial import chebyshev, legendre, polynomial, hermite, laguerre

res = 20
X,Y = np.meshgrid(np.linspace(0,1,res), np.linspace(0,1,res))
f = legendre.legval2d(X, Y, 2*(np.random.rand(5,5)-0.5))
f = f-np.mean(f)


classes = (f<0).astype(float)
classes[classes==0] = 2

plt.subplot(1,2,1)
plt.imshow(f)
plt.subplot(1,2,2)
plt.imshow(classes)

domain = np.vstack([X.flatten(),Y.flatten()])
Phom = PersistentHomology()
# hom = Phom.relative_homology(domain.T, [np.where(classes.flatten()==1)[0]], pairwise_distances, 0, [2,None])
# plt.figure()
# plot_diagrams(hom[1])
#%%
widths = [5,10,25,50]
n_models = 1
n_layers = 4
epochs = 1000
n_steps = epochs//200
n_overlaps = np.zeros([n_models, len(widths), n_layers+1, n_steps])
performance = np.zeros([n_models, len(widths), n_steps])
criterion = nn.CrossEntropyLoss()

for n in range(n_models):
    rand_fun = legendre.legval2d(X, Y, 2*(np.random.rand(5,5)-0.5)).flatten()
    rand_fun = (rand_fun-rand_fun.mean())/rand_fun.std()
    
    classes = torch.ones(len(rand_fun)).long()
    classes[rand_fun>np.percentile(rand_fun, 66.7)] = 2
    classes[rand_fun<np.percentile(rand_fun, 33.3)] = 0

    reg_data_train = TensorDataset(torch.Tensor(domain.T), 
                                   classes)
    trainloader = torch.utils.data.DataLoader(reg_data_train, batch_size=1, shuffle=True)
 
    for m,w in enumerate(widths):
        hidden_sizes = [w]*n_layers
        
        model = FeedforwardNetwork(input_size=2,hidden_sizes=hidden_sizes,
                                   out_layer_sz=3, init_type='none',activation=nn.ReLU)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        outputs, loss_hist, overlaps, performance_hist = optimize_nn(model, trainloader, optimizer, criterion, num_epochs=epochs)

        for s in range(n_steps):
            performance[n,m,s] = performance_hist[s]
            for l in range(n_layers+1):
                if len(overlaps[s][l])>1:
                    n_overlaps[n,m,l,s] = len(np.concatenate(overlaps[s][l]).squeeze())
                elif len(overlaps[s][l])==1:
                    n_overlaps[n,m,l,s] = len(overlaps[s][l][0])
            
#%%
fig, ax = plt.subplots(1,n_overlaps.shape[-1])
for l in range(n_overlaps.shape[-1]):
    ax[l].imshow(n_overlaps[0,:,:,l]/res**2,cmap='coolwarm',vmin=0,vmax=1)
    ax[l].axis('off')
# plt.tight_layout()
    

