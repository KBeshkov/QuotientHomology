from QuotientHomology import *
import torch
import polytope as pc


#%% rank solution of XOR
XOR_data = torch.Tensor([[0,0],[1,1],[1,0],[0,1]])

model = FeedforwardNetwork(input_size=2,hidden_sizes=[2],
                               out_layer_sz=1, activation=nn.ReLU)
model.layers[0].weight = torch.nn.Parameter(torch.Tensor([[-1,1],[1,-1]]))
model.layers[0].bias = torch.nn.Parameter(torch.Tensor([-0.5,-0.5]))
model.layers[2].weight = torch.nn.Parameter(torch.Tensor([[2],[2]]).T)
model.layers[2].bias = torch.nn.Parameter(torch.Tensor([0]))
out = model(XOR_data)
out = out[0].detach()
decomposer = NetworkDecompositions(model)
decomposer.compute_codeword_eq_classes(XOR_data,mode='local')
decomposer.compute_codeword_eq_classes(XOR_data,mode='global')
polyhedra =decomposer.weights_to_inpolyhedra(XOR_data,layer=1,bbox=[-0.1,1.1])

x_new = np.linspace(-5,5,100)
y_new = -x_new+0.3

clrs = ['#f6f6e6','#5ab4ac','#d8b365']

fig,ax = plt.subplots(1,2,dpi=300,figsize=(4,2.2))
[poly.plot(ax=ax[0],color=clrs[i]) for i,poly in enumerate(polyhedra[0])]
ax[0].plot(XOR_data[:2,0],XOR_data[:2,1],'bo')
ax[0].plot(XOR_data[2:,0],XOR_data[2:,1],'rx')
ax[0].set_xlim(-0.09,1.09)
ax[0].set_ylim(-0.09,1.09)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].vlines(0,0,2,'black',)
ax[1].hlines(0,0,2,'black',)
ax[1].plot(x_new,y_new,'k--')
ax[1].plot(out[:2,0],out[:2,1],'bo')
ax[1].plot(out[2:,0],out[2:,1],'rx')
ax[1].set_xlim(-0.05,0.6)
ax[1].set_ylim(-0.05,0.6)
ax[1].set_xticks([])
ax[1].set_yticks([])
# ax[1].grid('on')
fig.tight_layout()
fig.savefig('../figures/XOR_rank.png', dpi=1000,transparent=True)


#%% overlap solution of XOR

model = FeedforwardNetwork(input_size=2,hidden_sizes=[4,2],
                               out_layer_sz=1, activation=nn.ReLU)
model.layers[0].weight = torch.nn.Parameter(torch.Tensor([[2,1],[1,2],[-2,-1],[-1,-2]]))
model.layers[0].bias = torch.nn.Parameter(torch.Tensor([-0.5,-0.5,2.5,2.5]))
model.layers[2].weight = torch.nn.Parameter(torch.Tensor([[1,-1,0,0],[0,0,1,-1]]))
model.layers[2].bias = torch.nn.Parameter(torch.Tensor([0,0]))

out = model(XOR_data)
out = out[1].detach()
decomposer = NetworkDecompositions(model)
decomposer.compute_codeword_eq_classes(XOR_data,mode='local')
decomposer.compute_codeword_eq_classes(XOR_data,mode='global')
polyhedra = decomposer.weights_to_inpolyhedra(XOR_data,layer=2,bbox=[-0.1,1.1])
overlaps = decomposer.compute_overlap_decomp(XOR_data,sensitivity=0.1)

sampled_polyhedra = [model(torch.Tensor(decomposer.populate_polyhedra([poly.A,poly.b])))[1].detach().numpy() for poly in polyhedra[1]]

x_new = np.linspace(-5,5,100)
y_new = -x_new+0.5

clrs = ['#f6f6e6','#5ab4ac','#d8b365','#af8dc3','#7fbf7b']

fig,ax = plt.subplots(1,2,dpi=300,figsize=(4,2.2))
# [poly.plot(ax=ax[0],color=clrs[i]) for i,poly in enumerate(polyhedra[0])]
[poly.plot(ax=ax[0],color=clrs[i]) for i,poly in enumerate(polyhedra[1])]
ax[0].plot(XOR_data[:2,0],XOR_data[:2,1],'bo')
ax[0].plot(XOR_data[2:,0],XOR_data[2:,1],'rx')
ax[0].set_xlim(-0.09,1.09)
ax[0].set_ylim(-0.09,1.09)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].vlines(0,0,2,'black',)
ax[1].hlines(0,0,2,'black',)
ax[1].plot(x_new,y_new,'k--')
ax[1].plot(out[:2,0],out[:2,1],'bo')
ax[1].plot(out[2:,0],out[2:,1],'rx')
ax[1].set_xlim(-0.05,1.1)
ax[1].set_ylim(-0.05,1.1)
ax[1].set_xticks([])
ax[1].set_yticks([])
# [ax[2].plot(sampled_poly[:,0],sampled_poly[:,1],'.',color=clrs[i]) for i,sampled_poly in enumerate(sampled_polyhedra)]
# ax[2].set_xticks([])
# ax[2].set_yticks([])

# ax[1].grid('on')
fig.tight_layout()
fig.savefig('../figures/XOR_overlap.png', dpi=1000,transparent=True)

#%%expressivity plot
from matplotlib import colormaps
cmap = colormaps.get('Dark2').colors
N=9
relu = lambda x: np.maximum(x,np.zeros(len(x)))
x = np.linspace(-5,N+0.5,1000)
x_regions = np.arange(0,N)
f = relu(x)+np.sum(np.array([((-1)**n)*relu(2*x-2*n) for n in range(1,N)]),0)

plt.rcParams['font.family'] = 'Arial'
fig,ax = plt.subplots(1,figsize=(3,2))
ax.plot(x,f,'k')
[ax.plot(2*n+2,0,'*',color=cmap[n]) for n in range(4)]
ax.fill_between([-5,x_regions[0]],[-0.1,-0.1],[0.1,0.1],alpha=0.3)
[ax.fill_between([x_regions[i],x_regions[i+1]],[-0.1,-0.1],[0.1,0.1],alpha=0.3) for i in range(N-1)]
ax.fill_between([x_regions[-1],N+0.5],[-0.1,-0.1],[0.1,0.1],alpha=0.3)
ax.grid('on')
fig.savefig('../figures/relu_fun.png', dpi=500,transparent=True)