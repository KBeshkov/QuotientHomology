import sys
sys.path.append("/Users/kosio/Repos/network-relative-homology/src/")
from NetRelHom import *
from TopologicalMethods import *

n_layers = 3
hidden_sizes = [3]*n_layers
model = FeedforwardNetwork(input_size=2,hidden_sizes=hidden_sizes,#8*f_dim,4*f_dim,2*f_dim], 
                           out_layer_sz=1, init_type='none',activation=nn.ReLU)
X_data = torch.randn(1000,2)
decomposer = NetworkDecompositions(model)

box_region = pc.Region([pc.box2poly([[-10,10],[-10,10]])])
decomps_data  = decomposer.weights_to_inpolyhedra(X_data,layer=n_layers, bbox=[-10,10])

region_data = pc.Region(decomps_data[n_layers])
# region_data = region_data + (box_region-region_data)
region_data.plot()
plt.xlim(-10,10)
plt.ylim(-10,10)

decomps = decomposer.weights_to_all_polyhedra(X_data, layer=n_layers, bbox=[-10,10])
region = pc.Region(decomps[n_layers])


region.plot()
plt.xlim(-10,10)
plt.ylim(-10,10)
