import gudhi
from sympy import Matrix, FiniteField
from sympy.matrices.normalforms import smith_normal_form
import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn


class FeedforwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation=nn.ReLU, out_layer_sz = 0, mean=-0.5, std=2, init_type = 'uniform',conv_dict={}):
        super(FeedforwardNetwork, self).__init__()


        #decompositoins
        self.global_decomposition = []
        self.local_decomposition = []
        
        self.layers = nn.ModuleList()
        self.activation = activation
        
        self.mean = mean
        self.std = std
        
        self.input_size = input_size
        self.init_type = init_type
        self.out_layer_sz = out_layer_sz
        
        self.hidden_sizes = hidden_sizes

        self.conv_dict = conv_dict
        # Input layer
        if len(conv_dict)>0:
            # self.layers_ff = nn.ModuleList()
            
            self.in_channels = conv_dict['in_channels']
            self.out_channels = conv_dict['out_channels']
            self.kernel_sizes = conv_dict['kernel_sizes']
            self.image_size = conv_dict['image_size'] #only valid for square images so far
            
            self.feature_layer = []
            
            self.feature_layer.append(nn.Conv2d(self.in_channels,self.out_channels[0],self.kernel_sizes[0],bias=False))
            self.feature_layer.append(activation())
            self.feature_layer.append(nn.MaxPool2d(2,1))
            
            # self.layers_ff.append(nn.Linear(self.in_channels*self.image_size**2, self.out_channels[0]*(self.image_size-self.kernel_sizes[0]+1)**2,bias=False))
            # self.layers_ff.weight = self.convolution_to_dense(self.layers[0], self.image_size)
            # self.layers_ff.append(activation())
            
            self.grid_size = self.image_size-self.kernel_sizes[0]
            for i in range(1,len(self.out_channels)):
                self.feature_layer.append(nn.Conv2d(self.out_channels[i-1], self.out_channels[i], self.kernel_sizes[i],bias=False))
                self.feature_layer.append(activation())
                self.feature_layer.append(nn.MaxPool2d(2,1))

                # self.layers_ff.append(nn.Linear(self.out_channels[i-1]*self.grid_size**2, self.out_channels[i]*(self.grid_size-self.kernel_sizes[i]+1)**2,bias=False))
                # self.layers_ff.weight = self.convolution_to_dense(self.layers[-2], self.grid_size)
                
                self.grid_size = self.grid_size - self.kernel_sizes[i]
            self.flat_size = self.out_channels[-1]*self.grid_size**2
            self.feature_layer.append(nn.Flatten())
            self.layers.append(nn.Linear(self.flat_size, hidden_sizes[0]))
            self.layers.append(activation())
            
            self.feature_layer = nn.Sequential(*self.feature_layer)
            # self.layers_ff.append(self.layers[-2])
        else:
            self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
            self.layers.append(activation())
            
            # self.layers_ff.append(self.layers[0])
            # self.layers_ff.append(activation())

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(activation())
            
            # self.layers_ff.append(self.layers[-2])
            # self.layers_ff.append(activation())
        if out_layer_sz>0:
            self.layers.append(nn.Linear(hidden_sizes[-1],out_layer_sz))
            
            # self.layers_ff.append(self.layers[-1])

        # Initialize weights
        self.initialize_weights(mean, std)

    def forward(self, x):
        layerwise_activations = []
        if len(self.conv_dict)>0:
            x = self.feature_layer(x)
        for layer in self.layers:
            if isinstance(layer, self.activation):
                x = layer(x)
                layerwise_activations.append(torch.flatten(x.clone(),1))
            else:
                x = layer(x)
        if self.out_layer_sz>0:
            layerwise_activations.append(x)
        return layerwise_activations

    def initialize_weights(self, mean, std):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # nn.init.zeros_(layer.bias)#,-std,std)
                if self.init_type == 'normal':
                    nn.init.normal_(layer.weight, mean=mean, std=std)
                elif self.init_type == 'uniform':
                    std*(nn.init.uniform_(layer.weight,0,1)+mean)
                elif self.init_type == 'zeros':
                    nn.init.zeros_(layer.weight)
                elif self.init_type == 'orthogonal':
                    nn.init.orthogonal_(layer.weight)
                else:
                    break
                
class SimpComp:
    def __init__(self, method='Rips', threshold = 4, dim=2):
        self.method = method
        self.threshold = threshold
        self.dim = dim
        
    def assign_complex(self, point_cloud):
        if self.method=='Rips':
            rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=self.threshold)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.dim)
        elif self.method=='Alpha':
            alpha_complex = gudhi.AlphaComplex(points=point_cloud)
            simplex_tree = alpha_complex.create_simplex_tree(default_filtration_value=True)
        elif self.method=='Tangential':
            tangential_complex = gudhi.TangentialComplex(self.threshold,points=point_cloud)
            simplex_tree = tangential_complex.create_simplex_tree()
        return simplex_tree
    
    def get_skeleton(self, simplicial_complex):
        simplices = simplicial_complex.get_simplices()
        skeletons = {k: [] for k in range(self.dim+1)}
        for simplex in simplices:
            skeletons[len(simplex[0])-1].append(simplex[0])
        return skeletons
    
    def get_boundary_matrices(self, simplicial_complex):
        skeletons = self.get_skeleton(simplicial_complex)
        
        boundary_matrices = {k: [] for k in range(self.dim)}
        boundary_matrices[0] = np.ones([1,len(skeletons[0])])
    
        for k in range(self.dim):
            boundary_matrices[k+1] = np.zeros([len(skeletons[k]),len(skeletons[k+1])])
            for n,simp in enumerate(skeletons[k+1]):
                faces = [set(b) <= set(simp) for b in skeletons[k]]
                boundary_matrices[k+1][faces,n] = 1
        #boundary_matrices[self.dim+1] = np.ones([1,len(skeletons[0])])
        return boundary_matrices
    
    def get_subcomplex(self, simplicial_complex, subvertex_set):
        skeletons = self.get_skeleton(simplicial_complex)
        subcomplex = []
        for k in range(len(skeletons)):
            for simplex in skeletons[k]:
                if set(simplex) <= set(subvertex_set):
                    subcomplex.append(simplex)
        return subcomplex
    
    def compute_homology(self, simplicial_complex, subcomplex_list=[]):    
        for i,vertex_set in enumerate(subcomplex_list):
            subcomplex = self.get_subcomplex(simplicial_complex, vertex_set)
            cone_point = len(list(simplicial_complex.get_skeleton(0))) + i
            simplicial_complex.insert([cone_point])
            for simplex in subcomplex:
                simplicial_complex.insert(simplex + [cone_point])
        simplicial_complex.compute_persistence(persistence_dim_max = False)
        return simplicial_complex.betti_numbers()
    
    def homology_pointcloud(self,pcloud, subcomplex_list=[]):
        return self.compute_homology(self.assign_complex(pcloud),subcomplex_list)
        # boundary_matrices = self.get_boundary_matrices(simplicial_complex)
        # Z_ranks = []
        # B_ranks = []
        # for D in boundary_matrices.values():
        #     C = Matrix(D.astype(int))
        #     C_reduced = smith_normal_form(C,domain=FiniteField((2)))
        #     B_ranks.append(np.sum(C_reduced))
        #     Z_ranks.append(C_reduced.shape[1]-np.sum(C_reduced))
        # Betti = (np.array(Z_ranks)[:-1]-np.array(B_ranks)[1:]).astype(int)
        # return Betti, Z_ranks, B_ranks
        
class MapHomology:
    def __init__(self, step_map, input_space, scomplex_analyzer=SimpComp(),gluing_threshold = 1e-6, neural_model = None):
        if any(isinstance(module, nn.ModuleList) for module in neural_model.children()):
            self.step_map = self.neural_net_to_map_step
        else:
            self.step_map = step_map
        self.input_space = input_space
        self.SComplexAnalyzer = scomplex_analyzer
        self.gluing_threshold = gluing_threshold
        self.neural_model = neural_model
        self.current_step = 0
        
    def neural_net_to_map_step(self, x):
        #Requires a torch model with all the layers stored in a ModuleList
        layers = self.neural_model.layers   
        if self.current_step >= len(layers):
                raise IndexError("Current step exceeds number of layers.")
        
        x = layers[self.current_step](torch.Tensor(x))
        try:
            if isinstance(layers[self.current_step+1], self.neural_model.activation):
                x = layers[self.current_step+1](x)
                self.current_step += 2
                return x
        except:
            return x
    
    def decide_gluing(self, x):
        d = cdist(x, x)
        np.fill_diagonal(d,np.nan)
        contract_mask = np.argwhere(d<self.gluing_threshold)
        return contract_mask
    
    def map_homology_sequence(self, n_iterations=1):
        map_homology = [self.SComplexAnalyzer.homology_pointcloud(self.input_space)]
        f_x_iterations = []
        x = self.input_space.copy()
        for n in range(n_iterations):
            x = self.step_map(x).detach()
            gluing_regions = self.decide_gluing(x)
            relative_hom = self.SComplexAnalyzer.homology_pointcloud(self.input_space,gluing_regions)
            map_homology.append(relative_hom)
            f_x_iterations.append(x)
        return map_homology, f_x_iterations
        

res = 25
x = np.linspace(-1, 1, res)
y = np.linspace(-1, 1, res)
neighbor_scale = (max(x)-min(x))/(res-2)
neighbor_scale = np.sqrt(2*(neighbor_scale**2))
xx, yy = np.meshgrid(x, y)
pcloud = np.column_stack((xx.ravel(), yy.ravel()))
# pcloud = np.vstack([np.cos(np.linspace(-np.pi,np.pi,res)),np.sin(np.linspace(-np.pi,np.pi,res))]).T
# pcloud=np.random.randn(25,2).T

sc = SimpComp(threshold=neighbor_scale,dim=3,method='Rips')
tree = sc.assign_complex(pcloud)

mats = sc.get_boundary_matrices(tree)
# hom = sc.compute_homology(tree,subcomplex_list=[])
# print(hom)

def gingerbread_map(x):
    x_ = 1-x[:,1]+np.abs(x[:,0])
    y_ = x[:,0]
    return np.vstack([x_,y_]).T

def duffing_map(x): 
    x_ = x[:,1]
    y_ = -0.2*x[:,0]+2.75*x[:,1]-x[:,1]**3
    return np.vstack([x_,y_]).T

def bogdanov_map(x, eps=0, k=1.2, mu=2):
    y_ = x[:,1]+eps*x[:,1]+k*x[:,0]*(x[:,0]-1)+mu*x[:,0]*x[:,1]
    x_ = x[:,0]+y_
    return np.vstack([x_,y_]).T

def henon_map(x):
    x_ = 1- 1.4*x[:,0]**2+x[:,1]
    y_ = 0.3*x[:,1]
    return np.vstack([x_,y_]).T

def backers_map(x):
    x_1 = x[x[:,0]<0.5]
    x_2 = x[x[:,0]>=0.5]
    x_ = np.concatenate([2*x_1[:,0],2-2*x_2[:,0]])
    y_ = np.concatenate([x_1[:,1]/2, 1 - x_1[:,1]/2])
    return np.vstack([x_,y_]).T

model = FeedforwardNetwork(2,[10,3,3,3,3],out_layer_sz=2, init_type='None')

map_homology, fs = MapHomology(model, pcloud, scomplex_analyzer=sc, gluing_threshold=1e-16, neural_model=model).map_homology_sequence(6)
print(map_homology)

map_diagram = np.vstack(map_homology)

import matplotlib.pyplot as plt

plt.scatter(fs[-1][:,0], fs[-1][:,1],s=1)

