import gudhi
import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from scipy.spatial import KDTree
from scipy.stats import percentileofscore
import polytope as pc


class FeedforwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation=nn.ReLU, out_layer_sz = 0, mean=-0.5, std=2, init_type = 'uniform'):
        super(FeedforwardNetwork, self).__init__()        
        self.layers = nn.ModuleList()
        self.activation = activation
        
        self.mean = mean
        self.std = std
        
        self.input_size = input_size
        self.init_type = init_type
        self.out_layer_sz = out_layer_sz
        
        self.hidden_sizes = hidden_sizes

        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(activation())

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(activation())

        if out_layer_sz>0:
            self.layers.append(nn.Linear(hidden_sizes[-1],out_layer_sz))

        # Initialize weights
        self.initialize_weights(mean, std)

    def forward(self, x):
        layerwise_activations = []
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

class UnionFind:
    def __init__(self):
        self.parent = {}
        self. rank = {}
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
                
    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
    
    def merge_lists(self, lists):
        for lst in lists:
            if lst:
                first_element = lst[0]
                self.add(first_element)
                
            for elem in lst[1:]:
                self.add(elem)
                self.union(first_element, elem)
        components = {}
        for lst in lists:
            for elem in lst:
                root = self.find(elem)
                if root not in components:
                    components[root] = set()
                components[root].update(lst)

        return [list(comp) for comp in components.values()]
                    
                
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
        self.skeleton = self.get_skeleton(simplex_tree)
        return simplex_tree
    
    def get_skeleton(self, simplicial_complex):
        simplices = simplicial_complex.get_simplices()
        skeletons = {k: [] for k in range(self.dim+1)}
        for simplex in simplices:
            skeletons[len(simplex[0])-1].append(simplex[0])
        return skeletons
    
    
    def get_subcomplex(self, simplicial_complex, subvertex_set):
        subcomplex = []
        for k in range(len(self.skeleton)):
            for simplex in self.skeleton[k]:
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
        simp_comp_pcloud = self.assign_complex(pcloud)
        return self.compute_homology(simp_comp_pcloud,subcomplex_list), simp_comp_pcloud

        
    def plot_simplicial_complex(self, pcloud, simplicial_complex, subcomplex_list=[], decomp= []):
        if len(decomp)==0:
            plt.scatter(pcloud[:,0],pcloud[:,1],s=1,color='black',zorder=500)
        else:
            for i, clss in enumerate(decomp):
                plt.scatter(pcloud[clss,0],pcloud[clss,1],s=60,c=np.ones(len(clss))*(i/len(decomp)),cmap='gist_ncar', vmin=0,vmax=1,zorder=500)

        #draw edges
        for s in self.skeleton[1]:
            plt.plot(pcloud[s,0],pcloud[s,1],'k-')
        if len(self.skeleton)>2:
            for tr in self.skeleton[2]:
                plt.fill(pcloud[tr,0],pcloud[tr,1],'gray',alpha=0.1)
        for i, vertex_set in enumerate(subcomplex_list):
            if len(vertex_set)>2:
                plt.scatter(pcloud[vertex_set,0],pcloud[vertex_set,1],c=np.ones(len(vertex_set))*(i/len(subcomplex_list)),cmap='Pastel1', vmin=0,vmax=1,s=20,zorder=1000)
                plt.plot(pcloud[vertex_set,0],pcloud[vertex_set,1],'k-') 
                plt.fill(pcloud[vertex_set,0],pcloud[vertex_set,1],alpha=0.3,zorder=1000)
            else:
                plt.scatter(pcloud[vertex_set,0],pcloud[vertex_set,1],c=np.ones(len(vertex_set))*(i/len(subcomplex_list)),cmap='Pastel1', vmin=0,vmax=1,s=20,zorder=1000)
                plt.plot(pcloud[vertex_set,0],pcloud[vertex_set,1],'k-') 
                
                
class NetworkDecompositions:
    def __init__(self, model):
        self.model = model
        self.layers = model.layers
        
        #decompositoins
        self.global_decomposition = []
        self.local_decomposition = []
        self.overlap_decomopsition = []
        self.rank_decomposition = []
        
        self.uf = UnionFind()

        
    def get_map_at_point(self,x, layer_num):
        bias = 0
        for n, layer in enumerate(self.layers):
            x = layer(x.flatten())
            if isinstance(layer, nn.Linear):
                out_sign = (x>0).float()
                Q = (torch.eye(layer.weight.shape[0])*out_sign)
                if n==0:
                    out_mat = Q@layer.weight
                    bias = Q@layer.bias
                else:
                    out_mat = Q@layer.weight@out_mat
                    bias = Q@layer.weight@bias + Q@layer.bias
                if n//2==layer_num:
                    return out_mat, bias
        return out_mat, bias
    
    def get_premap_at_point(self, x, layer_num):
        bias = 0
        for n, layer in enumerate(self.layers):
            x = layer(x.flatten())
            if isinstance(layer, nn.Linear):
                out_sign = (x>0).float()
                Q = (torch.eye(layer.weight.shape[0])*out_sign)
                if n==0:
                    out_mat = Q@layer.weight
                    bias = Q@layer.bias
                    if n==layer_num:
                        out_sign[out_sign==0] = -1
                        Q = (torch.eye(layer.weight.shape[0])*out_sign)
                        return layer.weight, layer.bias, Q
                elif n//2==layer_num:
                    out_sign[out_sign==0] = -1
                    Q = (torch.eye(layer.weight.shape[0])*out_sign)
                    return layer.weight, layer.bias, Q
                else:
                    out_mat = Q@layer.weight@out_mat
                    bias = Q@layer.weight@bias + Q@layer.bias
        return out_mat, bias

        
    def compute_codeword_at_point(self, x, top_layer=-1, mode='global'):
        codeword = []
        layer_counter = 0
        if mode=='global':
            for n, layer in enumerate(self.layers):
                x = layer(x.flatten())
                if isinstance(layer, nn.Linear):
                    if n ==len(self.layers):
                        out_sign = map(str,list(np.ones(len(x)).astype(np.int64)))
                    else:
                        out_sign = map(str,list((x>0).detach().numpy().astype(np.int64)))
                    codeword.append(''.join(out_sign))
                    layer_counter+=1
                    if layer_counter==top_layer:
                        return ''.join(codeword)
        if mode=='local':
            for n, layer in enumerate(self.layers):
                x = layer(x.flatten())
                if isinstance(layer, nn.Linear):
                    if n ==len(self.layers):
                        out_sign = map(str, list(np.ones(len(x)).astype(np.int64)))
                    else:
                        out_sign = map(str,list((x>0).detach().numpy().astype(np.int64)))
                    layer_counter+=1
                    if layer_counter==top_layer:
                        codeword=out_sign
                        return ''.join(codeword)
                    
    def compute_codeword_eq_classes(self, X, top_layer=-1, mode='global'):
        final_codewords = np.empty([len(self.model.hidden_sizes)+1,len(X)], dtype=object)
        for n in range(len(X)):
            for i in range(len(self.model.hidden_sizes)+1):
                final_codewords[i,n] = self.compute_codeword_at_point(X[n],top_layer=i+1,mode=mode)                    
        _, inverse_indices = np.unique(final_codewords, return_inverse=True)
        eq_classes = [[] for i in range(len(self.model.hidden_sizes)+1)]
        for i in range(len(self.model.hidden_sizes)+1):
            _, inverse_indices = np.unique(final_codewords[i], return_inverse=True)
            for n in np.unique(inverse_indices):
                entries = np.where(inverse_indices==n)[0]
                if len(entries)>0:
                    eq_classes[i].append(entries)
        if mode=='global':
            self.global_decomposition = [eq_classes, final_codewords]
        elif mode=='local':
            self.local_decomposition = [eq_classes, final_codewords]
        return eq_classes, final_codewords
    
    def compute_rank_at_point(self, x):
        ranks = []
        for n in range(len(self.model.hidden_sizes)+1):
            out_mat = self.get_map_at_point(x,layer_num=n)[0]
            fin_rank = torch.linalg.matrix_rank(out_mat)
            ranks.append(fin_rank)
        return ranks
    
    def compute_rank_classes(self,X):
        final_rankwords = np.zeros([len(self.model.hidden_sizes)+1,len(X)])
        for n in range(len(X)):
            final_rankwords[:,n] = self.compute_rank_at_point(X[n])[0]
        eq_classes = [[] for i in range(len(self.model.hidden_sizes)+1)]
        for i in range(len(self.model.hidden_sizes)+1):
            _, inverse_indices = np.unique(final_rankwords[i], return_inverse=True)
            for n in np.unique(inverse_indices):
                entries = np.where(inverse_indices==n)[0]
                if len(entries)>0:
                    eq_classes[i].append(entries)
        self.rank_decomposition = eq_classes
        return eq_classes, final_rankwords
    
    def compute_overlap_classes(self, X, top_layer=-1, mode='global'):
        final_codewords = np.empty([len(self.model.hidden_sizes)+1,len(X)], dtype=object)#.astype(np.int64)
        for n in range(len(X)):
            for i in range(len(self.hidden_sizes)+1):
                final_codewords[i,n] = self.compute_codeword_at_point(X[n],top_layer=i+1,mode=mode)                    
        _, inverse_indices = np.unique(final_codewords, return_inverse=True)
        eq_classes = [[] for i in range(len(self.hidden_sizes)+1)]
        for i in range(len(self.hidden_sizes)+1):
            _, inverse_indices = np.unique(final_codewords[i], return_inverse=True)
            for n in np.unique(inverse_indices):
                entries = np.where(inverse_indices==n)[0]
                if len(entries)>0:
                    eq_classes[i].append(entries)
        if mode=='global':
            self.global_decomposition = [eq_classes, final_codewords]
        elif mode=='local':
            self.local_decomposition = [eq_classes, final_codewords]
        return eq_classes, final_codewords
    
    def weights_to_polyhedra(self, X, layer=-1,bbox=[-1,1]):
        if len(self.global_decomposition)==0:
            global_classes = self.compute_codeword_eq_classes(X)[0]
        else:
            global_classes = self.global_decomposition[0]
        polyhedra = []
        if layer==0:
            bbox_matrix = np.vstack([np.eye(self.model.input_size),-np.eye(self.model.input_size)])
            bbox_bias_lower = -np.array([[bbox[0]]*self.model.input_size])
            bbox_bias_upper = np.array([[bbox[1]]*self.model.input_size])
            bbox_bias = np.hstack([bbox_bias_lower,bbox_bias_upper])[0]
            for curr_cls in global_classes[layer]:
                T_ck, B_ck, J_ck = self.get_premap_at_point(X[curr_cls[0]], layer)
                T_ck = np.vstack([(J_ck@T_ck).detach().numpy(),bbox_matrix])
                B_ck = np.concatenate([(J_ck@B_ck).detach().numpy(),bbox_bias])
                polyhedra.append(pc.reduce(pc.Polytope(T_ck, B_ck)))
        else:                
            bbox_matrix = np.vstack([np.eye(self.model.hidden_sizes[layer]),-np.eye(self.model.hidden_sizes[layer])])
            bbox_bias_lower = -np.array([[bbox[0]]*self.model.hidden_sizes[layer]])
            bbox_bias_upper = np.array([[bbox[1]]*self.model.hidden_sizes[layer]])
            bbox_bias = np.hstack([bbox_bias_lower,bbox_bias_upper])[0]            
            for prev_cls in global_classes[layer-1]:
                for curr_cls in global_classes[layer]:
                    if set(curr_cls) <= set(prev_cls):
                        #define the weight operator at the current and previous layers
                        T_ck, B_ck, J_ck = self.get_premap_at_point(X[curr_cls[0]], layer)
                        T_prev, B_prev = self.get_map_at_point(X[prev_cls[0]], layer-1)
                        T_dagger = torch.linalg.pinv(T_prev)
                        eq_system_W = np.vstack([(T_prev@T_dagger).detach().numpy(),
                                                  (J_ck@T_ck@T_prev@T_dagger).detach().numpy(),
                                                  bbox_matrix])
                        eq_system_b = np.concatenate([(-T_prev@T_dagger@B_prev).detach().numpy(),
                                                    (-J_ck@T_ck@T_prev@T_dagger@B_prev).detach().numpy()+(J_ck@B_ck).detach().numpy(),
                                                     bbox_bias])
                        polyhedra.append(pc.reduce(pc.Polytope(eq_system_W, eq_system_b)))
        return polyhedra
        
    
    def find_overlapping_eq_classes(self, X, layer=-1, sensitivity=1e-6):
        if len(self.local_decomposition)==0:
            local_classes = self.compute_codeword_eq_classes(X,mode='local')[0][layer]
        else:
            local_classes = self.local_decomposition[0][layer]
        if len(self.global_decomposition)==0:
            global_classes = self.compute_codeword_eq_classes(X)[0][layer]
        else:
            global_classes = self.global_decomposition[0][layer]
        all_classes = []
        contractible_classes = []
        for n, class_n in enumerate(global_classes):#class_partitions[i]):
            all_classes.append(list(class_n))
            glob_in_loc_A = np.where([set(class_n) <= set(loc_class) for loc_class in local_classes])[0]
            for m, class_m in enumerate(global_classes):#class_partitions[i]):
                glob_in_loc_B = np.where([set(class_m) <= set(loc_class) for loc_class in local_classes])[0]             
                if n>=m and glob_in_loc_A==glob_in_loc_B:
                    intersect = self.find_intersection(X[class_n],X[class_m],layer,epsilon=sensitivity)
                    overlap_class = [[class_n[intersect[i,0]], class_m[intersect[i,1]]] for i in range(len(intersect))]
                    all_classes.extend(overlap_class)
                    contractible_classes.extend(overlap_class)
        return self.uf.merge_lists(contractible_classes), self.uf.merge_lists(all_classes)
    
    def find_intersection(self,C1,C2,layer=-1,epsilon=1e-6):
        A1,b1 = self.get_map_at_point(C1[0], layer)
        A2,b2 = self.get_map_at_point(C2[0], layer)
        if torch.all(A1==A2) and torch.all(b1==b2) and torch.linalg.matrix_rank(A1).item()==0:
            epsilon= 1e16
        # epsilon = (epsilon*max(max(torch.linalg.svd(A1.T)[1]),max(torch.linalg.svd(A2.T)[1]))).item()
        F1 = self.model(C1)[layer].detach()
        F2 = self.model(C2)[layer].detach()
        
        contraction_factor1 = np.nanmean(cdist(F1,F1)/cdist(C1,C1))
        contraction_factor2 = np.nanmean(cdist(F2,F2)/cdist(C2,C2))
        coef = np.nanmin(np.stack([contraction_factor1, contraction_factor2]))
        if ~np.isnan(coef):
            epsilon = epsilon*coef
        # d_2 = cdist(F2,F2)
        # d_in = cdist(C1,C2)
        d_out  = cdist(F1,F2)
        # d = d_out/d_in
        # contract_mask = np.argwhere((d_out)<=epsilon)
        # if torch.all(A1==A2) and torch.all(b1==b2):
        #     return np.argwhere(d_out==0)
        
        # kdtree_C1 = KDTree(F1)
        # kdtree_C2 = KDTree(F2)
        # d_1, _ = kdtree_C1.query(F1,k=2)
        # d_2, _ = kdtree_C2.query(F2,k=2)
        contract_mask = np.argwhere(d_out<=epsilon)#np.percentile(d_1[:,1],99))#(percentileofscore(d_1[:,1], d_out)/100)<1)#np.argwhere(d_out.T<=d_1[:,1])#
        contract_mask = contract_mask[contract_mask[:,0]!=contract_mask[:,1]]
        # contract_mask = np.vstack([contract_mask[:,1],contract_mask[:,0]]).T
        # contract_mask = np.vstack([contract_mask,np.argwhere(d_out<=d_2[:,1])])#

        return contract_mask

    def compute_overlap_decomp(self,X,sensitivity = 1e-6):
        overlap_decomps = []
        for n in range(len(self.model.hidden_sizes)+1):
            overlap_decomps.append(self.find_overlapping_eq_classes(X,layer=n, sensitivity=sensitivity)[0])
        self.overlap_decomopsition = overlap_decomps
        return overlap_decomps

    def compute_all_decomps(self, X, sensitivity = 1e-6):
        self.local_decomposition = self.compute_codeword_eq_classes(X,mode='local')
        self.global_decomposition = self.compute_codeword_eq_classes(X,mode='global')
        self.rank_decomposition = self.compute_rank_classes(X)
        self.overlap_decomposition = self.compute_overlap_decomp(X, sensitivity=sensitivity)

        
class MapHomology:
    def __init__(self, step_map, input_space, scomplex_analyzer=SimpComp(),gluing_threshold = 1e-6, neural_model = None):
        try:
            if any(isinstance(module, nn.ModuleList) for module in neural_model.children()):
                self.step_map = self.neural_net_to_map_step
            else:
                raise('Module list is missing in neural network model.')
        except:
            self.step_map = step_map
        self.input_space = input_space
        self.SComplexAnalyzer = scomplex_analyzer
        self.gluing_threshold = gluing_threshold
        self.neural_model = neural_model
        self.current_step = 0
        self.decomps = None
        self.uf = UnionFind()
        
    def neural_net_to_map_step(self, x):
        #Requires a torch model with all the layers stored in a ModuleList
        if self.current_step==0:
            self.decomps = NetworkDecompositions(self.neural_model)
            self.overlaps = self.decomps.compute_overlap_decomp(x,sensitivity=self.gluing_threshold)
        layers = self.neural_model.layers   
        if self.current_step >= len(layers):
                raise IndexError("Current step exceeds number of layers.")
        
        x = layers[self.current_step](torch.Tensor(x))
        try:
            if isinstance(layers[self.current_step+1], self.neural_model.activation):
                x = layers[self.current_step+1](x)
                self.current_step += 2
            else:
                self.current_step += 1
            return x
        except:
            return x

    def decide_gluing(self, x):
        if self.neural_model==None or self.current_step==0:
            d = cdist(x, x)
            np.fill_diagonal(d,np.nan)
            contract_mask = np.argwhere(d<self.gluing_threshold)
            contract_mask = [[contract_mask[i,0], contract_mask[i,1]] for i in range(len(contract_mask))]
            contract_mask = self.uf.merge_lists(contract_mask)
        else:
            contract_mask = self.overlaps[self.current_step//2]
        return contract_mask
    
    def map_homology_sequence(self, n_iterations=1):
        map_homology = []
        ovrlap_complexes = []
        f_x_iterations = []
        x = torch.Tensor(self.input_space.copy())
        for n in range(n_iterations+1):
            gluing_regions = self.decide_gluing(x)
            relative_hom, simp_comp = self.SComplexAnalyzer.homology_pointcloud(self.input_space,gluing_regions)
            map_homology.append(relative_hom[:self.SComplexAnalyzer.dim])
            ovrlap_complexes.append(gluing_regions)
            f_x_iterations.append(x)
            x = self.step_map(x).detach()
        return map_homology, f_x_iterations, ovrlap_complexes
        
#%%
res = 32
x = np.linspace(-3, 3, res)
y = np.linspace(-3, 3, res)
neighbor_scale = (max(x)-min(x))/(res-1)
neighbor_scale = np.sqrt(2*(neighbor_scale**2))+1e-1
xx, yy = np.meshgrid(x, y )
pcloud = np.column_stack((xx.ravel(), yy.ravel()))
# pcloud = 2*(np.random.rand(res,3)-0.5)
# pcloud = np.vstack([np.cos(np.linspace(-np.pi,np.pi,res,endpoint=False)),np.sin(np.linspace(-np.pi,np.pi,res,endpoint=False))]).T #+ 1
# pcloud = pcloud + 0.005*np.random.randn(*pcloud.shape)
# d = cdist(pcloud,pcloud)
# neighbor_scale = 2*np.min(d[d>1e-12])+0.05
# pcloud = np.linspace(0,1,res)[:,None]
# pcloud=np.random.randn(25,2).T

sc = SimpComp(threshold=neighbor_scale,dim=4,method='Rips')
tree = sc.assign_complex(pcloud)
sc.plot_simplicial_complex(pcloud,tree,[[0,2,32,71],[1,9]])
#%%
hidden_sizes = [2]*5
model = FeedforwardNetwork(2,hidden_sizes,out_layer_sz=2, init_type='none', activation=nn.ReLU)
decomps = NetworkDecompositions(model)
decomps_1 = decomps.weights_to_polyhedra(torch.Tensor(pcloud),layer=0,bbox=[-5,5])
decomps_2 = decomps.weights_to_polyhedra(torch.Tensor(pcloud),layer=1,bbox=[-5,5])

#%%
# hom = sc.compute_homology(tree,subcomplex_list=[])
# print(hom)
import random
seed = 420

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

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
    x_ = 1-1.4*x[:,0]**2+x[:,1]
    y_ = 0.3*x[:,1]
    return np.vstack([x_,y_]).T

def bakers_map(x):
    x_1 = x[x[:,0]<0.5]
    x_2 = x[x[:,0]>=0.5]
    x_ = np.concatenate([2*x_1[:,0],2-2*x_2[:,0]])
    y_ = np.concatenate([x_1[:,1]/2, 1 - x_1[:,1]/2])
    return np.vstack([x_,y_]).T

def recurse_point(x):
    return 1.8*(x-x**5)


hidden_sizes = [4]*5
model = FeedforwardNetwork(2,hidden_sizes,out_layer_sz=2, init_type='unifrom', activation=nn.ReLU)
# model = lambda x: torch.nn.RNN(2,2,1,nonlinearity='relu')(torch.Tensor(x))[0]

P_ = torch.randn(hidden_sizes[0],hidden_sizes[0])
E, V = torch.linalg.eig(P_)
E = E/(abs(E).max())
P = (V@torch.diag(E)@torch.linalg.pinv(V)).real

for n, layer in enumerate(model.layers): #make all weight matrices the same for RNN condition
    if isinstance(layer, nn.Linear) and n>2:
        layer.weight = model.layers[2].weight#nn.Parameter(P)
        layer.bias = model.layers[2].bias  
#%%
thresh = 0#neighbor_scale/50
MPH =  MapHomology(model, pcloud, scomplex_analyzer=sc, gluing_threshold=thresh, neural_model=model)
map_homology, fs, ovrlap_complex = MPH.map_homology_sequence(len(hidden_sizes))
print(map_homology)
get_all_decomps = MPH.decomps.compute_all_decomps(torch.Tensor(pcloud),sensitivity=thresh)
# map_diagram = map_homology
import matplotlib.pyplot as plt

# [plt.plot(fs[i],'.') for i in range(len(fs))]
reduced_mfld = []
reducer = Isomap(n_components=2)
for i in range(len(fs)):
    pcl = reducer.fit_transform(fs[i])
    plt.scatter(pcl[:,0], pcl[:,1],s=1)   
    reduced_mfld.append(pcl)     

for i in range(len(ovrlap_complex)):
    plt.figure()
    # sc.plot_simplicial_complex(pcloud, tree, ovrlap_complex[i])
    sc.plot_simplicial_complex(pcloud, tree, MPH.decomps.overlap_decomopsition[i],MPH.decomps.local_decomposition[0][i])
    # sc.plot_simplicial_complex(reduced_mfld[i], tree, MPH.decomps.overlap_decomopsition[i],MPH.decomps.local_decomposition[0][i-1])
# 

    #%%
import umap.umap_ as umap
reducer = umap.UMAP(n_components=2, n_neighbors=4,min_dist=0, metric='precomputed')
# reducer = Isomap(n_components=2, n_neighbors=2, metric='precomputed')
# mfld = reducer.fit_transform(cdist(fs[-1],fs[-1])).T
mfld = reducer.fit_transform(cdist(pcloud,pcloud)).T
plt.scatter(mfld[0],mfld[1],c=np.linspace(0,1,res),s=1)








