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
from scipy.optimize import linprog
import polytope as pc
import cdd
from itertools import product
from tqdm.auto import tqdm


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

    def forward(self, x, pre=False):
        layerwise_activations = []
        for layer in self.layers:
            if pre:
                if isinstance(layer, nn.Linear):
                    x = layer(x)
                    layerwise_activations.append(torch.flatten(x.clone(),1))
                else:
                    x = layer(x)

            else:
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
                

def union_find(connections):
    def find(x, parent):
        # Path compression
        if parent[x] != x:
            parent[x] = find(parent[x], parent)
        return parent[x]

    def union(x, y, parent, rank):
        # Union by rank
        root_x = find(x, parent)
        root_y = find(y, parent)

        if root_x != root_y:
            if rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            elif rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            else:
                parent[root_y] = root_x
                rank[root_x] += 1

    # Get all unique elements
    elements = set(x for sublist in connections for x in sublist)

    # Initialize parent and rank dictionaries
    parent = {x: x for x in elements}
    rank = {x: 0 for x in elements}

    # Apply union operations for all connections
    for a, b in connections:
        union(a, b, parent, rank)

    # Group connected components
    groups = {}
    for element in elements:
        root = find(element, parent)
        if root not in groups:
            groups[root] = []
        groups[root].append(element)

    return list(groups.values())


class NetworkDecompositions:
    def __init__(self, model):
        self.model = model
        self.layers = model.layers
        
        #decompositoins
        self.global_decomposition = []
        self.local_decomposition = []
        self.overlap_decomposition = []
        self.rank_decomposition = []
        self.polyhedral_decomposition = []
        self.union_find = union_find
        

        
    def get_map_at_point(self,x, layer_num):
        bias = 0
        layer_n = 0
        for n, layer in enumerate(self.layers):
            x = layer(x.flatten())
            if isinstance(layer, nn.Linear):
                out_sign = (x>0).float()
                Q = (torch.eye(layer.weight.shape[0])*out_sign)
                if n==0:
                    out_mat = Q@layer.weight
                    bias = Q@layer.bias
                elif n==len(self.layers)-1:
                    out_mat = layer.weight@out_mat
                    bias = layer.weight@bias + layer.bias
                else:
                    out_mat = Q@layer.weight@out_mat
                    bias = Q@layer.weight@bias + Q@layer.bias
                if layer_n==layer_num:
                    return out_mat, bias
                layer_n += 1
        return out_mat, bias
    
    def get_premap_at_point(self, x, layer_num):
        bias = 0
        layer_n = 0
        for n, layer in enumerate(self.layers):
            x = layer(x.flatten())
            if isinstance(layer, nn.Linear):
                out_sign = (x>0).float()
                Q = (torch.eye(layer.weight.shape[0])*out_sign)
                if n==0:
                    if n==layer_num:
                        out_sign[out_sign==0] = -1
                        Q = (torch.eye(layer.weight.shape[0])*out_sign)
                        return layer.weight, layer.bias, Q
                    out_mat = Q@layer.weight
                    bias = Q@layer.bias

                elif layer_n==layer_num:
                    out_sign[out_sign==0] = -1
                    Q = (torch.eye(layer.weight.shape[0])*out_sign)
                    return layer.weight@out_mat, layer.weight@bias + layer.bias, Q
                else:
                    out_mat = Q@layer.weight@out_mat
                    bias = Q@layer.weight@bias + Q@layer.bias
                layer_n += 1
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
    
    def init_bbox(self,bbox=[-10,10]):
        bbox_matrix = np.vstack([-np.eye(self.model.input_size),np.eye(self.model.input_size)])
        bbox_bias_lower = -np.array([[bbox[0]]*self.model.input_size])
        bbox_bias_upper = np.array([[bbox[1]]*self.model.input_size])
        bbox_bias = np.hstack([bbox_bias_lower,bbox_bias_upper])[0]
        return bbox_matrix, bbox_bias
    
    def weights_to_inpolyhedra(self, X, layer=-1,bbox=[-100,100],recomp_global_decomp = False):
        if len(self.global_decomposition)==0 or recomp_global_decomp:
            global_classes = self.compute_codeword_eq_classes(X)[0]
        else:
            global_classes = self.global_decomposition[0]
        polyhedra = [[] for i in range(layer+1)]
        maps = [[] for i in range(layer+1)]
        bbox_matrix, bbox_bias = self.init_bbox(bbox)
        
        for l in range(layer+1):
            if l==0:
                for curr_cls in global_classes[0]:
                    T_ck, B_ck, J_ck = self.get_premap_at_point(X[curr_cls[0]], l)
                    eq_system_W = np.vstack([-(J_ck@T_ck).detach().numpy(),bbox_matrix])
                    eq_system_b = np.concatenate([(J_ck@B_ck).detach().numpy(),bbox_bias])
                    phedra = pc.reduce(pc.Polytope(eq_system_W, eq_system_b))
                    polyhedra[0].append(phedra)
            else:
                for curr_cls in global_classes[l]:
                    T_new, B_new, J_new = self.get_premap_at_point(X[curr_cls[0]], l)
                    parent_region = np.where([set(curr_cls) <= set(prev_class) for prev_class in global_classes[l-1]])[0].item()
                    eq_system_W = np.vstack([-(J_new@T_new).detach().numpy(),bbox_matrix])
                    eq_system_b = np.concatenate([(J_new@B_new).detach().numpy(),bbox_bias])
                    phedra2 = pc.Polytope(eq_system_W, eq_system_b)
                    phedra = pc.reduce(polyhedra[l-1][parent_region].intersect(phedra2))
                    polyhedra[l].append(phedra)
        return polyhedra
    
    
    def h_rep_at_point(self,x,layer,bbox=[-100,100]):
        bbox_matrix, bbox_bias = self.init_bbox(bbox)

        for l in range(layer+1):
            if l==0:
                T_ck, B_ck, J_ck = self.get_premap_at_point(x, l)
                eq_system_W = np.vstack([-(J_ck@T_ck).detach().numpy(),bbox_matrix])
                eq_system_b = np.concatenate([(J_ck@B_ck).detach().numpy(),bbox_bias])
                phedra = pc.reduce(pc.Polytope(eq_system_W, eq_system_b))
            else:
                T_new, B_new, J_new = self.get_premap_at_point(x, l)
                eq_system_W = np.vstack([-(J_new@T_new).detach().numpy(),bbox_matrix])
                eq_system_b = np.concatenate([(J_new@B_new).detach().numpy(),bbox_bias])
                phedra2 = pc.Polytope(eq_system_W, eq_system_b)
                phedra = pc.reduce(phedra.intersect(phedra2))
        return phedra.A, phedra.b
    
    def populate_polyhedra(self,H_rep, n_samples = 100):
        min_coords = []
        max_coords = []
        for i in range(H_rep[0].shape[1]):
            c = np.zeros(H_rep[0].shape[1])
            c[i] = -1  # Minimize this coordinate
            res_min = linprog(c, A_ub=H_rep[0], b_ub=H_rep[1], method='highs',bounds=(None,None))
            c[i] = 1   # Maximize this coordinate
            res_max = linprog(c, A_ub=H_rep[0], b_ub=H_rep[1], method='highs',bounds=(None,None))
            
            min_coords.append(res_min.x[i])
            max_coords.append(res_max.x[i])

        samples = []

        while len(samples) < n_samples:
            # Generate random point in bounding box
            point = np.array([
                np.random.uniform(min_coords[j], max_coords[j]) 
                for j in range(len(min_coords))
            ])
            if np.all(H_rep[0] @ point <= H_rep[1]):
                samples.append(point)
        return np.array(samples).squeeze()

    
    def weights_to_all_polyhedra(self, X, layer, bbox=[-100,100]):
        #Only use this for very small networks as it uses up to 2^{\sum{n_i}} permutations, with n_i being the width of layer i. 
        bbox_matrix, bbox_bias = self.init_bbox(bbox)
        box_region = pc.Region([pc.Polytope(bbox_matrix,bbox_bias)])
        polyhedra = self.weights_to_inpolyhedra(X,layer, bbox)
        for l in range(layer+1):
            test_region = pc.Region(polyhedra[l])
            complement = box_region-test_region
            if isinstance(complement,pc.Polytope):
                complement = pc.Region([complement])
            while complement.volume!=0:
                for pol in complement:                    
                    X_new = torch.Tensor(self.populate_polyhedra((pol.A,pol.b), n_samples=10))
                    polyhedra[l].extend(self.weights_to_inpolyhedra(X_new,l, bbox, recomp_global_decomp=True)[l])
                test_region = pc.Region(polyhedra[l])
                complement = box_region-test_region 
                if isinstance(complement,pc.Polytope):
                    complement = pc.Region([complement])
        return polyhedra
    
    def all_polyhedra_approx(self, polyh_data, bbox=[-10,10]):
        bbox_matrix, bbox_bias = self.init_bbox(bbox)
        box_reg = pc.Region([pc.Polytope(bbox_matrix,bbox_bias)])
        complement = pc.separate(box_reg - polyh_data)
        full_decomp = [*complement, *polyh_data]
        return full_decomp
    
    def dmat_gdecomp(self, X, layer):
        global_classes = self.global_decomposition[0][layer]
        outs = self.model(X)[layer].detach().numpy()
        d = np.zeros([len(global_classes),len(global_classes)])
        for n in range(len(global_classes)):
            for m in range(len(global_classes)):
                if n>m:
                    d[n,m] = np.min(cdist(outs[global_classes[n]], 
                                          outs[global_classes[m]]))
        return d+d.T
        
    def find_overlapping_eq_classes(self, X, layer=-1, sensitivity=1e-6):
        # if len(self.local_decomposition)==0:
        #     local_classes = self.compute_codeword_eq_classes(X,mode='local')[0][layer]
        # else:
        #     local_classes = self.local_decomposition[0][layer]
        if len(self.global_decomposition)==0:
            global_classes = self.compute_codeword_eq_classes(X)[0][layer]
        else:
            global_classes = self.global_decomposition[0][layer]
            
        if len(self.polyhedral_decomposition)==0:
            self.polyhedral_decomposition = self.weights_to_inpolyhedra(X,layer = len(self.model.hidden_sizes))
        all_classes = []
        contractible_classes = []
        d_indexing = self.dmat_gdecomp(X, layer)
        for n, class_n in enumerate(global_classes):#class_partitions[i]):
            all_classes.append(list(class_n))
            # glob_in_loc_A = np.where([set(class_n) <= set(loc_class) for loc_class in local_classes])[0]
            for m, class_m in enumerate(global_classes):#class_partitions[i]):
                # glob_in_loc_B = np.where([set(class_m) <= set(loc_class) for loc_class in local_classes])[0]             
                if n>m and d_indexing[n,m]<sensitivity:#glob_in_loc_A==glob_in_loc_B:
                    intersect = self.find_intersection(X[class_n],X[class_m],n,m,layer)
                    overlap_class = [[class_n[intersect[i,1]], class_m[intersect[i,0]]] for i in range(len(intersect))]
                    all_classes.extend(overlap_class)
                    contractible_classes.extend(overlap_class)
        print('Done with layer '+str(layer))
        return self.union_find(contractible_classes), all_classes
    
    def find_intersection(self,C1,C2,n,m,layer=-1):
        A1,b1 = self.get_map_at_point(C1[0], layer)
        A2,b2 = self.get_map_at_point(C2[0], layer)
        
        H1 = (self.polyhedral_decomposition[layer][n].A, self.polyhedral_decomposition[layer][n].b)#self.h_rep_at_point(C1[0], layer)
        H2 = (self.polyhedral_decomposition[layer][m].A, self.polyhedral_decomposition[layer][m].b)#self.h_rep_at_point(C2[0], layer)

        contract_mask = [[],[]]
        
        bounds = [(None, None) for _ in range(self.model.input_size)] + [(0, 0)]
        c = np.zeros(self.model.input_size+1)
        c[-1] = 1
        H_constraint = np.hstack([H1[0], -np.ones([H1[0].shape[0],1])])
        H_eq = np.zeros([A1.shape[0], self.model.input_size+1])
        H_eq[:,:self.model.input_size] = A1.detach().numpy()
        for n, x in enumerate(C2): #check if points in C2 are in the image of A1
            y = A2@x+b2
            A1_intersect = linprog(
                c = c,
                A_ub = H_constraint,
                b_ub = H1[1],
                A_eq = H_eq,
                b_eq = (y-b1).detach().numpy(),
                method='highs',
                bounds=bounds)
            if A1_intersect.success:
                transformed_x = (A1@torch.Tensor(A1_intersect.x[:-1])+b1[None])[0]
                is_member = (torch.allclose(transformed_x, y, atol=0) and
                             np.all(H1[0]@A1_intersect.x[:-1]<=H1[1]+0))
                if is_member:
                    contract_mask[0].append(n)
                    
        H_constraint = np.hstack([H2[0], -np.ones([H2[0].shape[0],1])])
        H_eq = np.zeros([A2.shape[0], self.model.input_size+1])
        H_eq[:,:self.model.input_size] = A2.detach().numpy()
        for n, x in enumerate(C1): #check if points in C1 are in the image of A2
            y = A1@x+b1
            A2_intersect = linprog(
                c = c,
                A_ub = H_constraint,
                b_ub = H2[1],
                A_eq = H_eq,
                b_eq = (y-b2).detach().numpy(),
                method='highs',
                bounds=bounds)
            if A2_intersect.success:
                transformed_x = (A2@torch.Tensor(A2_intersect.x[:-1])+b2[None])[0]
                is_member = (torch.allclose(transformed_x, y, atol=0) and
                             np.all(H2[0]@A2_intersect.x[:-1]<=H2[1]+0))            
                if is_member:
                    contract_mask[1].append(n)
                    
        if len(contract_mask[0])>0 and len(contract_mask[1])>0:
            contract_mask = [[i,j] for i in contract_mask[0] for j in contract_mask[1]]
        else:
            contract_mask = []
        # ptope1 = pc.Polytope(H1[0],H1[1])
        # ptope2 = pc.Polytope(H2[0],H2[1])
        # C1_new = torch.Tensor(self.populate_polyhedra(H1,n_samples=1000))#[:,None]
        # C2_new = torch.Tensor(self.populate_polyhedra(H2,n_samples=1000))#[:,None]
        
        # # plt.plot(C1_new,0*C1_new,'.')
        # # plt.plot(C2_new,0*C2_new,'.')
        # region = pc.Region([ptope1,ptope2])
        # region.plot()
        # plt.plot(C1_new[:,0],C1_new[:,1],'.')
        # plt.plot(C2_new[:,0],C2_new[:,1],'.')
        # plt.plot(C1[:,0],C1[:,1],'b.')
        # plt.plot(C2[:,0],C2[:,1],'r.')

        # plt.plot([A1_intersect.x[:-1][0]],[A1_intersect.x[:-1][1]],'b.')
        # # plt.plot([A2_intersect.x[:-1][0]],[A2_intersect.x[:-1][1]],'r.')
        # T1 = (A1@C1.T+b1[:,None]).detach().numpy().T
        # T2 = (A2@C2.T+b2[:,None]).detach().numpy().T
        # T1_model = self.model(C1)[layer].detach().numpy()#(A1@C1.T+b1[:,None]).detach().numpy().T
        # T2_model = self.model(C2)[layer].detach().numpy()#(A2@C2.T+b2[:,None]).detach().numpy().T
        # plt.figure()
        # plt.plot(T1[:,0],T1[:,1],'.')
        # plt.plot(T2[:,0],T2[:,1],'.')
        # plt.plot(T1_model[:,0],T1_model[:,1],'.')
        # plt.plot(T2_model[:,0],T2_model[:,1],'.')    
        return np.array(contract_mask)


    def compute_overlap_decomp(self,X,sensitivity = 1):
        overlap_decomps = []
        for n in range(len(self.model.hidden_sizes)+1):
            overlap_decomps.append(self.find_overlapping_eq_classes(X,layer=n, sensitivity=sensitivity)[0])
        self.overlap_decomposition = overlap_decomps
        return overlap_decomps

    def compute_all_decomps(self, X, sensitivity = 1):
        self.local_decomposition = self.compute_codeword_eq_classes(X,mode='local')
        self.global_decomposition = self.compute_codeword_eq_classes(X,mode='global')
        self.rank_decomposition = self.compute_rank_classes(X)
        self.overlap_decomposition = self.compute_overlap_decomp(X, sensitivity=sensitivity)
        self.polyhedral_decomposition = self.weights_to_inpolyhedra(X,layer = len(self.model.hidden_sizes))

        
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
        self.union_find = union_find
        
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
            contract_mask = self.union_find(contract_mask)
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
# res = 40
# x = np.linspace(-3, 3, res)
# y = np.linspace(-3, 3, res)
# neighbor_scale = (max(x)-min(x))/(res-1)
# neighbor_scale = np.sqrt(2*(neighbor_scale**2))+1e-12
# xx, yy = np.meshgrid(x, y )
# pcloud = np.column_stack((xx.ravel(), yy.ravel()))
# # pcloud = 2*(np.random.rand(res,3)-0.5)
# # pcloud = np.vstack([np.cos(np.linspace(-np.pi,np.pi,res,endpoint=False)),np.sin(np.linspace(-np.pi,np.pi,res,endpoint=False))]).T #+ 1
# # pcloud = pcloud + 0.005*np.random.randn(*pcloud.shape)
# # d = cdist(pcloud,pcloud)
# # neighbor_scale = 2*np.min(d[d>1e-12])+0.05
# # pcloud = np.linspace(0,1,res)[:,None]
# # pcloud=np.random.randn(25,2).T

# sc = SimpComp(threshold=neighbor_scale,dim=4,method='Rips')
# tree = sc.assign_complex(pcloud)
# # sc.plot_simplicial_complex(pcloud,tree,[[0,2,32,71],[1,9]])
# #%%
# hidden_sizes = [3]*3
# model = FeedforwardNetwork(2,hidden_sizes,out_layer_sz=2, init_type='none', activation=nn.ReLU)
# decomps = NetworkDecompositions(model)
# all_phedra = decomps.weights_to_all_polyhedra(2)
# phedra = decomps.weights_to_inpolyhedra(torch.Tensor(pcloud),layer=2,bbox=[-100,100])
# polyhedra = decomps.weights_to_inpolyhedra(torch.Tensor(pcloud),layer=1,bbox=[-100,100])

# #%%
# for n in range(len(hidden_sizes)):
#     region = pc.Region(all_phedra[n])
#     region.plot()
#     plt.xlim(-10,10)
#     plt.ylim(-10,10)
#     [plt.scatter(pcloud[clss,0],pcloud[clss,1]) for clss in decomps.global_decomposition[0][n]]
# #%%
# # hom = sc.compute_homology(tree,subcomplex_list=[])
# # print(hom)
# import random
# seed = 15

# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# def gingerbread_map(x):
#     x_ = 1-x[:,1]+np.abs(x[:,0])
#     y_ = x[:,0]
#     return np.vstack([x_,y_]).T

# def duffing_map(x): 
#     x_ = x[:,1]
#     y_ = -0.2*x[:,0]+2.75*x[:,1]-x[:,1]**3
#     return np.vstack([x_,y_]).T

# def bogdanov_map(x, eps=0, k=1.2, mu=2):
#     y_ = x[:,1]+eps*x[:,1]+k*x[:,0]*(x[:,0]-1)+mu*x[:,0]*x[:,1]
#     x_ = x[:,0]+y_
#     return np.vstack([x_,y_]).T

# def henon_map(x):
#     x_ = 1-1.4*x[:,0]**2+x[:,1]
#     y_ = 0.3*x[:,1]
#     return np.vstack([x_,y_]).T

# def bakers_map(x):
#     x_1 = x[x[:,0]<0.5]
#     x_2 = x[x[:,0]>=0.5]
#     x_ = np.concatenate([2*x_1[:,0],2-2*x_2[:,0]])
#     y_ = np.concatenate([x_1[:,1]/2, 1 - x_1[:,1]/2])
#     return np.vstack([x_,y_]).T

# def recurse_point(x):
#     return 1.8*(x-x**5)


# hidden_sizes = [3]*5
# model = FeedforwardNetwork(2,hidden_sizes,out_layer_sz=2, init_type='none', activation=nn.ReLU)
# # model = lambda x: torch.nn.RNN(2,2,1,nonlinearity='relu')(torch.Tensor(x))[0]

# P_ = torch.randn(hidden_sizes[0],hidden_sizes[0])
# E, V = torch.linalg.eig(P_)
# E = E/(abs(E).max())
# P = (V@torch.diag(E)@torch.linalg.pinv(V)).real

# for n, layer in enumerate(model.layers): #make all weight matrices the same for RNN condition
#     if isinstance(layer, nn.Linear) and n>2:
#         layer.weight = model.layers[2].weight#nn.Parameter(P)
#         layer.bias = model.layers[2].bias  
# #%%
# thresh = 0#neighbor_scale/50
# MPH =  MapHomology(model, pcloud, scomplex_analyzer=sc, gluing_threshold=thresh, neural_model=model)
# map_homology, fs, ovrlap_complex = MPH.map_homology_sequence(len(hidden_sizes))
# print(map_homology)
# get_all_decomps = MPH.decomps.compute_all_decomps(torch.Tensor(pcloud),sensitivity=thresh)
# # map_diagram = map_homology
# import matplotlib.pyplot as plt

# # [plt.plot(fs[i],'.') for i in range(len(fs))]
# reduced_mfld = []
# reducer = Isomap(n_components=4)
# for i in range(len(fs)):
#     pcl = reducer.fit_transform(fs[i])
#     plt.scatter(pcl[:,0], pcl[:,1],s=1)   
#     reduced_mfld.append(pcl)     

# for i in range(len(ovrlap_complex)):
#     plt.figure()
#     # sc.plot_simplicial_complex(pcloud, tree, ovrlap_complex[i])
#     # sc.plot_simplicial_complex(pcloud, tree, MPH.decomps.overlap_decomopsition[i],MPH.decomps.local_decomposition[0][i])
#     sc.plot_simplicial_complex(reduced_mfld[i], tree, MPH.decomps.overlap_decomopsition[i],MPH.decomps.rank_decomposition[0][i-1])
# # 

# #%%
# import umap.umap_ as umap
# from scipy.sparse.csgraph import shortest_path, laplacian

# l = -1
# d = cdist(pcloud,pcloud).copy()
# for decmp in MPH.decomps.overlap_decomposition[l]:
#     d[np.ix_(decmp,decmp)] = 0
# # d = shortest_path(d)
# # reducer = umap.UMAP(n_components=3, n_neighbors=8,min_dist=1,metric='precomputed')
# reducer = Isomap(n_components=3, n_neighbors=8, metric='precomputed')
# mfld = reducer.fit_transform(d).T
# # mfld = reducer.fit_transform(fs[l]).T#cdist(pcloud,pcloud)).T


# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Create the scatter plot
# [ax.scatter(mfld[0,clss],mfld[1,clss],mfld[2,clss]) for clss in MPH.decomps.global_decomposition[0][l]]
# [ax.scatter(mfld[0,clss],mfld[1,clss],mfld[2,clss],c='k') for clss in MPH.decomps.overlap_decomposition[l]]









