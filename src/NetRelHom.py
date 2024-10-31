import gudhi
from sympy import Matrix, FiniteField
from sympy.matrices.normalforms import smith_normal_form
import numpy as np

class SimpComp:
    def __init__(self, method='Rips', threshold = 4, dim=2):
        self.method = method
        self.threshold = threshold
        self.dim = dim
        
    def assign_complex(self, point_cloud):
        if self.method=='Rips':
            rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=self.threshold)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.dim)
        return simplex_tree
    
    def get_boundary_matrices(self, simplicial_complex):
        simplices = simplicial_complex.get_simplices()
        skeletons = {k: [] for k in range(self.dim+1)}
        for simplex in simplices:
            skeletons[len(simplex[0])-1].append(simplex[0])
        
        boundary_matrices = {k: [] for k in range(self.dim)}
        boundary_matrices[0] = np.ones([1,len(skeletons[0])])
    
        for k in range(self.dim):
            boundary_matrices[k+1] = np.zeros([len(skeletons[k]),len(skeletons[k+1])])
            for n,simp in enumerate(skeletons[k+1]):
                faces = [set(b) <= set(simp) for b in skeletons[k]]
                boundary_matrices[k+1][faces,n] = 1
        boundary_matrices[self.dim+1] = np.ones([1,len(skeletons[0])])
        return boundary_matrices
    
    def compute_homology(self, simplicial_complex):
        boundary_matrices = self.get_boundary_matrices(simplicial_complex)
        Z_ranks = []
        B_ranks = []
        for D in boundary_matrices.values():
            C = Matrix(D.astype(int))
            C_reduced = smith_normal_form(C,domain=FiniteField((2)))
            B_ranks.append(np.sum(C_reduced))
            Z_ranks.append(C_reduced.shape[1]-np.sum(C_reduced))
        Betti = np.array(Z_ranks)[:-1]-np.array(B_ranks)[1:]
        return Betti, Z_ranks, B_ranks
        

x = np.linspace(0, 2, 3)  # 3 points along x-axis
y = np.linspace(0, 2, 3)  # 3 points along y-axis
xx, yy = np.meshgrid(x, y)
pcloud = np.column_stack((xx.ravel(), yy.ravel()))
pcloud=np.random.randn(50,2)

sc = SimpComp(threshold=np.sqrt(2),dim=3)
tree = sc.assign_complex(pcloud)

mats = sc.get_boundary_matrices(tree)
hom = sc.compute_homology(tree)
print(hom)