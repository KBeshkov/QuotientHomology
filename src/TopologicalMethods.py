# implementation of algorithms
import math
import numpy as np
import time as time
from cmath import sqrt as isqrt
import warnings


from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
import matplotlib.gridspec as gridspec
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
#import seaborn as sns

plt.style.use("default")  # ../misc/report.mplstyle')

import numpy.matlib
import numpy.polynomial as npoly
import random

from scipy import stats
from scipy.spatial import ConvexHull
from scipy import sparse
from scipy.sparse.csgraph import shortest_path, laplacian
from scipy.spatial.distance import cdist, jensenshannon
from scipy.special import iv, hyp1f1
from scipy.ndimage import laplace
from scipy.optimize import linear_sum_assignment, curve_fit
from scipy.stats import (
    rankdata,
    ortho_group,
    percentileofscore,
    spearmanr,
    special_ortho_group,
    ranksums
)
from scipy.signal import find_peaks, argrelextrema
from scipy.spatial.transform import Rotation
from scipy.interpolate import splrep, splev

from sklearn.metrics import pairwise_distances, r2_score, silhouette_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans

import networkx as nx
from ripser import ripser as tda
from persim import plot_diagrams, bottleneck, sliced_wasserstein

import torch


#np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class ManifoldGenerator:
    """
    Manifold generator class.
    """

    def __init__(self):
        return None

    def __call__(self, amount_of_points, manifold_type, *args):
        """


        Parameters
        ----------
        amount_of_points : int
            The number of points to be sampled from the manifold.
        manifold_type : string
            String specifying the type of manifold to be sampled from.
        *args : list
            List of args specific to each particualr manifold.

        Returns
        -------
        call
            calls the specified method.

        """
        return getattr(self, manifold_type)(amount_of_points, args)

    def R3(self, amount_of_points, *args):
        """
        Samples equidistant points from $\mathbb{R}^3$.

        Parameters
        ----------
        amount_of_points : int
            The number of samples.
        *args : list
            Lower and upper xyz coordinates of the bounding box.

        Returns
        -------
        R : numpy array
            array of xyz coordinates.

        """
        x, y, z = np.meshgrid(
            np.linspace(-args[0], args[0], amount_of_points),
            np.linspace(-args[1], args[1], amount_of_points),
            np.linspace(-args[2], args[2], amount_of_points),
        )
        R = np.vstack([x.flatten(), y.flatten(), z.flatten()])
        return R

    def S1(self, amount_of_points, *args):
        """
        Samples equidistant points from $S^1$.

        Parameters
        ----------
        amount_of_points : int
            The number of samples.
        *args : list
            Radius.

        Returns
        -------
        numpy array
            array of xy circle coordinates.

        """
        r, theta = args[0], np.linspace(0, 2 * np.pi, amount_of_points, endpoint=True)
        return np.array([r * np.cos(theta), r * np.sin(theta)])

    def S2(self, amount_of_points, *args):
        """
        Samples equidistant points from $S^2$.

        Parameters
        ----------
        amount_of_points : int
            The number of samples.
        *args : list
            Radius.

        Returns
        -------
        numpy array
            array of xyz sphere coordinates.

        """
        r, phi, theta = (
            args[0],
            np.linspace(0, np.pi, amount_of_points, endpoint=False),
            np.linspace(0, 2 * np.pi, amount_of_points, endpoint=False),
        )
        Phi, Theta = np.meshgrid(phi, theta)
        Phi, Theta = Phi.flatten(), Theta.flatten()
        return np.array(
            [
                r * np.cos(Theta) * np.sin(Phi),
                r * np.sin(Theta) * np.sin(Phi),
                r * np.cos(Phi),
            ]
        )
    
    def S2_Icosphere(self, amount_of_points, *args):
        """
        Creates a geodesic icosahedron which is a decent approximation of a sphere.
        The amount of points in this case describes the subdivision frequency.
        Which is further explained in: https://github.com/vedranaa/icosphere

        Parameters
        ----------
        amount_of_points : int
            The subdivision frequency.
        *args: list
            Placeholder list, for consistency with other class methods.

        Returns
        -------
        vertices: numpy array
            Returns the vertices of the icosphere.

        """
        from icosphere import icosphere
        vertices, _ = icosphere(amount_of_points)
        return vertices.T

    def S3(self, amount_of_points, *args):
        """
        Samples equidistant points from $S^3$.

        Parameters
        ----------
        amount_of_points : int
            The number of samples.
        *args : list
            Radius.

        Returns
        -------
        numpy array
            array of xyzw sphere coordinates.

        """
        r, phi, theta, psi = (
            args[0],
            np.linspace(0, 2 * np.pi, amount_of_points, endpoint=False),
            np.linspace(0, np.pi, amount_of_points, endpoint=False),
            np.linspace(0, np.pi, amount_of_points, endpoint=False)
        )
        Phi, Theta, Psi = np.meshgrid(phi, theta, psi)
        Phi, Theta, Psi = Phi.flatten(), Theta.flatten(), Psi.flatten()
        return np.array(
            [
                r * np.sin(Theta) * np.sin(Psi) * np.sin(Phi),
                r * np.sin(Theta) * np.sin(Psi) * np.cos(Phi),
                r * np.cos(Theta) * np.sin(Psi),
                r * np.cos(Psi),
            ]
        )

        
    def T2(self, amount_of_points, *args):
        """
        Samples equidistant points from $T^2$.

        Parameters
        ----------
        amount_of_points : int
            The number of samples.
        *args : list
            Small and large torus radius.

        Returns
        -------
        numpy array
            array of xyz torus coordinates.

        """
        R, r, phi, theta = (
            args[0],
            args[1],
            np.linspace(0, 2 * np.pi, amount_of_points, endpoint=False),
            np.linspace(0, 2 * np.pi, amount_of_points, endpoint=False),
        )
        Phi, Theta = np.meshgrid(theta, phi)
        Phi, Theta = Phi.flatten(), Theta.flatten()
        return np.array(
            [
                (R + r * np.cos(Theta)) * np.cos(Phi),
                (R + r * np.cos(Theta)) * np.sin(Phi),
                r * np.sin(Theta),
            ]
        )

    def T2F(self, amount_of_points, *args):
        """
        Samples equidistant points from a flat torus $T^2$.

        Parameters
        ----------
        amount_of_points : int
            The number of samples.
        *args : list
            Torus radius.

        Returns
        -------
        numpy array
            array of xyzw torus coordinates.

        """
        phi, theta = (
            np.linspace(0, 2 * np.pi, amount_of_points, endpoint=False),
            np.linspace(0, 2 * np.pi, amount_of_points, endpoint=False),
        )
        Phi, Theta = np.meshgrid(theta, phi)
        Phi, Theta = Phi.flatten(), Theta.flatten()
        return np.array(
            [
                args[0]*np.cos(Phi),
                args[0]*np.sin(Phi),
                args[0]*np.cos(Theta),
                args[0]*np.sin(Theta),
            ]
        )/np.sqrt(2)
    
    def TN(self, amount_of_points, *args):
        """
        Samples equidistant points from a flat n-Torus $\prod{S^1}$
        
        Parameters
        ----------
        amount_of_points : int
            The number of samples.
        *args : list
            Torus radius and intrinsic dimension.

        Returns
        -------
        numpy array
            array of torus coordinates.

        """
        angles = [np.linspace(0, 2 * np.pi, amount_of_points, endpoint=False)]*args[1]
        angle_grid = np.meshgrid(*angles)
        
        TN_array = []
        for phi_i in angle_grid:
            TN_array.append(args[0]*np.cos(phi_i.flatten()))
            TN_array.append(args[0]*np.sin(phi_i.flatten()))
        return np.array(TN_array)
        
    def ParametricKnot(self, amount_of_points, a, b, c, d, *args):
        r, theta = args[0], np.linspace(0, 2 * np.pi, amount_of_points, endpoint=False)
        return np.array([r * np.cos(a*theta) * np.cos(b*theta), r * np.cos(c*theta) * np.sin(d*theta)])
        
    
    def Sn(self, amount_of_points, *args):
        """
        Samples random points from an $S^n$ sphere.

        Parameters
        ----------
        amount_of_points : int
            The number of samples.
        *args : list
            Dimension of the sphere.

        Returns
        -------
        numpy array
            array of $[x_1,x_2,...,x_N]$ sphere coordinates.

        """
        x = np.random.randn(amount_of_points, args[0][0])
        x = x.T / np.linalg.norm(x, axis=1)
        return x

    def KB(self, amount_of_points, immersion, *args):
        """


        Samples points from the Klein bottle.

        Parameters
        ----------
        amount_of_points : int
            The number of samples.
        immersion: bool
        Whether to use the 3 dimensional immersion or the 4 dimensional embedding.
        *args : list
            R, P and epsilon parameters for the 4 dimensional embedding.

        Returns
        -------
        numpy array
            array of Klein bottle coordinates.

        """
        if immersion == True:
            phi = np.linspace(0, np.pi, amount_of_points)
            theta = np.linspace(0, 2 * np.pi, amount_of_points)
            Phi, Theta = np.meshgrid(phi, theta)
            Phi_KB, Theta_KB = Phi.flatten(), Theta.flatten()
            x = (
                (-2 / 15)
                * np.cos(Phi_KB)
                * (
                    3 * np.cos(Theta_KB)
                    - 30 * np.sin(Phi_KB)
                    + 90 * np.sin(Phi_KB) * np.cos(Phi_KB) ** 2
                    - 60 * np.sin(Phi_KB) * np.cos(Phi_KB) ** 6
                    + 5 * np.cos(Phi_KB) * np.cos(Theta_KB) * np.sin(Phi_KB)
                )
            )
            y = (
                (-1 / 15)
                * np.sin(Phi_KB)
                * (
                    3 * np.cos(Theta_KB)
                    - 3 * np.cos(Theta_KB) * np.cos(Phi_KB) ** 2
                    - 48 * np.cos(Theta_KB) * np.cos(Phi_KB) ** 2
                    + 48 * np.cos(Theta_KB) * np.cos(Phi_KB) ** 6
                    - 60 * np.sin(Phi_KB)
                    + 5 * np.cos(Phi_KB) * np.cos(Theta_KB) * np.sin(Phi_KB)
                    - 5 * np.cos(Theta_KB) * np.sin(Phi_KB) * np.cos(Phi_KB) ** 3
                    - 80 * np.cos(Theta_KB) * np.sin(Phi_KB) * np.cos(Phi_KB) ** 5
                    + 80 * np.cos(Theta_KB) * np.sin(Phi_KB) * np.cos(Phi_KB) ** 7
                )
            )
            z = (2 / 15) * (3 + 5 * np.cos(Phi_KB) * np.sin(Phi_KB)) * np.sin(Theta_KB)
            KB = np.array([x, y, z])
        else:
            theta = np.linspace(0, 2 * np.pi, amount_of_points)
            v = np.linspace(0, 2 * np.pi, amount_of_points)
            Theta, V = np.meshgrid(theta, v)
            Theta, V = Theta.flatten(), V.flatten()
            R, P, eps = args[0], args[1], args[2]
            x = R * (np.cos(Theta / 2) * np.cos(V) + np.sin(Theta / 2) * np.sin(2 * V))
            y = R * (np.sin(Theta / 2) * np.cos(V) + np.cos(Theta / 2) * np.sin(2 * V))
            z = P * np.cos(Theta) * (1 + eps * np.sin(V))
            w = P * np.sin(Theta) * (1 + eps * np.sin(V))
            KB = np.array([x, y, z, w])
        return KB


class PersistentHomology:
    def __init__(self):
        None

    def __call__(self, manifold, metric, normalized, dimred=0, *args, coeff=2):
        distance_matrix, birth_death_diagram, cocycles = self.homology_analysis(
            manifold, metric, dimred, *args, coeff=coeff
        )
        return [
            distance_matrix,
            self.normalize(birth_death_diagram) if normalized else birth_death_diagram,
            cocycles,
        ]
    

    def homology_analysis(self, manifold, metric, dimred, *args, coeff=2):
        """
        Computes persistent homology
        -----------------------
        Outputs the distance_matrix and birth_death_diagram for the given manifold and metric.
        """
        if dimred > 0:
            manifold = PCA(n_components=dimred).fit_transform(manifold)
        distance_matrix = metric(manifold)
        birth_death_diagram = tda(
            distance_matrix,
            distance_matrix=True,
            maxdim=args[0][0],
            n_perm=args[0][1],
            do_cocycles=True,
            coeff=coeff,
        )
        diagram = birth_death_diagram["dgms"]
        cocycles = birth_death_diagram["cocycles"]
        return distance_matrix, diagram, cocycles
    
    def homology_from_dmat(self, distance_matrix, normalized=True, *args, coeff=2):
        '''
        Computes persistent homology from a distance matrix

        '''
        birth_death_diagram = tda(
            distance_matrix,
            distance_matrix=True,
            maxdim=args[0][0],
            n_perm=args[0][1],
            do_cocycles=True,
            coeff=coeff,
        )
        diagram = birth_death_diagram["dgms"]
        cocycles = birth_death_diagram["cocycles"]
        return [self.normalize(diagram) if normalized else birth_death_diagram, cocycles]
    
    def relative_homology(self, manifold, submanifolds, metric, dimred, *args, coeff=2, quotient_metric='Iso'):
        """
        Computes relative persistent homology
        -----------------------
        Outputs the distance_matrix and birth_death_diagram for the given manifold, submanifold and metric.
        """
        if dimred > 0:
            manifold = PCA(n_components=dimred).fit_transform(manifold)
        distance_matrix = metric(manifold)
        
        for submanifold in submanifolds:
            for simplex_i in submanifold:
                for simplex_j in submanifold:
                    distance_matrix[simplex_i, simplex_j] = 0
           
            
            if quotient_metric=='Iso': 
                # Assuming the quotient space is the orbit of an isometry group see:
                #https://math.stackexchange.com/questions/3117663/quotient-space-metric-with-nice-equivalence-classes?noredirect=1&lq=1
                #and Theorem 2.1 https://www.degruyter.com/document/doi/10.1515/forum-2012-0152/html
                # Also works if the quotient is a single connected set
                for point in range(len(manifold)):
                    distance_matrix[point,submanifold] = np.min(distance_matrix[point, submanifold])
                    distance_matrix[submanifold, point] = np.min(distance_matrix[submanifold, point])            
            
            
            if quotient_metric=='Average':
                for point in range(len(manifold)):
                    distance_matrix[point,submanifold] = np.mean(distance_matrix[point, submanifold])
                    distance_matrix[submanifold, point] = np.mean(distance_matrix[submanifold, point])      
                    
        set_complement = np.setdiff1d(np.arange(len(distance_matrix)), np.concatenate(submanifolds))
        quotient_set = np.sort(np.append(set_complement, [submanifold[0] for submanifold in submanifolds]))
        distance_matrix = distance_matrix[quotient_set,:][:,quotient_set].copy(order='C')
        distance_matrix = shortest_path(distance_matrix)
        np.fill_diagonal(distance_matrix,0)
        
        birth_death_diagram = tda(
            distance_matrix,
            distance_matrix=True,
            maxdim=args[0][0],
            n_perm=args[0][1],
            do_cocycles=True,
            coeff=coeff,
        )
        diagram = birth_death_diagram["dgms"]
        cocycles = birth_death_diagram["cocycles"]
        return distance_matrix, diagram, cocycles

    def normalize(self, birth_death_diagram):
        """
        Normalized birth/death distance
        -----------------------
        Outputs the persistence of a feature normalized between 0 and 1.
        """
        birth_death_diagram_copy = [np.copy(birth_death_diagram[i]) for i in range(len(birth_death_diagram))]
        a = np.concatenate(birth_death_diagram_copy).flatten()
        finite_dgm = a[np.isfinite(a)]
        ax_max = np.max(finite_dgm)
        norm_pers = []
        for i in range(len(birth_death_diagram)):
            norm_pers.append((birth_death_diagram[i]) / ax_max)
        return norm_pers


    def perm_test(self, manifold, metric, n_perms, dimred=0, dim=1, pval=99,nperms=None):
        """
        Permutation test to estimate a p value for a manifold appearing by chance
        ----------------------
        Outputs p values at different confidence thresholds.
        """
        hom = self.normalize(
            self.homology_analysis(manifold, metric, dimred, [dim, nperms])[1]
        )
        cycle_lengths = hom[dim][:, 1] - hom[dim][:, 0]
        percentiles = []
        for n in range(n_perms):
            permute = np.vstack(
                [np.random.permutation(len(manifold)) for i in range(len(manifold.T))]
            ).T
            temp_manifold = np.vstack(
                [manifold[permute[:, i], i] for i in range(len(manifold.T))]
            ).T
            temp_hom = self.normalize(
                self.homology_analysis(temp_manifold, metric, dimred, [dim, nperms])[1]
            )
            if len(temp_hom[dim]) > 0:
                persistances = temp_hom[dim][:, 1] - temp_hom[dim][:, 0]
                percentiles.append(np.max(persistances))
            else:
                percentiles.append(0)
        pvalue = np.percentile(percentiles, pval)

        print(
            str(sum(cycle_lengths > pvalue)) + " significant " + str(dim) + "-cocycles"
        )
        return pvalue, cycle_lengths, percentiles

    def barcode_plot(
        self, diagram, dims=2, cutoff_ax=0, pval=0, normalize=True, figsize=(14, 6)
    ):
        if normalize:
            diagram = self.normalize(diagram)
        results = {}
        if cutoff_ax == 0:
            largest_pers = 0
            for d in range(dims):
                results["h" + str(d)] = diagram[d]
                if np.max(diagram[d][np.isfinite(diagram[d])]) > largest_pers:
                    largest_pers = np.max(diagram[d][np.isfinite(diagram[d])])
        elif cutoff_ax != 0:
            largest_pers = cutoff_ax
        clrs = cm.get_cmap(
            "Set2"
        ).colors  # ['tab:blue','tab:orange','tab:green','tab:red']#['b','r','g','m','c']
        diagram[0][~np.isfinite(diagram[0])] = largest_pers + 0.1 * largest_pers
        plot_prcnt = 0 * np.ones(dims)
        to_plot = []
        for curr_h, cutoff in zip(diagram, plot_prcnt):
            bar_lens = curr_h[:, 1] - curr_h[:, 0]
            plot_h = curr_h[bar_lens >= np.percentile(bar_lens, cutoff)]
            to_plot.append(plot_h)
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(dims, 4)
        for curr_betti, curr_bar in enumerate(to_plot):
            ax = fig.add_subplot(gs[curr_betti, :])
            for i, interval in enumerate(reversed(curr_bar)):
                plt.plot(
                    [interval[0], interval[1]], [i, i], color=clrs[curr_betti], lw=1.5
                )
                if pval > 0 and curr_betti==1:
                    plt.plot(
                        [interval[0], interval[0] + pval],
                        [i, i],
                        color="k",
                        alpha=0.05,
                        lw=2,
                    )
            if curr_betti == dims - 1:
                ax.set_xlim([0, largest_pers + 0.01])
                ax.set_ylim([-1, len(curr_bar)])
                ax.set_yticks([])
            else:
                ax.set_xlim([0, largest_pers + 0.01])
                ax.set_xticks([])
                ax.set_ylim([-1, len(curr_bar)])
                ax.set_yticks([])
                
            
                
    

class BarcodeAnalyzer:
    def __init__(self):
        None

    def histograms(self, barcode, binslst=[20, 15, 10]):
        """
        Betti number persistance histograms
        -----------------------
        Outputs a list of histograms for persistance/birth/death for all specified Betti numbers.
        """
        histograms = []
        persistant_values = []
        for i in range(len(barcode)):
            persistant_values.append(barcode[i][:, 1] - barcode[i][:, 0])
            b_i = np.histogram(
                barcode[i][:, 1] - barcode[i][:, 0],
                np.linspace(0, 0.1, binslst[i]),
                density=True,
            )
            histograms.append(b_i)
        return [persistant_values, histograms]

    def curves(self, barcode, number_of_curves=3, epsilon=0.001, duration=1000):
        """
        Betti curves
        -----------------------
        Outputs a list of Betti curves, indicating how many n-holes there are for a given epsilon value.
        """
        curves = []
        finite_dgm = barcode[0][np.isfinite(barcode[0])]
        x_maxs = []
        for i in range(len(barcode)):
            if len(barcode[i]) > 0 and i != 0:
                x_maxs.append(np.max(barcode[i]))
            else:
                x_maxs.append(np.max(finite_dgm))
        e = np.copy(epsilon)
        for i in range(len(barcode)):
            Bn = np.zeros(duration)
            for j in range(duration):
                count = 0
                for k in range(len(barcode[i][:, 0])):
                    if barcode[i][k, 0] < epsilon and barcode[i][k, 1] > epsilon:
                        count = count + 1
                Bn[j] = count
                epsilon = epsilon + e
            epsilon = np.copy(e)
            curves.append(Bn)
        return [np.linspace(0, epsilon * duration, duration), curves]

    def plotCocycle2D(
        self,
        D,
        X,
        thresh,
        labels=None,
        node_cmap="Greys",
        edge_color="black",
        n_labels=12,
    ):
        """
        Given a 2D point cloud X, display the graph
        at a given threshold "thresh"
        """
        N = X.shape[0]
        c = plt.get_cmap(node_cmap)
        C = c(np.array(np.round(np.linspace(0, 255, N)), dtype=np.int32))
        D_thresh = np.copy(D)
        for i in range(N):
            for j in range(N):
                if D[i, j] > thresh:
                    D_thresh[i, j] = 0
                # plt.plot([X[i,0],X[j,0]],[X[i,1],X[j,1]],color=edge_color,linewidth=1,alpha=0.1)
        G = nx.from_numpy_array(D_thresh / np.linalg.norm(D_thresh))
        circ_layout = nx.circular_layout(G)
        lbl_offset = 1.25
        nx.draw_circular(
            G,
            node_color=np.linspace(0, 1, len(G)),
            cmap=node_cmap,
            node_size=160,
            alpha=0.75,
            edge_color=edge_color,
            linewidths=0.001,
        )
        # Plot vertex labels
        # plt.scatter(X[:, 0], X[:, 1],color='green',edgecolor='black',s=8,linewidths=0.25)
        if n_labels != 0:
            txt_labels = int(N / n_labels)
            for i in range(n_labels):
                plt.text(
                    lbl_offset * circ_layout[i * txt_labels][0],
                    lbl_offset * circ_layout[i * txt_labels][1],
                    str(int(round(labels[i * txt_labels]))),
                    color="black",
                    ha="center",
                    va="center",
                )  # C[i*txt_labels])
        plt.axis("equal")



class GeodesicKNN:
    def __init__(self, k=2, adaptive=False, mode='distance'):
        self.k = k
        self.adaptive = adaptive
        self.mode = mode
        
    def symmetrize(self, d):
        d_stack = np.stack([d,d.T])
        d_stack[d_stack==0] = np.inf
        d = np.min(d_stack,0)
        return d

    def fit(self, X):
        d = self.symmetrize(kneighbors_graph(X, self.k, mode=self.mode).toarray())
        d_geod = shortest_path(d)
        if self.adaptive:
            store_dmats = []
            adaptive_k = self.k
            store_dmats.append(d_geod)
            while np.any(np.isinf(d_geod)):
                adaptive_k += 1
                d = self.symmetrize(kneighbors_graph(X, adaptive_k, mode=self.mode).toarray())
                d[d == 0] = np.inf
                d_geod = shortest_path(d)
                store_dmats.append(d_geod)
            d_geod = store_dmats[0]
            for i in range(len(store_dmats)):
                d_current = store_dmats[i]
                d_geod[
                    np.logical_and(~np.isinf(d_current), np.isinf(d_geod))
                ] = d_current[np.logical_and(~np.isinf(d_current), np.isinf(d_geod))]
        return d_geod

    

def adjacency_structure(adj_matrix,adj_range=2):
    """
    Generates a weight matrix based on the adjacency matrix of a graph.

    Parameters
    ----------
    adj_matrix : torch Tensor or numpy array
        An adjacency matrix.
    adj_range : int, optional
        The range at which neurons are considered "neighbors" and assigend a positive weight. The default is 2.

    Returns
    -------
    w : torch Tensor
        The output weight matrix.

    """
    w = torch.zeros(*adj_matrix.shape)
    neighbors = torch.logical_and(adj_matrix<=adj_range,adj_matrix!=0)
    w[neighbors] = 1
    N_neighbors = torch.sum(neighbors,1)
    N_outside = torch.sum(adj_matrix>adj_range,1)
    for i in range(len(w)):
        w[i,(adj_matrix>adj_range)[i]] = -N_neighbors[i]/N_outside[i]
    return w


def calculate_laplacian(adjacency):
    '''   
    Calculates the graph laplacian of a graph, given by the adjacecny matrix.

    Parameters
    ----------
    adjacency : numpy array
        The adjecency matrix of a graph.

    Returns
    -------
    The Laplacian of a graph.

    '''
    degrees = np.diag(np.sum(adjacency,0))
    laplacian = degrees-adjacency
    return laplacian