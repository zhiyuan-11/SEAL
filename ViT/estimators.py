import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class Frob_Entropy(nn.Module):
    def __init__(self, kernel_type='rbf', kappa=1.0):
        """
        Frob_Entropy class for computing kernel-based entropy measures.
        It supports two kernel types: 'rbf' and 'vMF'.

        In 'rbf' mode, we align each z_start to a fixed 'pole' direction
        (e.g., [1, 0, 0, ...]) before computing the RBF kernel.
        In 'vMF' mode, no alignment is performed.

        Args:
            kernel_type (str): Kernel type to use ('rbf' or 'vmf').
            kappa (float): Concentration parameter for vMF kernel.
        """
        super(Frob_Entropy, self).__init__()
        self.kernel_type = kernel_type
        self.kappa = kappa

        # Define a fixed 'pole' direction for RBF alignment.
        # We will initialize it later once we know the dimension (c).
        self.pole = None

    def compute_rotation_matrix_batch(self, z_start, delta):
        """
        Compute the rotation (via Householder reflection matrices) for a batch
        of z_start so that each z_start[i] is aligned to the fixed 'pole' direction.

        NOTE: We keep the signature (z_start, delta) for compatibility with the
        original structure, but in this modified version, 'delta' is not used
        to define the reflection axis. Instead, the reflection is determined by
        (z_start[i] -> self.pole).

        Args:
            z_start (torch.Tensor): A (batch, c) tensor of starting points.
            delta (torch.Tensor): A (batch, c) tensor (not used for alignment here).

        Returns:
            torch.Tensor: A (batch, c, c) tensor of Householder reflection matrices
                          that rotate each z_start[i] to self.pole.
        """
        batch_size, c = z_start.shape

        # Initialize the pole vector [1, 0, 0, ..., 0] if not set yet
        if self.pole is None:
            pole = torch.zeros(c, device=z_start.device)
            pole[0] = 1.0
            self.pole = pole

        # (1) Normalize z_start
        z_norm = z_start / (torch.norm(z_start, p=2, dim=-1, keepdim=True) + 1e-8)  # (batch, c)

        # (2) For each sample, define v_i = z_norm[i] - pole
        #     Then use Householder reflection: R_i = I - 2 * v_i v_i^T
        pole_norm = self.pole  # already [1,0,0,...]
        v = z_norm - pole_norm.unsqueeze(0)  # (batch, c)
        v = v / (torch.norm(v, p=2, dim=-1, keepdim=True) + 1e-8)  # normalize each v_i

        # (3) Construct reflection matrices
        I = torch.eye(c, device=z_start.device).unsqueeze(0).expand(batch_size, -1, -1)
        vvT = torch.einsum('bi,bj->bij', v, v)  # (batch, c, c)
        R = I - 2.0 * vvT                       # (batch, c, c)

        return R

    def align_deltas_batch(self, deltas, z_starts):
        """
        Align all deltas in a batch to the 'pole' direction by rotating
        each z_start[i] -> pole, and applying the same rotation to deltas[i].

        Args:
            deltas (torch.Tensor): (batch, c) tensor of delta vectors.
            z_starts (torch.Tensor): (batch, c) tensor of starting points.

        Returns:
            torch.Tensor: (batch, c) aligned deltas.
        """
        # Compute rotation matrices (z_start -> pole)
        R_batch = self.compute_rotation_matrix_batch(z_starts, deltas)
        # Rotate each delta accordingly
        aligned_deltas = torch.einsum('bij,bj->bi', R_batch, deltas)
        return aligned_deltas

    def compute_rbf_kernel(self, deltas, z_starts, sigma=None):
        """
        Compute the RBF kernel using the 'pole' alignment for z_starts.

        Args:
            deltas (torch.Tensor): (batch, c) tensor of delta vectors.
            z_starts (torch.Tensor): (batch, c) tensor of starting points.
            sigma (float, optional): RBF kernel bandwidth. If None, it will be estimated.

        Returns:
            torch.Tensor: (batch, batch) RBF kernel matrix.
        """
        # (1) Align deltas (z_starts -> pole)
        aligned_deltas = deltas #self.align_deltas_batch(deltas, z_starts)

        # (2) Pairwise squared distances
        pairwise_dists = torch.cdist(aligned_deltas, aligned_deltas, p=2)

        # (3) Estimate sigma if not provided
        if sigma is None:
            upper_tri_indices = torch.triu_indices(
                pairwise_dists.size(0),
                pairwise_dists.size(0),
                offset=1
            )
            upper_tri_dists = pairwise_dists[upper_tri_indices[0], upper_tri_indices[1]]
            sigma = max(0.5 * torch.median(upper_tri_dists).item(), 1E-2)

        # (4) RBF kernel
        K = torch.exp(-pairwise_dists ** 2 / (2.0 * sigma**2))
        return K

    def von_mises_fisher_kernel(self, X, kappa):
        """
        Compute the von Mises-Fisher (vMF) kernel without any alignment.

        Args:
            X (torch.Tensor): (batch, c) input tensor.
            kappa (float): Concentration parameter.

        Returns:
            torch.Tensor: (batch, batch) vMF kernel matrix.
        """
        X_norm = X / (torch.norm(X, dim=1, keepdim=True) + 1e-8)  # (batch, c)
        cosine_similarity = torch.matmul(X_norm, X_norm.T)
        K = torch.exp(kappa * cosine_similarity)
        return K

    def get_kernel_matrix(self, deltas, z_starts, sigma=None):
        """
        Select the kernel type and compute the kernel matrix. 
        - 'rbf': use the compute_rbf_kernel() with pole alignment.
        - 'vMF': use the von_mises_fisher_kernel() with no alignment.

        Args:
            deltas (torch.Tensor): (batch, c) delta vectors.
            z_starts (torch.Tensor): (batch, c) starting points.
            sigma (float, optional): Bandwidth for RBF.

        Returns:
            torch.Tensor: (batch, batch) kernel matrix.
        """
        if self.kernel_type == 'rbf':
            return self.compute_rbf_kernel(deltas, z_starts, sigma=sigma)
        elif self.kernel_type == 'vMF':
            return self.von_mises_fisher_kernel(deltas, self.kappa)
        else:
            raise ValueError("Unsupported kernel type. Choose 'rbf' or 'vmf'.")

    def forward(self, deltas, z_starts):
        """
        Compute the entropy based on the selected kernel type.
        - For 'rbf', all z_starts are aligned to the pole, then RBF is computed.
        - For 'vMF', no alignment is performed, directly compute vMF kernel.

        Finally, we normalize the kernel by trace(K), then compute:
            entropy = - log(||K||_F^2)

        Args:
            deltas (torch.Tensor): (batch, c) delta vectors.
            z_starts (torch.Tensor): (batch, c) starting points.

        Returns:
            torch.Tensor: Scalar entropy value.
        """
        K = self.get_kernel_matrix(deltas, z_starts)  # Compute kernel (RBF or vMF)
        K = K / torch.trace(K)                       # Normalize by trace
        entropy = -torch.log(torch.norm(K, p='fro') ** 2)  # Frobenius norm -> negative log
        return entropy

class Alpha_Joint_Entropy(nn.Module):
    def __init__(self, kappa=2.0):
        """
        Joint entropy computation using the von Mises-Fisher (vMF) kernel.
        Args:
            kappa (float): Concentration parameter for the vMF kernel. Larger values emphasize angular similarity.
        """
        super(Alpha_Joint_Entropy, self).__init__()
        self.kappa = kappa

    def von_mises_fisher_kernel(self, X):
        """
        Compute the von Mises-Fisher (vMF) kernel matrix or scalar score based on cosine similarity.
        
        - If X is (N, D): computes (N, N) kernel matrix.
        - If X is (B, M, D): computes per-batch normalized trace inner product.

        Args:
            X (torch.Tensor): Input tensor. Shape (N, D) or (B, M, D).

        Returns:
            torch.Tensor: vMF kernel matrix (N, N) or scalar per batch (B,)
        """
        if X.ndim == 2:
            # Original logic for (N, D)
            norms = torch.norm(X, p=2, dim=1, keepdim=True)
            normalized_X = X / (norms + 1e-8)
            cosine_similarity_matrix = normalized_X @ normalized_X.T
            vmf_kernel_matrix = torch.exp(self.kappa * cosine_similarity_matrix)
            return vmf_kernel_matrix

        elif X.ndim == 3:                     # -------- (B, M, D) -----
            B, M, D = X.shape

            # Frobenius-norm  (B,1,1)
            frob = (X.pow(2).sum(dim=(1, 2), keepdim=True).sqrt() + 1e-8)
            X_norm = X / frob                 # 每个样本矩阵做 L2-归一化

            # trace(A_i^T A_j)  ==> einsum 不需要展开
            #   result shape: (B,B)
            sim = torch.einsum('bmd,cmd->bc', X_norm, X_norm)

            return torch.exp(self.kappa * sim)  # (B,B)
           # (B, B)
        else:
            raise ValueError("Input tensor must be of shape (N, D) or (B, M, D).")

    # def von_mises_fisher_kernel(self, X):
    #     """
    #     Compute the von Mises-Fisher (vMF) kernel matrix based on cosine similarity.

    #     Args:
    #         X (torch.Tensor): Input tensor of shape (N, D), where N is the number of data points 
    #                           and D is the dimensionality of each point.

    #     Returns:
    #         torch.Tensor: The vMF kernel matrix of shape (N, N).
    #     """
    #     # Normalize input vectors
    #     norms = torch.norm(X, p=2, dim=1, keepdim=True)  # Shape: (N, 1)
    #     normalized_X = X / (norms + 1e-8)  # Add epsilon for numerical stability

    #     # Compute cosine similarity
    #     cosine_similarity_matrix = normalized_X @ normalized_X.T  # Shape: (N, N)

    #     # Apply von Mises-Fisher kernel
    #     vmf_kernel_matrix = torch.exp(self.kappa * cosine_similarity_matrix)
    #     return vmf_kernel_matrix

    def joint_entropy(self, X, Y):
        """
        Compute the joint entropy between X and Y using the vMF kernel.

        Args:
            X (torch.Tensor): First input tensor of shape (N, D).
            Y (torch.Tensor): Second input tensor of shape (N, D).

        Returns:
            torch.Tensor: Joint entropy scalar value.
        """
        # Compute vMF kernel matrices
        K_X = self.von_mises_fisher_kernel(X)
        K_Y = self.von_mises_fisher_kernel(Y)

        # Joint kernel matrix
        K = K_X * K_Y

        # Normalize by trace
        K = K / torch.trace(K)

        # Frobenius norm squared and log
        frobenius_norm_squared = torch.norm(K, p='fro') ** 2
        joint_entropy = -torch.log(frobenius_norm_squared)
        # n = K.shape[0]
        # max_entropy = torch.log(torch.tensor(n))
        # joint_entropy = joint_entropy / max_entropy

        return joint_entropy

    def forward(self, X, Y):
        """
        Forward pass to compute joint entropy between X and Y.

        Args:
            X (torch.Tensor): First input tensor of shape (N, D).
            Y (torch.Tensor): Second input tensor of shape (N, D).

        Returns:
            torch.Tensor: Joint entropy scalar value.
        """
        return self.joint_entropy(X, Y)

    def forward(self, X, Y):
        """Calculate the contrastive graph loss between X and Y."""
        joint_entropy_XY = self.joint_entropy(X, Y)
        return joint_entropy_XY

class Alpha_Mutual_Information(nn.Module):
    def __init__(self, kappa=2.0):
        """
        Mutual information computation using the von Mises-Fisher (vMF) kernel.

        Args:
            kappa (float): Concentration parameter for the vMF kernel. Larger values emphasize angular similarity.
        """
        super(Alpha_Mutual_Information, self).__init__()
        self.kappa = kappa

    def von_mises_fisher_kernel(self, X):
        """
        Compute the von Mises-Fisher (vMF) kernel matrix based on cosine similarity.

        Args:
            X (torch.Tensor): Input tensor of shape (N, D), where N is the number of data points 
                              and D is the dimensionality of each point.

        Returns:
            torch.Tensor: The vMF kernel matrix of shape (N, N).
        """
        # Normalize input vectors
        norms = torch.norm(X, p=2, dim=1, keepdim=True)  # Shape: (N, 1)
        normalized_X = X / (norms + 1e-8)  # Add epsilon for numerical stability

        # Compute cosine similarity
        cosine_similarity_matrix = normalized_X @ normalized_X.T  # Shape: (N, N)

        # Apply von Mises-Fisher kernel
        vmf_kernel_matrix = torch.exp(self.kappa * cosine_similarity_matrix)
        return vmf_kernel_matrix

    def renyi_entropy(self, X):
        """
        Calculate the Rényi entropy for input X using the vMF kernel.

        Args:
            X (torch.Tensor): Input tensor of shape (N, D).

        Returns:
            torch.Tensor: Scalar Rényi entropy value.
        """
        K = self.von_mises_fisher_kernel(X)
        K = K / torch.trace(K)  # Normalize by trace
        frobenius_norm_squared = torch.norm(K, p='fro') ** 2
        renyi_entropy = -torch.log(frobenius_norm_squared)
        return renyi_entropy

    def joint_entropy(self, X, Y):
        """
        Calculate the joint entropy between X and Y using the vMF kernel.

        Args:
            X (torch.Tensor): First input tensor of shape (N, D).
            Y (torch.Tensor): Second input tensor of shape (N, D).

        Returns:
            torch.Tensor: Scalar joint entropy value.
        """
        K_X = self.von_mises_fisher_kernel(X)
        K_Y = self.von_mises_fisher_kernel(Y)
        K = K_X * K_Y  # Element-wise multiplication of kernel matrices
        K = K / torch.trace(K)  # Normalize by trace
        frobenius_norm_squared = torch.norm(K, p='fro') ** 2
        joint_entropy = -torch.log(frobenius_norm_squared)
        return joint_entropy

    def forward(self, X, Y):
        """
        Compute the mutual information between X and Y.

        Args:
            X (torch.Tensor): First input tensor of shape (N, D).
            Y (torch.Tensor): Second input tensor of shape (N, D).

        Returns:
            torch.Tensor: Scalar mutual information value.
        """
        # Compute entropy terms
        renyi_entropy_X = self.renyi_entropy(X)
        renyi_entropy_Y = self.renyi_entropy(Y)
        joint_entropy_XY = self.joint_entropy(X, Y)

        # Mutual information
        MI = renyi_entropy_X + renyi_entropy_Y - joint_entropy_XY
        return MI

class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kappa=2.0):
        """
        Maximum Mean Discrepancy (MMD) Loss supporting both RBF and vMF kernels.
        Args:
            kernel_type (str): Type of kernel to use ('rbf' or 'vmf').
            kappa (float): Concentration parameter for the vMF kernel (only used if kernel_type='vmf').
        """
        super(MMDLoss, self).__init__()
        self.kernel_type = kernel_type
        self.kappa = kappa

    @staticmethod
    def sigma_estimation(X):
        """
        Estimate sigma using the median distance from the upper triangular part of the distance matrix.
        Args:
            X (torch.Tensor): Input tensor of shape (N, D).
        Returns:
            float: Estimated sigma value.
        """
        # Compute pairwise distances
        pairwise_distances = torch.cdist(X, X, p=2)

        # Extract the upper triangular part of the matrix (excluding the diagonal)
        triu_indices = torch.triu_indices(pairwise_distances.size(0), pairwise_distances.size(0), offset=1)
        upper_triangular_distances = pairwise_distances[triu_indices[0], triu_indices[1]]

        # Compute the median of the distances
        med = torch.median(upper_triangular_distances).item()

        # Handle edge cases where the median might be invalid
        if med <= 0:
            med = torch.mean(upper_triangular_distances).item()
        if med < 1E-2:
            med = 1E-2

        return 0.25 * med 

    @staticmethod
    def rbf_kernel(X, Y, sigma):
        """
        Compute the RBF kernel matrix between two datasets.
        Args:
            X (torch.Tensor): Source tensor of shape (N, D).
            Y (torch.Tensor): Target tensor of shape (M, D).
            sigma (float): Bandwidth parameter for the RBF kernel.
        Returns:
            torch.Tensor: Kernel matrix of shape (N, M).
        """
        # Compute pairwise squared distances
        pairwise_distances = torch.cdist(X, Y, p=2) ** 2

        # Apply the RBF kernel with clamping for stability
        K = torch.clamp(torch.exp(-pairwise_distances / (2 * sigma ** 2)), min=1e-8)
        return K

    @staticmethod
    def vmf_kernel(X, Y, kappa):
        """
        Compute the von Mises-Fisher (vMF) kernel matrix between two datasets.
        Args:
            X (torch.Tensor): Source tensor of shape (N, D).
            Y (torch.Tensor): Target tensor of shape (M, D).
            kappa (float): Concentration parameter for the vMF kernel.
        Returns:
            torch.Tensor: Kernel matrix of shape (N, M).
        """
        # Normalize input vectors
        X_norm = X / (torch.norm(X, dim=1, keepdim=True) + 1e-8)
        Y_norm = Y / (torch.norm(Y, dim=1, keepdim=True) + 1e-8)

        # Compute cosine similarity and apply the vMF kernel
        cosine_similarity = torch.matmul(X_norm, Y_norm.T)
        K = torch.exp(kappa * cosine_similarity)
        return K

    def kernel_matrix(self, X, Y, sigma=None):
        """
        Compute the kernel matrix between two datasets based on the selected kernel type.
        Args:
            X (torch.Tensor): Source tensor of shape (N, D).
            Y (torch.Tensor): Target tensor of shape (M, D).
            sigma (float): Bandwidth parameter for the RBF kernel (only used if kernel_type='rbf').
        Returns:
            torch.Tensor: Kernel matrix of shape (N, M).
        """
        if self.kernel_type == 'rbf':
            if sigma is None:
                raise ValueError("Sigma must be provided for RBF kernel.")
            return self.rbf_kernel(X, Y, sigma)
        elif self.kernel_type == 'vmf':
            return self.vmf_kernel(X, Y, self.kappa)
        else:
            raise ValueError("Unsupported kernel type. Choose 'rbf' or 'vmf'.")

    def forward(self, source, target):
        """
        Compute the MMD loss between the source and target distributions.
        Args:
            source (torch.Tensor): Source data of shape (N, D).
            target (torch.Tensor): Target data of shape (M, D).
        Returns:
            torch.Tensor: Scalar MMD loss.
        """
        if self.kernel_type == 'rbf':
            sigma = self.sigma_estimation(torch.cat([source, target], dim=0))
        else:
            sigma = None  # Sigma is not needed for vMF kernel

        # Compute kernels
        K_ss = self.kernel_matrix(source, source, sigma)  # Source-source kernel
        K_tt = self.kernel_matrix(target, target, sigma)  # Target-target kernel
        K_st = self.kernel_matrix(source, target, sigma)  # Source-target kernel

        # Compute MMD loss
        loss = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
        return loss

class CSDivergence(nn.Module):
    def __init__(self, kernel_type='rbf', kappa=2.0):
        """
        CS Divergence implementation supporting both RBF and vMF kernels.
        Args:
            kernel_type (str): Type of kernel to use ('rbf' or 'vmf').
            kappa (float): Concentration parameter for the vMF kernel.
        """
        super(CSDivergence, self).__init__()
        self.kernel_type = kernel_type
        self.kappa = kappa

    @staticmethod
    def sigma_estimation(X):
        pairwise_distances = torch.cdist(X, X, p=2)
        triu_indices = torch.triu_indices(pairwise_distances.size(0), pairwise_distances.size(0), offset=1)
        upper_triangular_distances = pairwise_distances[triu_indices[0], triu_indices[1]]
        med = torch.median(upper_triangular_distances).item()
        return max(med, 1E-2)

    @staticmethod
    def rbf_kernel(X, Y, sigma):
        pairwise_distances = torch.cdist(X, Y, p=2) ** 2
        return torch.exp(-pairwise_distances / (2 * sigma ** 2))

    @staticmethod
    def vmf_kernel(X, Y, kappa):
        X_norm = X / (torch.norm(X, dim=1, keepdim=True) + 1e-8)
        Y_norm = Y / (torch.norm(Y, dim=1, keepdim=True) + 1e-8)
        cosine_similarity = torch.matmul(X_norm, Y_norm.T)
        return torch.exp(kappa * cosine_similarity)

    def kernel_matrix(self, X, Y, sigma=None):
        if self.kernel_type == 'rbf':
            return self.rbf_kernel(X, Y, sigma)
        elif self.kernel_type == 'vmf':
            return self.vmf_kernel(X, Y, self.kappa)
        else:
            raise ValueError("Unsupported kernel type. Choose 'rbf' or 'vmf'.")

    def forward(self, X, Y):
        """
        Compute the CS divergence between X and Y.
        Args:
            X (torch.Tensor): Input tensor of shape (M, D).
            Y (torch.Tensor): Input tensor of shape (M, D).
        Returns:
            torch.Tensor: Scalar value representing the CS divergence.
        """
        if self.kernel_type == 'rbf':
            sigma = self.sigma_estimation(torch.cat([X, Y], dim=0))
        else:
            sigma = None

        # Compute kernel matrices
        K_XX = self.kernel_matrix(X, X, sigma)  # Self-similarity for X
        K_YY = self.kernel_matrix(Y, Y, sigma)  # Self-similarity for Y
        K_XY = self.kernel_matrix(X, Y, sigma)  # Cross-similarity between X and Y

        # Compute CS divergence using direct mean
        self_term_X = K_XX.mean()
        self_term_Y = K_YY.mean()
        cross_term = K_XY.mean()

        return -2 * torch.log(cross_term) + torch.log(self_term_X) + torch.log(self_term_Y)

class CSQMI(nn.Module):
    def __init__(self, kernel_type='rbf', kappa=2.0):
        """
        CS QMI implementation supporting RBF and vMF kernels.
        Args:
            kernel_type (str): Type of kernel to use ('rbf' or 'vmf').
            kappa (float): Concentration parameter for the vMF kernel.
        """
        super(CSQMI, self).__init__()
        self.kernel_type = kernel_type
        self.kappa = kappa

    @staticmethod
    def sigma_estimation(X):
        pairwise_distances = torch.cdist(X, X, p=2)
        triu_indices = torch.triu_indices(pairwise_distances.size(0), pairwise_distances.size(0), offset=1)
        upper_triangular_distances = pairwise_distances[triu_indices[0], triu_indices[1]]
        med = torch.median(upper_triangular_distances).item()
        return max(med, 1E-2)

    @staticmethod
    def rbf_kernel(X, Y, sigma):
        pairwise_distances = torch.cdist(X, Y, p=2) ** 2
        return torch.exp(-pairwise_distances / (2 * sigma ** 2))

    @staticmethod
    def vmf_kernel(X, Y, kappa):
        X_norm = X / (torch.norm(X, dim=1, keepdim=True) + 1e-8)
        Y_norm = Y / (torch.norm(Y, dim=1, keepdim=True) + 1e-8)
        cosine_similarity = torch.matmul(X_norm, Y_norm.T)
        return torch.exp(kappa * cosine_similarity)

    def kernel_matrix(self, X, Y, sigma=None):
        if self.kernel_type == 'rbf':
            return self.rbf_kernel(X, Y, sigma)
        elif self.kernel_type == 'vmf':
            return self.vmf_kernel(X, Y, self.kappa)
        else:
            raise ValueError("Unsupported kernel type. Choose 'rbf' or 'vmf'.")

    def compute_cs_qmi(self, X, Y):
        """
        Compute the CS QMI between X and Y.
        Args:
            X (torch.Tensor): Input tensor of shape (N, D_x).
            Y (torch.Tensor): Input tensor of shape (N, D_y).
        Returns:
            torch.Tensor: Scalar value representing the CS QMI.
        """
        N = X.shape[0]

        # Estimate sigma for RBF kernel if necessary
        if self.kernel_type == 'rbf':
            sigma_x = 10 * self.sigma_estimation(X)
            sigma_y = 10 * self.sigma_estimation(Y)
            Kx = self.rbf_kernel(X, X, sigma_x)
            Ky = self.rbf_kernel(Y, Y, sigma_y)
        elif self.kernel_type == 'vmf':
            Kx = self.vmf_kernel(X, X, self.kappa)
            Ky = self.vmf_kernel(Y, Y, self.kappa)
        else:
            raise ValueError("Unsupported kernel type. Choose 'rbf' or 'vmf'.")

        # First term
        self_term1 = torch.trace(Kx @ Ky.T) / (N ** 2)

        # Second term
        self_term2 = (torch.sum(Kx) * torch.sum(Ky)) / (N ** 4)

        # Third term
        term_a = torch.ones(1, N).to(X.device)
        term_b = torch.ones(N, 1).to(X.device)
        cross_term = (term_a @ Kx.T @ Ky @ term_b) / (N ** 3)

        # Compute CS QMI
        cs_qmi = -2 * torch.log2(cross_term) + torch.log2(self_term1) + torch.log2(self_term2)

        return cs_qmi

    def forward(self, X, Y):
        return self.compute_cs_qmi(X, Y)


def cs_qmi_normalized(X, Y, model):
    """
    Compute the normalized CS QMI between X and Y.
    Args:
        X (torch.Tensor): Input tensor of shape (N, D_x).
        Y (torch.Tensor): Input tensor of shape (N, D_y).
        model (CSQMI): An instance of the CSQMI class.
    Returns:
        torch.Tensor: Normalized CS QMI.
    """
    QMI = model.compute_cs_qmi(X, Y)
    var1 = torch.sqrt(model.compute_cs_qmi(X, X))
    var2 = torch.sqrt(model.compute_cs_qmi(Y, Y))
    return QMI / (var1 * var2)