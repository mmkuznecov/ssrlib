import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List
import numpy as np


class TripletLoss(nn.Module):
    """
    Triplet Loss implementation with various distance metrics and mining strategies.
    
    The triplet loss encourages embeddings where:
    - distance(anchor, positive) + margin < distance(anchor, negative)
    
    Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    
    Args:
        margin: Minimum margin between positive and negative pairs
        distance: Distance metric ('euclidean', 'cosine', 'squared_euclidean')
        reduction: Loss reduction ('mean', 'sum', 'none')
        normalize: Whether to L2 normalize embeddings
    """
    
    def __init__(
        self, 
        margin: float = 1.0,
        distance: str = 'euclidean',
        reduction: str = 'mean',
        normalize: bool = False
    ):
        super().__init__()
        self.margin = margin
        self.distance = distance
        self.reduction = reduction
        self.normalize = normalize
        
        assert distance in ['euclidean', 'cosine', 'squared_euclidean'], \
            f"Unsupported distance: {distance}"
        assert reduction in ['mean', 'sum', 'none'], \
            f"Unsupported reduction: {reduction}"
    
    def compute_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between two tensors.
        
        Args:
            x1, x2: Tensors of shape (N, D)
            
        Returns:
            Distance tensor of shape (N,)
        """
        if self.normalize:
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p=2, dim=1)
        
        if self.distance == 'euclidean':
            return torch.norm(x1 - x2, p=2, dim=1)
        elif self.distance == 'squared_euclidean':
            return torch.sum((x1 - x2) ** 2, dim=1)
        elif self.distance == 'cosine':
            if not self.normalize:
                x1 = F.normalize(x1, p=2, dim=1)
                x2 = F.normalize(x2, p=2, dim=1)
            return 1 - torch.sum(x1 * x2, dim=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")
    
    def forward(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for triplet loss.
        
        Args:
            anchor: Anchor embeddings (N, D)
            positive: Positive embeddings (N, D)  
            negative: Negative embeddings (N, D)
            
        Returns:
            Triplet loss
        """
        # Compute distances
        pos_dist = self.compute_distance(anchor, positive)
        neg_dist = self.compute_distance(anchor, negative)
        
        # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class OnlineTripletLoss(nn.Module):
    """
    Online triplet loss with hard/semi-hard negative mining.
    
    Mines triplets within each batch based on labels, avoiding the need
    to pre-compute triplets.
    
    Args:
        margin: Triplet margin
        mining_strategy: 'hard', 'semi_hard', 'all'
        distance: Distance metric
        normalize: Whether to normalize embeddings
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        mining_strategy: str = 'hard',
        distance: str = 'euclidean',
        normalize: bool = True
    ):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        self.distance = distance
        self.normalize = normalize
        
        assert mining_strategy in ['hard', 'semi_hard', 'all'], \
            f"Unsupported mining strategy: {mining_strategy}"
    
    def compute_pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between all embeddings.
        
        Args:
            embeddings: Tensor of shape (N, D)
            
        Returns:
            Distance matrix of shape (N, N)
        """
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        if self.distance == 'euclidean':
            # Efficient computation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            dot_product = torch.mm(embeddings, embeddings.t())
            squared_norms = torch.diag(dot_product)
            distances = squared_norms.unsqueeze(0) - 2.0 * dot_product + squared_norms.unsqueeze(1)
            distances = torch.clamp(distances, min=0.0)
            return torch.sqrt(distances)
        
        elif self.distance == 'squared_euclidean':
            dot_product = torch.mm(embeddings, embeddings.t())
            squared_norms = torch.diag(dot_product)
            distances = squared_norms.unsqueeze(0) - 2.0 * dot_product + squared_norms.unsqueeze(1)
            return torch.clamp(distances, min=0.0)
        
        elif self.distance == 'cosine':
            if not self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            similarity = torch.mm(embeddings, embeddings.t())
            return 1 - similarity
        
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")
    
    def get_triplet_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Get mask for valid triplets (a, p, n) where a != p != n and label[a] == label[p] != label[n].
        
        Args:
            labels: Label tensor of shape (N,)
            
        Returns:
            Boolean mask of shape (N, N, N)
        """
        # Check that i, j, k are distinct
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)
        
        distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k
        
        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)
        
        valid_labels = i_equal_j & ~i_equal_k
        
        return distinct_indices & valid_labels
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with online triplet mining.
        
        Args:
            embeddings: Embedding tensor (N, D)
            labels: Label tensor (N,)
            
        Returns:
            Triplet loss
        """
        # Compute pairwise distances
        pairwise_dist = self.compute_pairwise_distances(embeddings)
        
        # Get valid triplet mask
        triplet_mask = self.get_triplet_mask(labels)
        
        if self.mining_strategy == 'all':
            # Use all valid triplets
            anchor_positive_dist = pairwise_dist.unsqueeze(2)
            anchor_negative_dist = pairwise_dist.unsqueeze(1)
            
            triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
            triplet_loss = torch.clamp(triplet_loss, min=0.0)
            
            # Apply mask and compute mean
            triplet_loss = triplet_loss * triplet_mask.float()
            num_positive_triplets = triplet_mask.sum().float()
            
            if num_positive_triplets > 0:
                triplet_loss = triplet_loss.sum() / num_positive_triplets
            else:
                triplet_loss = torch.tensor(0.0, device=embeddings.device)
                
        elif self.mining_strategy == 'hard':
            # Hard negative mining: for each anchor-positive pair, find hardest negative
            labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
            labels_not_equal = ~labels_equal
            
            # Mask out same-sample distances
            masked_dist = pairwise_dist.clone()
            masked_dist = masked_dist + labels_equal.float() * 1e6  # Large number for same labels
            
            triplet_losses = []
            
            for i in range(len(labels)):
                # Find all positives for anchor i
                positive_mask = labels_equal[i] & (torch.arange(len(labels), device=labels.device) != i)
                if not positive_mask.any():
                    continue
                    
                # Find all negatives for anchor i  
                negative_mask = labels_not_equal[i]
                if not negative_mask.any():
                    continue
                
                # Get distances from anchor i to all positives and negatives
                pos_distances = pairwise_dist[i][positive_mask]
                neg_distances = pairwise_dist[i][negative_mask]
                
                # For each positive, find hardest negative
                hardest_negative_dist = neg_distances.min()
                
                # Compute loss for all anchor-positive pairs with hardest negative
                losses = torch.clamp(pos_distances - hardest_negative_dist + self.margin, min=0.0)
                triplet_losses.append(losses)
            
            if triplet_losses:
                triplet_loss = torch.cat(triplet_losses).mean()
            else:
                triplet_loss = torch.tensor(0.0, device=embeddings.device)
                
        elif self.mining_strategy == 'semi_hard':
            # Semi-hard negative mining: negatives that violate margin but are not hardest
            labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
            labels_not_equal = ~labels_equal
            
            triplet_losses = []
            
            for i in range(len(labels)):
                positive_mask = labels_equal[i] & (torch.arange(len(labels), device=labels.device) != i)
                if not positive_mask.any():
                    continue
                    
                negative_mask = labels_not_equal[i]
                if not negative_mask.any():
                    continue
                
                pos_distances = pairwise_dist[i][positive_mask]
                neg_distances = pairwise_dist[i][negative_mask]
                
                # For each positive distance, find semi-hard negatives
                for pos_dist in pos_distances:
                    # Semi-hard: pos_dist < neg_dist < pos_dist + margin
                    semi_hard_mask = (neg_distances > pos_dist) & (neg_distances < pos_dist + self.margin)
                    
                    if semi_hard_mask.any():
                        # Use hardest semi-hard negative
                        semi_hard_neg_dist = neg_distances[semi_hard_mask].min()
                    else:
                        # Fallback to hardest negative
                        semi_hard_neg_dist = neg_distances.min()
                    
                    loss = torch.clamp(pos_dist - semi_hard_neg_dist + self.margin, min=0.0)
                    triplet_losses.append(loss)
            
            if triplet_losses:
                triplet_loss = torch.stack(triplet_losses).mean()
            else:
                triplet_loss = torch.tensor(0.0, device=embeddings.device)
        
        return triplet_loss


class TripletDataset:
    """
    Helper class to generate triplets from a dataset with labels.
    """
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = embeddings
        self.labels = labels
        self.label_to_indices = {}
        
        # Group indices by label
        for idx, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
    
    def sample_triplet(self) -> Tuple[int, int, int]:
        """
        Sample a random triplet (anchor, positive, negative).
        
        Returns:
            Tuple of (anchor_idx, positive_idx, negative_idx)
        """
        # Sample anchor label and index
        anchor_label = np.random.choice(list(self.label_to_indices.keys()))
        anchor_idx = np.random.choice(self.label_to_indices[anchor_label])
        
        # Sample positive (same label, different index)
        positive_candidates = [idx for idx in self.label_to_indices[anchor_label] if idx != anchor_idx]
        if not positive_candidates:
            # If only one sample for this label, sample again
            return self.sample_triplet()
        positive_idx = np.random.choice(positive_candidates)
        
        # Sample negative (different label)
        negative_labels = [label for label in self.label_to_indices.keys() if label != anchor_label]
        if not negative_labels:
            raise ValueError("Need at least 2 different labels for triplet sampling")
        negative_label = np.random.choice(negative_labels)
        negative_idx = np.random.choice(self.label_to_indices[negative_label])
        
        return anchor_idx, positive_idx, negative_idx
    
    def generate_triplets(self, num_triplets: int) -> List[Tuple[int, int, int]]:
        """Generate multiple triplets."""
        return [self.sample_triplet() for _ in range(num_triplets)]


# Example usage and testing
def example_usage():
    """Demonstrate different triplet loss variants."""
    
    # Generate sample data
    batch_size = 32
    embedding_dim = 128
    num_classes = 8
    
    # Create embeddings and labels
    embeddings = torch.randn(batch_size, embedding_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print("=== Basic Triplet Loss ===")
    
    # Basic triplet loss with explicit triplets
    triplet_loss = TripletLoss(margin=1.0, distance='euclidean')
    
    # Manually create triplets (anchor, positive, negative)
    anchor = embeddings[:10]
    positive = embeddings[10:20] 
    negative = embeddings[20:30]
    
    loss = triplet_loss(anchor, positive, negative)
    print(f"Basic triplet loss: {loss.item():.4f}")
    
    print("\n=== Online Triplet Loss ===")
    
    # Online triplet loss with different mining strategies
    strategies = ['all', 'hard', 'semi_hard']
    
    for strategy in strategies:
        online_loss = OnlineTripletLoss(
            margin=1.0, 
            mining_strategy=strategy,
            distance='euclidean',
            normalize=True
        )
        loss = online_loss(embeddings, labels)
        print(f"Online triplet loss ({strategy}): {loss.item():.4f}")
    
    print("\n=== Different Distance Metrics ===")
    
    distances = ['euclidean', 'squared_euclidean', 'cosine']
    
    for dist in distances:
        loss_fn = TripletLoss(margin=1.0, distance=dist, normalize=(dist=='cosine'))
        loss = loss_fn(anchor, positive, negative)
        print(f"Triplet loss ({dist}): {loss.item():.4f}")
    
    print("\n=== Triplet Mining Example ===")
    
    # Example of offline triplet generation
    embeddings_np = embeddings.detach().numpy()
    labels_np = labels.detach().numpy()
    
    dataset = TripletDataset(embeddings_np, labels_np)
    triplets = dataset.generate_triplets(5)
    
    print("Sample triplets (anchor_idx, positive_idx, negative_idx):")
    for i, (a, p, n) in enumerate(triplets):
        print(f"  Triplet {i+1}: ({a}, {p}, {n}) | Labels: ({labels_np[a]}, {labels_np[p]}, {labels_np[n]})")


def compute_triplet_accuracy(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> float:
    """
    Compute triplet accuracy: fraction of triplets satisfying the margin constraint.
    
    Args:
        embeddings: Embedding tensor (N, D)
        labels: Label tensor (N,)
        margin: Triplet margin
        
    Returns:
        Accuracy as float
    """
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    
    correct_triplets = 0
    total_triplets = 0
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j or labels[i] != labels[j]:
                continue
                
            for k in range(len(labels)):
                if k == i or k == j or labels[i] == labels[k]:
                    continue
                
                # Check if d(i,j) + margin < d(i,k)
                if dist_matrix[i, j] + margin < dist_matrix[i, k]:
                    correct_triplets += 1
                total_triplets += 1
    
    return correct_triplets / total_triplets if total_triplets > 0 else 0.0


if __name__ == "__main__":
    example_usage()
    
    # Test accuracy computation
    print("\n=== Triplet Accuracy Test ===")
    embeddings = torch.randn(20, 64)
    labels = torch.randint(0, 4, (20,))
    
    accuracy = compute_triplet_accuracy(embeddings, labels, margin=1.0)
    print(f"Triplet accuracy: {accuracy:.4f}")