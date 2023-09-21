import torch


def compute_centroid(data, labels, class_label):
    """Computes the centroid for the specified class."""
    class_data = data[labels == class_label]
    centroid = torch.mean(class_data, axis=0)
    return centroid


def compute_separabilities(X, Y, S):
    """Computes separability for a given dataset X, Y considering class pairs with at least one class in S."""
    # Convert numpy arrays to PyTorch tensors
    X, Y = torch.tensor(X), torch.tensor(Y)

    # Determine all unique classes and filter for pairs (c1, c2) with at least one class in S
    unique_classes = torch.unique(Y)
    relevant_pairs = [(c1, c2) for c1 in unique_classes for c2 in unique_classes if c1 != c2 and (c1 in S or c2 in S)]

    separabilities = []

    for c1, c2 in relevant_pairs:
        centroid_c1 = compute_centroid(X, Y, c1)
        centroid_c2 = compute_centroid(X, Y, c2)

        d_inter = torch.norm(centroid_c1 - centroid_c2)

        d_intra_c1 = torch.mean(torch.norm(X[Y == c1] - centroid_c1, dim=1))
        d_intra_c2 = torch.mean(torch.norm(X[Y == c2] - centroid_c2, dim=1))

        separability = d_inter / ((d_intra_c1 + d_intra_c2) / 2)
        separabilities.append((c1.item(), c2.item(), separability.item()))

    return separabilities
