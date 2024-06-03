import torch

# label_frequency
def calc_label_freq(dataset):
    all_num = len(dataset)
    positive_num = len([sample for sample in dataset if sample['label'] == 1])
    negative_num = len([sample for sample in dataset if sample['label'] == 0])
    positive_frequency = positive_num / all_num
    negative_frequency = negative_num / all_num
    return positive_num, negative_num, positive_frequency, negative_frequency

# cluster method
def prototyping(X, k, max_iters=1000, tol=1e-4, alpha=0.1):
    prototypes = X[torch.randperm(X.size(0))[:k]]
    for i in range(max_iters):
        distances = torch.cdist(X, prototypes)
        labels = torch.argmin(distances, dim=1)
        prototypes_old = prototypes.clone()
        for j in range(k):
            cluster_points = X[labels == j]
            for m in range(len(cluster_points)):
                prototypes[j] = (1 - alpha) * prototypes[j] + alpha * cluster_points[m]

        if torch.norm(prototypes - prototypes_old) < tol:
            print(f"Converged after {i} iterations")
            break

    return labels, prototypes


def one_hot(labels, num_classes):
    one_hot_labels = torch.zeros(labels.size(0), num_classes)
    one_hot_labels[torch.arange(labels.size(0)), labels] = 1
    return one_hot_labels.float()


def sample_step(prototypes, positive_frequency, negative_frequency, h_batch, model, sample_ratio):
    distances = torch.cdist(h_batch, prototypes)
    inverse_distances = 1.0 / (distances + 1e-10)
    inverse_distances = torch.softmax(inverse_distances, dim=1)

    def gini_coefficient(p):
        return 1 - torch.sum(p * p, dim=1)

    distance_coefficient = 1.0 / (gini_coefficient(inverse_distances) + 1e-10)

    y_true = model.prepare_labels(h_batch[model.label_key], model.label_tokenizer).squeeze()
    frequeny_tensor = torch.where(y_true > 0, positive_frequency, negative_frequency)
    distance_coefficient /= frequeny_tensor
    sample_prob = torch.softmax(distance_coefficient, dim=0)
    assert sample_ratio <= 1.0
    sample_num = round(sample_ratio * h_batch.size(0))
    sample_index = torch.multinomial(sample_prob, sample_num, replacement=True)
    return sample_index