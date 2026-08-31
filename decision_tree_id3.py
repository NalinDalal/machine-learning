"""
A simple implementation of the ID3 Decision Tree algorithm (Quinlan, 1986)
for categorical data, with entropy and information gain.

This is a minimal, educational version for demonstration purposes.
"""
import math
from collections import Counter, defaultdict

def entropy(class_labels):
    """

    :param class_labels: 

    """
    total = len(class_labels)
    counts = Counter(class_labels)
    return -sum((count/total) * math.log2(count/total) for count in counts.values() if count)

def information_gain(data, labels, feature_index):
    """

    :param data: param labels:
    :param feature_index: param labels:
    :param labels: 

    """
    total_entropy = entropy(labels)
    feature_values = [row[feature_index] for row in data]
    value_subsets = defaultdict(list)
    for i, value in enumerate(feature_values):
        value_subsets[value].append(i)
    weighted_entropy = 0.0
    for indices in value_subsets.values():
        subset_labels = [labels[i] for i in indices]
        weighted_entropy += (len(indices)/len(labels)) * entropy(subset_labels)
    return total_entropy - weighted_entropy

def majority_class(labels):
    """

    :param labels: 

    """
    return Counter(labels).most_common(1)[0][0]

def id3(data, labels, feature_names):
    """

    :param data: param labels:
    :param feature_names: param labels:
    :param labels: 

    """
    # If all labels are the same, return that label
    if len(set(labels)) == 1:
        return labels[0]
    # If no features left, return majority class
    if not feature_names:
        return majority_class(labels)
    # Find best feature to split
    gains = [information_gain(data, labels, i) for i in range(len(feature_names))]
    best_idx = gains.index(max(gains))
    best_feature = feature_names[best_idx]
    tree = {best_feature: {}}
    feature_values = set(row[best_idx] for row in data)
    for value in feature_values:
        # Partition data
        sub_data = [row[:best_idx] + row[best_idx+1:] for row in data if row[best_idx] == value]
        sub_labels = [labels[i] for i, row in enumerate(data) if row[best_idx] == value]
        sub_features = feature_names[:best_idx] + feature_names[best_idx+1:]
        if not sub_data:
            tree[best_feature][value] = majority_class(labels)
        else:
            tree[best_feature][value] = id3(sub_data, sub_labels, sub_features)
    return tree

def print_tree(tree, indent=""):
    """

    :param tree: param indent:  (Default value = "")
    :param indent: Default value = "")

    """
    if not isinstance(tree, dict):
        print(indent + "->", tree)
        return
    for feature, branches in tree.items():
        for value, subtree in branches.items():
            print(f"{indent}if {feature} == {value}:")
            print_tree(subtree, indent + "    ")

if __name__ == "__main__":
    # Example: Play Tennis dataset (simplified)
    feature_names = ["Outlook", "Humidity"]
    data = [
        ["Sunny", "High"],
        ["Sunny", "Normal"],
        ["Overcast", "High"],
        ["Rain", "High"],
        ["Rain", "Normal"],
        ["Overcast", "Normal"],
    ]
    labels = ["No", "Yes", "Yes", "No", "Yes", "Yes"]
    tree = id3(data, labels, feature_names)
    print("Decision Tree:")
    print_tree(tree)
