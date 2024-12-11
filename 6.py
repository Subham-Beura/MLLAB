# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
iris = load_iris()
X = iris.data
y = iris.target

# %%
X

# %%
y

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
def gain_ratio(feature_column, target):
    total_entropy = entropy(target)
    values, counts = np.unique(feature_column, return_counts=True)
    feature_entropy = 0
    split_info = 0

    for i in range(len(values)):
        value_entropy = entropy(target[feature_column == values[i]])
        weight = counts[i] / np.sum(counts)
        feature_entropy += weight * value_entropy
        split_info -= weight * np.log2(weight) if weight != 0 else 0

    info_gain = total_entropy - feature_entropy
    gain_ratio = info_gain / split_info if split_info != 0 else 0

    return gain_ratio

# %%
def entropy(y):
    elements, counts = np.unique(y, return_counts=True)
    ent = np.sum([-counts[i]/np.sum(counts) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return ent

# %%
clf_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)
print(f"Gini Index Accuracy: {accuracy_score(y_test, y_pred_gini):.2f}")

# %%
clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_entropy.fit(X_train, y_train)
y_pred_entropy = clf_entropy.predict(X_test)
print(f"Entropy Accuracy: {accuracy_score(y_test, y_pred_entropy):.2f}")

# %%
for i in range(X_train.shape[1]):
    ratio = gain_ratio(X_train[:, i], y_train)
    print(f"Gain Ratio for feature {iris.feature_names[i]}: {ratio:.4f}")


# %%
clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
clf.fit(X_train, y_train)

# %%
n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold
values = clf.tree_.value

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, 0)]
while len(stack) > 0:
    node_id, depth = stack.pop()
    node_depth[node_id] = depth
    is_split_node = children_left[node_id] != children_right[node_id]
    if is_split_node:
        stack.append((children_left[node_id], depth + 1))
        stack.append((children_right[node_id], depth + 1))
    else:
        is_leaves[node_id] = True
print(
    "The binary tree structure has {n} nodes and has "
    "the following tree structure:\n".format(n=n_nodes)
)
for i in range(n_nodes):
    if is_leaves[i]:
        print(
            "{space}node={node} is a leaf node with value={value}.".format(
                space=node_depth[i] * "\t", node=i, value=np.around(values[i], 3)
            )
        )
    else:
        print(
            "{space}node={node} is a split node with value={value}: "
            "go to node {left} if X[:, {feature}] <= {threshold} "
            "else to node {right}.".format(
                space=node_depth[i] * "\t",
                node=i,
                left=children_left[i],
                feature=feature[i],
                threshold=threshold[i],
                right=children_right[i],
                value=np.around(values[i], 3),
            )
        )

# %%
tree.plot_tree(clf_gini, proportion=True)
plt.show()

# %%
tree.plot_tree(clf_entropy, proportion=True)
plt.show()

# %%
tree.plot_tree(clf, proportion=True)
plt.show()


