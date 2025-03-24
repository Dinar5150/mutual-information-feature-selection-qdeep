# Copyright 2019 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import os

import matplotlib
matplotlib.use("agg")  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd

# D-Wave Ocean tools used for BQM construction
import dimod

from qdeepsdk import QDeepHybridSolver

def prob(dataset):
    """Joint probability distribution P(X) for the given data."""
    num_rows, num_columns = dataset.shape
    bin_boundaries = [np.hstack((np.unique(dataset[:, ci]), np.inf)) for ci in range(num_columns)]
    p, _ = np.histogramdd(dataset, bins=bin_boundaries)
    return p / np.sum(p)

def shannon_entropy(p):
    """Shannon entropy H(X) is the negative sum of P(X)log(P(X))."""
    p = p.flatten()
    return -sum(pi * np.log2(pi) for pi in p if pi)

def conditional_shannon_entropy(p, *conditional_indices):
    """Conditional Shannon entropy H(X|Y) = H(X,Y) - H(Y)."""
    assert all(ci < p.ndim for ci in conditional_indices)
    axis = tuple(i for i in np.arange(len(p.shape)) if i not in conditional_indices)
    return shannon_entropy(p) - shannon_entropy(np.sum(p, axis=axis))

def mutual_information(p, j):
    """Mutual information between variables X and variable Y:
       I(X; Y) = H(X) - H(X|Y)."""
    return (shannon_entropy(np.sum(p, axis=j))
            - conditional_shannon_entropy(p, j))

def conditional_mutual_information(p, j, *conditional_indices):
    """Mutual information between variables X and variable Y conditional on variable Z:
       I(X;Y|Z) = H(X|Z) - H(X|Y,Z)."""
    marginal_conditional_indices = [i - 1 if i > j else i for i in conditional_indices]
    return (conditional_shannon_entropy(np.sum(p, axis=j), *marginal_conditional_indices)
            - conditional_shannon_entropy(p, j, *conditional_indices))

def maximum_energy_delta(bqm):
    """Compute a conservative bound on the maximum energy change when flipping a single variable."""
    return max(abs(bqm.get_linear(i)) +
               sum(abs(bqm.get_quadratic(i, j)) for j, _ in bqm.iter_neighborhood(i))
               for i in bqm.variables)

def mutual_information_bqm(dataset, features, target):
    """
    Build a BQM that maximizes the mutual information (MI) between the target variable
    and each feature (and pairs of features for interactions).
    """
    variables = ((feature, -mutual_information(prob(dataset[[target, feature]].values), 1))
                 for feature in features)
    interactions = ((f0, f1, -conditional_mutual_information(prob(dataset[[target, f0, f1]].values), 1, 2))
                    for f0, f1 in itertools.permutations(features, 2))
    return dimod.BinaryQuadraticModel(variables, interactions, 0, dimod.BINARY)

def add_combination_penalty(bqm, k, penalty):
    """Return a new BQM with an additional combination penalty (biasing towards k-combinations)."""
    kbqm = dimod.generators.combinations(bqm.variables, k, strength=penalty)
    kbqm.update(bqm)
    return kbqm

def mutual_information_feature_selection(dataset, features, target):
    """
    Run the MI Feature Selection algorithm using QDeepHybridSolver.
    For each number of features k, add a combination penalty and solve the resulting QUBO.
    Returns a matrix (k x num_features) showing selection (0 or 1) for each feature.
    """
    solver = QDeepHybridSolver()
    solver.token = "your-auth-token-here"  # Replace with your valid auth token

    selected_features = np.zeros((len(features), len(features)))
    bqm = mutual_information_bqm(dataset, features, target)
    penalty = maximum_energy_delta(bqm)

    # For each number of features k, add a penalty and solve
    for k in range(1, len(features) + 1):
        kbqm = add_combination_penalty(bqm, k, penalty)
        # Convert the BQM to a QUBO matrix.
        # Get an ordering for the variables (feature names)
        ordering = list(kbqm.variables)
        n = len(ordering)
        Q_dict, offset = kbqm.to_qubo()
        Q = np.zeros((n, n))
        for (i, j), val in Q_dict.items():
            # Map variable names to indices in our ordering.
            idx_i = ordering.index(i)
            idx_j = ordering.index(j)
            Q[idx_i, idx_j] = val

        # Solve the QUBO using the QDeepHybridSolver API.
        result = solver.solve(Q)
        configuration = result["configuration"]  # Binary vector of length n
        # Create a sample dict mapping each variable (feature) to its binary decision.
        sample = {var: configuration[i] for i, var in enumerate(ordering)}
        for fi, f in enumerate(features):
            selected_features[k - 1, fi] = sample.get(f, 0)
    return selected_features

def run_demo(dataset, target):
    """Compute MI Feature Selection (MIFS) for each k and visualize the results."""
    # Rank MI between target and every other feature.
    scores = {feature: mutual_information(prob(dataset[[target, feature]].values), 0)
              for feature in set(dataset.columns) - {target}}
    labels, values = zip(*sorted(scores.items(), key=lambda pair: pair[1], reverse=True))

    # Plot MI scores
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Mutual Information")
    ax1.set_ylabel("MI between '{}' and Feature".format(target))
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    plt.bar(np.arange(len(labels)), values)

    # For this demo, select only the top-scoring features (e.g. top 8)
    keep = 8
    sorted_scores = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    # Build a new dataset with only the top features plus the target.
    dataset = dataset[[col for col, _ in sorted_scores[0:keep]] + [target]]
    features = sorted(list(set(dataset.columns) - {target}))
    selected_features = mutual_information_feature_selection(dataset, features, target)

    # Plot the best feature selection per number of selected features.
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Best Feature Selection")
    ax2.set_ylabel("Number of Selected Features")
    ax2.set_xticks(np.arange(len(features)))
    ax2.set_xticklabels(features, rotation=90)
    ax2.set_yticks(np.arange(len(features)))
    ax2.set_yticklabels(np.arange(1, len(features) + 1))
    ax2.set_xticks(np.arange(-0.5, len(features)), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(features)), minor=True)
    ax2.grid(which="minor", color="black")
    ax2.imshow(selected_features, cmap=colors.ListedColormap(["white", "red"]))

if __name__ == "__main__":
    # Load the Titanic dataset from the 'data' directory relative to this script.
    demo_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(demo_path, "data", "formatted_titanic.csv")
    dataset = pd.read_csv(data_path)
    run_demo(dataset, "survived")
    plots_path = os.path.join(demo_path, "plots.png")
    plt.savefig(plots_path, bbox_inches="tight")
    print("Your plots are saved to {}".format(plots_path))
