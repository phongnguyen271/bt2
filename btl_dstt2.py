# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("D:\Study\Linear Algebra\Materials\dataset2.csv")

# Print the dataset
print(df, '\n')

# Separte the features and the target variable
X = df.iloc[:, :-1].values # Features
y = df.iloc[:, -1].values # Target variable

# Print the features and the target variable
print(X,'\n')
print(y,'\n')

# Find the number of features
num_features = X.shape[1]

# Find the number of classes
class_labels = np.unique(y)

# Plot the dataset
for cl in class_labels:
    plt.scatter(X[y == cl][:, 0], X[y == cl][:, 1], label=cl)
plt.xlabel("Muscle fatigue")
plt.ylabel("Fouls recieved")
plt.legend()
plt.title("Scatter plot of the dataset")
plt.show()

# Compute the mean vector of each class
mean_vectors = []
for cl in class_labels:
    mean_vectors.append(np.mean(X[y == cl], axis=0))

# Print the mean vectors
mean_vectors_rounded = [np.round(mv, 2) for mv in mean_vectors]
print("Mean vectors:")
for i, mean_vec in enumerate(mean_vectors_rounded):
    print(f"Class {class_labels[i]}: {mean_vec}\n")

# Compute the within-class scatter matrix
S_W = np.zeros((num_features, num_features))
for cl, mv in zip(class_labels, mean_vectors):
    class_scatter = np.zeros((num_features, num_features))
    for row in X[y == cl]:
        row, mv = row.reshape(num_features, 1), mv.reshape(num_features, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter

# Print the within-class scatter matrix
print("Within-class scatter matrix:\n", np.round(S_W, 2))

# Compute the between-class scatter matrix
overall_mean = np.mean(X, axis=0)
S_B = np.zeros((num_features, num_features))
for cl, mv in zip(class_labels, mean_vectors):
    n = X[y == cl].shape[0]
    mv = mv.reshape(num_features, 1)
    overall_mean = overall_mean.reshape(num_features, 1)
    S_B += n * (mv - overall_mean).dot((mv - overall_mean).T)

# Print the between-class scatter matrix
print("Between-class scatter matrix:\n", np.round(S_B, 2))

# Compute the eigenvalues and eigenvectors of S_W^-1 S_B
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Check the calculation
for i in range(len(eig_vals)):
    eigv = eig_vecs[:, i].reshape(num_features, 1)
    np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv),
                                         eig_vals[i] * eigv,
                                         decimal=6, err_msg='', verbose=True)
print("The calculation is correct.")

# Print the eigenvalues and eigenvectors in decreasing order
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
print("Eigenvalues and eigenvectors in decreasing order:\n")
for i in range(len(eig_pairs)):
    print(f"Eigenvalue {i+1}: {np.round(eig_pairs[i][0], 2)}")
    print(f"Eigenvector {i+1}: {np.round(eig_pairs[i][1], 2)}\n")

# Select the top linear discriminant (1D projection)
W = eig_pairs[0][1].reshape(num_features, 1)

# Project the data onto the new feature space (1D)
X_lda = X.dot(W)

# Print the LDA-transformed dataset
print("\nLDA-transformed dataset: ", np.round(X_lda, 2))

# Plot the LDA-transformed dataset
for cl in class_labels:
    plt.scatter(X_lda[y == cl], np.zeros_like(X_lda[y == cl]), label=cl)
plt.xlabel("LD1")
plt.legend()
plt.title("LDA-transformed dataset")
plt.show()

# Predict the class of a new data point
new_data_point = np.array([[60, 4.3]]) # New data point
new_data_point_lda = new_data_point.dot(W) # Project the new data point onto the new feature space
print("\nNew data point in LDA space: ", np.round(new_data_point_lda, 2))

# Plot the prediction
for cl in class_labels:
    plt.scatter(X_lda[y == cl], np.zeros_like(X_lda[y == cl]), label=cl)
plt.scatter(new_data_point_lda, 0, color='black', label='New data point', marker='x')
plt.xlabel("LD1")
plt.legend()
plt.title("LDA-transformed dataset with the prediction")
plt.show()