import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset and select the six spending variables.
df = pd.read_csv("Wholesale customers data.csv")
variabile = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
x = df[variabile]

# Standardize the variables
x_std = StandardScaler().fit_transform(x)

# Create the correlation matrix using seaborn
R = x.corr()
sns.heatmap(R, annot=True, fmt=".2f", cmap="RdYlBu", vmin=-1, vmax=1)
plt.title("Matricea de corelatie")
plt.show()

# Apply PCA and calculate the explained variance ratio for each component.
pca = PCA()
pca.fit(x_std)
ratio = pca.explained_variance_ratio_
for i, r in enumerate(ratio, start=1):
    print(f"PC{i}: {r * 100:.2f}%")

# Determine how many components are necessary to explain at least 80% of the total variance.
cumulat = np.cumsum(ratio)
nr_comp = int(np.argmax(cumulat >= 0.80)) + 1
print(f"Sunt necesare {nr_comp} componente pentru a explica cel putin 80% din varianta ({cumulat[nr_comp - 1] * 100:.2f}%).")