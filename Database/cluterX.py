import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


# Load the new data file without headers
data_df2 = pd.read_csv('x2.csv', header=None)
centroids_df = pd.read_csv('centroids.csv', header=None)

# Extract the last three columns for each row in the data
last_three_columns = data_df2.iloc[:, -3:]

# Compute the distance between each of these points and the centroids
distances2 = cdist(last_three_columns.values, centroids_df.values)

# Find the index of the closest centroid for each point
closest_centroids_indices2 = np.argmin(distances2, axis=1)

# Replace the last three columns with the corresponding centroid values
for i, index in enumerate(closest_centroids_indices2):
    data_df2.iloc[i, -3:] = centroids_df.iloc[index].values

# Save the modified data
output_path2_corrected = 'x2_modified_corrected.csv'
data_df2.to_csv(output_path2_corrected, index=False, header=False)




