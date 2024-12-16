import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# 读取CSV文件，假设没有表头
data = pd.read_csv('Database/train_x.csv', header=None)

# 提取最后三列
last_three_columns = data.iloc[:, -3:]


# 应用K-Means聚类算法
# 假设我们想要将数据聚成3个簇，这个数字可以根据实际情况调整
kmeans = KMeans(n_clusters=8, random_state=1).fit(last_three_columns)

# 将聚类结果添加到原始数据中
centroids = kmeans.cluster_centers_

centroids_df = pd.DataFrame(centroids)
# 如果需要，可以将聚类结果保存到新的CSV文件中
print(centroids_df)
centroids_df.to_csv('Database/centroids.csv', index=False, header=False)