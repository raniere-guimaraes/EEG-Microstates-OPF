# Import libraries
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import opfython.math.general as g
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.stream import loader
from opfython.models import KNNSupervisedOPF
import time
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support

# Read txt file
df_microstates = loader.load_txt('C:/microstates_features.txt')

# Separeting data and label
features = df_microstates[:,2:322]
label = df_microstates[:,1]
index = df_microstates[:,0]

# Create list for storege all 15 values of time
list_time_umap = []
import umap.umap_ as umap

# Start time pca processing
t_start_umap = time.time()

# Applying UMAP
umap_tuab = umap.UMAP(n_neighbors=15, n_components=3, min_dist=0.1, metric='euclidean').fit_transform(features)

# End time pca processing
t_end_umap = time.time()
t_total_umap = t_end_umap - t_start_umap
print('Time total UMAP: ', t_total_umap)

# Append all list_matrix_all_values
list_time_umap.append(t_total_umap)

# Number of repetitions counter 
print('Number of values of times: ', len(list_time_umap))

# Mean and Standard Deviation in the list time
print('Value mean: ', np.mean(list_time_umap))
print('Standard Deviation: ', np.std(list_time_umap))

# Data Frame to create plot 3D
df_umap_tuab = pd.DataFrame(dict(label=label, x=umap_tuab[:, 0], y=umap_tuab[:, 1], z=umap_tuab[:, 2]))
df_umap_tuab_0 = df_umap_tuab[df_umap_tuab["label"] == 0]
df_umap_tuab_1 = df_umap_tuab[df_umap_tuab["label"] == 1]

#Plot reduced dimension for 3D
colors=['b', 'r']
ax = plt.subplot(111, projection='3d')
ax.scatter(df_umap_tuab_0[['x']], df_umap_tuab_0[['y']], df_umap_tuab_0[['z']], color=colors[0], s=20 , label='Normal')
ax.scatter(df_umap_tuab_1[['x']], df_umap_tuab_1[['y']], df_umap_tuab_1[['z']], color=colors[1], s=20 , label='Abnormal')
plt.legend(numpoints=1, ncol=1, fontsize=8, bbox_to_anchor=(1.0, 1.0))
plt.show()

# Normalization scales each input variable separately to the range [0-1]
scaler = MinMaxScaler()
model = scaler.fit(umap_tuab)
scaled_umap_tuab = model.transform(umap_tuab)

# Create input array for OPF
tuab_umap_opf = np.stack((df_microstates[:,0], df_microstates[:,1], scaled_umap_tuab[:,0], scaled_umap_tuab[:,1], scaled_umap_tuab[:,2]), axis = 1)

# Parsing a pre-loaded numpy array
X, Y = p.parse_loader(tuab_umap_opf)

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = s.split(X, Y, percentage=0.8, random_state=1)

# Splitting data into training and validation sets
X_train, X_val, Y_train, Y_val = s.split(X_train, Y_train, percentage=0.25, random_state=1)

# All distances
distances = ["additive_symmetric","average_euclidean","bhattacharyya","bray_curtis","canberra","chebyshev","chi_squared","chord","clark",
    "cosine","dice","divergence","euclidean","gaussian","gower","hamming","hassanat","hellinger","jaccard","jeffreys","jensen",
    "jensen_shannon","k_divergence","kulczynski","kullback_leibler","log_euclidean","log_squared_euclidean","lorentzian","manhattan",
    "matusita","max_symmetric","mean_censored_euclidean","min_symmetric","neyman","non_intersection","pearson","sangvi","soergel","squared",
    "squared_chord","squared_euclidean","statistic","topsoe","vicis_symmetric1","vicis_symmetric2","vicis_symmetric3","vicis_wave_hedges"]

# Create list for storege all 15 matrix
list_all_matrix = []

# All metrics about distances
list_acc = []
list_distance = []
list_t_total_train = []
list_t_total_pred = []
list_prec_rec_fs = []
list_matrix_all_values = []
for i in range (len(distances)):
    # Creates an SupervisedOPF instance
    opf = KNNSupervisedOPF(max_k=10, distance=distances[i], pre_computed_distance=None)
    # Fits training data into the classifier
    t_start_train = time.time()
    opf.fit(X_train, Y_train, X_val, Y_val)
    t_end_train = time.time()
    t_total_train = t_end_train - t_start_train
    list_t_total_train.append(t_total_train)
    # Predicts new data
    t_start_pred = time.time()
    preds = opf.predict(X_test)
    t_end_pred = time.time()
    t_total_pred = t_end_pred - t_start_pred
    list_t_total_pred.append(t_total_pred)
    # Calculating accuracy
    acc = g.opf_accuracy(Y_test, preds)
    list_distance.append(distances[i])
    list_acc.append(acc)
    # Precision, Recall and F-score
    prec_rec_fs = precision_recall_fscore_support(Y_test, preds, average='macro')
    list_prec_rec_fs.append(prec_rec_fs)

# Create full matrix from all lists
matrix_all_values = np.column_stack((list_distance, list_acc, list_t_total_train, list_t_total_pred, list_prec_rec_fs))
list_matrix_all_values.append(matrix_all_values)

# Append all list_matrix_all_values
list_all_matrix.append(list_matrix_all_values[0])

# Number of repetitions counter 
print('Number of matrices: ', len(list_all_matrix))

# List all values accuracy
list_accuracy = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    for j in range(len(list_all_matrix)):
        list_accuracy[i].append(list_all_matrix[j][i][1])

# Mean accuracy all distances
list_mean_accuracy = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    list_mean_accuracy[i].append(np.mean(list_accuracy[i]))

# Standard Deviation accuracy all distances
list_std_accuracy = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    list_std_accuracy[i].append(np.std(list_accuracy[i]))

# List all values time train
list_time_train = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    for j in range(len(list_all_matrix)):
        list_time_train[i].append(list_all_matrix[j][i][2])

# Mean time train all distances
list_mean_time_train = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    list_mean_time_train[i].append(np.mean(list_time_train[i]))

# Standard Deviation time train all distances
list_std_time_train = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    list_std_time_train[i].append(np.std(list_time_train[i]))

# List all values time prediction
list_time_pred = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    for j in range(len(list_all_matrix)):
        list_time_pred[i].append(list_all_matrix[j][i][3])

# Mean time prediction all distances
list_mean_time_pred = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    list_mean_time_pred[i].append(np.mean(list_time_pred[i]))

# Standard Deviation time prediction all distances
list_std_time_pred = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    list_std_time_pred[i].append(np.std(list_time_pred[i]))

# List all values precision
list_precision = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    for j in range(len(list_all_matrix)):
        list_precision[i].append(list_all_matrix[j][i][4])

# Mean precision all distances
list_mean_precision = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    list_mean_precision[i].append(np.mean(list_precision[i]))

# Standard Deviation precision all distances
list_std_precision = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    list_std_precision[i].append(np.std(list_precision[i]))

# List all values recall
list_recall = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    for j in range(len(list_all_matrix)):
        list_recall[i].append(list_all_matrix[j][i][5])

# Mean recall all distances
list_mean_recall = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    list_mean_recall[i].append(np.mean(list_recall[i]))

# Standard Deviation recall all distances
list_std_recall = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    list_std_recall[i].append(np.std(list_recall[i]))

# List all values f-score
list_f_score = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    for j in range(len(list_all_matrix)):
        list_f_score[i].append(list_all_matrix[j][i][6])

# Mean f-score all distances
list_mean_f_score = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    list_mean_f_score[i].append(np.mean(list_f_score[i]))

# Standard Deviation f-score all distances
list_std_f_score = [[] for i in range(len(list_distance))]
for i in range(len(list_distance)):
    list_std_f_score[i].append(np.std(list_f_score[i]))

# Matrix with all statistics
matrix_all_statistics = np.column_stack((list_distance, list_mean_accuracy, list_std_accuracy, list_mean_time_train, list_std_time_train, 
list_mean_time_pred, list_std_time_pred, list_mean_precision, list_std_precision, list_mean_recall, list_std_recall, list_mean_f_score,
list_std_f_score))

# Row and column names
row_names = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19',
'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32', 'D33', 'D34', 'D35', 'D36', 'D37', 'D38',
'D39', 'D40', 'D41', 'D42', 'D43', 'D44', 'D45', 'D46', 'D47']
column_names = ['Distance', 'Mean Accuracy', 'Std Accuracy', 'Mean Time Train', 'Std Time Train', 'Mean Time Predict', 'Std Time Predict', 
'Mean Precision', 'Std Precision', 'Mean Recall',  'Std Recall', 'Mean F-score', 'Std F-score']