import sys

sys.path.append("..")
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from functions import *
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import pickle
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go

################################################################
# start by processing our E31 data and using it to train the machine learning model
################################################################

#  Read in data and pre-process

E_55_control_NucleiObject = pd.read_csv("../E55_LaminB1_P21_data/E55_control_220323_P21_LaminB1NucleiObject.csv")
E_55_radiated_NucleiObject = pd.read_csv("../E55_LaminB1_P21_data/E55_rad_220323_P21_LaminB1NucleiObject.csv")

E_55_control_NucleiObject = data_processing_Nuclei(E_55_control_NucleiObject)
E_55_radiated_NucleiObject = data_processing_Nuclei(E_55_radiated_NucleiObject)

E_55_control_DilatedNuclei = pd.read_csv("../E55_LaminB1_P21_data/E55_control_220323_P21_LaminB1DilatedNuclei_1.csv")
E_55_control_DilatedNuclei = data_processing_Dilated(E_55_control_DilatedNuclei)
E_55_radiated_DilatedNuclei = pd.read_csv("../E55_LaminB1_P21_data/E55_rad_220323_P21_LaminB1DilatedNuclei_1.csv")
E_55_radiated_DilatedNuclei = data_processing_Dilated(E_55_radiated_DilatedNuclei)

control_data = pd.concat([E_55_control_NucleiObject, E_55_control_DilatedNuclei], axis=1)
radiated_data = pd.concat([E_55_radiated_NucleiObject, E_55_radiated_DilatedNuclei], axis=1)

control_data['Metadata_Radiated'] = "control"
radiated_data['Metadata_Radiated'] = "radiated"

control_data["CELL_ID"] = control_data['Metadata_CellLine'].astype(str) + "_" + \
                          control_data['ImageNumber'].astype(str) + "_" + control_data['ObjectNumber'].astype(str) \
                          + "_" + control_data['Metadata_Radiated'].astype(str)
radiated_data["CELL_ID"] = radiated_data['Metadata_CellLine'].astype(str) + "_" + \
                           radiated_data['ImageNumber'].astype(str) + "_" + radiated_data['ObjectNumber'].astype(str) \
                           + "_" + radiated_data['Metadata_Radiated'].astype(str)
control_data = control_data.set_index("CELL_ID")
radiated_data = radiated_data.set_index("CELL_ID")

# create one data object containing all the data
data = pd.concat([control_data, radiated_data])
data['Rad_number'] = data['Metadata_Radiated'].astype(str) + "_" + data['ImageNumber'].astype(str)

# Rescale intensity measures based on background levels

E_55_image_control = pd.read_csv("../E55_LaminB1_P21_data/E55_control_220323_P21_LaminB1Image.csv")
E_55_image_radiated = pd.read_csv("../E55_LaminB1_P21_data/E57_rad_220323_P21_LaminB1Image.csv")

# subtract background from each image
data = rescale_from_background_E55(data, E_55_image_control, E_55_image_radiated)

# check columns
# data.columns[0:100]

# Remove columns that are entirely NaN

data = data.dropna(axis='columns')

# Remove cells that are an outlier in many catagories

data = find_outliers_E55(data, 70)

# Filter based on cell size

data = data[data['AreaShape_Area'] > 150]

# Filter based on Std dev

data = data[data['Intensity_StdIntensity_CorrNuclei'] > 0.001]

# Create new features

data = create_new_features(data)

# Use this to add a column labelling those with low LaminB1 and high P21 as senescent


data['Senescent'] = 0
data.loc[(data.Metadata_Radiated == "radiated"), 'Senescent'] = 1


data['Not Senescent'] = 0
data.loc[(data.Metadata_Radiated == "control"), 'Not Senescent'] = 1

print('Number of cells defined as senescent:')
print(sum(data['Senescent']))

print('Number of cells defined as not senescent:')
print(sum(data['Not Senescent']))

# Project onto senescence axis

data = project_onto_line_pca(data, 'Intensity_MeanIntensity_CorrLaminB1', 'Intensity_MeanIntensity_CorrP21')

# Drop column with nan as all of the entries
data_for_pca = data.dropna(axis='columns')
# replace an infinities with nan, then drop cells with nan
data_for_pca.replace([np.inf, -np.inf], np.nan, inplace=True)
data_for_pca = data_for_pca.dropna()

####################################################################
# Machine learning
####################################################################

# Split data into test and train

y = data_for_pca["Senescent"]

# remove everything not DAPI related
x = data_for_pca.copy()
x = x.drop(
    ['Metadata_CellLine', 'ImageNumber', 'ObjectNumber', 'Number_Object_Number', 'Senescent',
     'Not Senescent'], axis=1)

features_to_keep_template = ["AreaShape_Area", "AreaShape_Compactness", "AreaShape_Eccentricity",
                             "AreaShape_Extent", "AreaShape_FormFactor", "AreaShape_MajorAxisLength",
                             "AreaShape_MaxFeretDiameter",
                             "AreaShape_MaximumRadius", "AreaShape_MeanRadius", "AreaShape_MedianRadius",
                             "AreaShape_MinFeretDiameter",
                             "AreaShape_MinorAxisLength", "AreaShape_Perimeter", "AreaShape_Solidity",
                             "Intensity_IntegratedIntensityEdge", "Intensity_IntegratedIntensity",
                             "Intensity_LowerQuartileIntensity",
                             "Intensity_MADIntensity", "Intensity_MassDisplacement", "Intensity_MaxIntensityEdge",
                             "Intensity_MaxIntensity", "Intensity_MeanIntensityEdge", "Intensity_MeanIntensity",
                             "Intensity_MedianIntensity",
                             "Intensity_MinIntensityEdge", "Intensity_MinIntensity", "Intensity_StdIntensityEdge",
                             "Intensity_StdIntensity", "Intensity_UpperQuartileIntensity", "cell_line", "Metadata_Radiated"]

# keep only features that we have drug discovery data for

features_to_keep = []
for feature in x.columns:
    split_array = feature.split('_')
    double_name = split_array[0] + "_" + split_array[1]
    if double_name in features_to_keep_template:
        features_to_keep.append(feature)

x = x[features_to_keep]

to_drop = []
for column in x.columns:
    split_cols = column.split('_')
    if len(split_cols) > 2:
        if split_cols[2] == 'CorrLaminB1' or split_cols[2] == 'CorrP21':
            to_drop.append(column)
x = x.drop(to_drop, axis=1)

print("Shape of x data")
print(x.shape)

# split into test and train

fraction_to_test = 0.5

# split into test and train
x_train_full, x_test_full, y_train, y_test = train_test_split(x, y, test_size=fraction_to_test)

# train 2 scalers based only on the control data in the test and train sets

train_scaler = StandardScaler().fit(x_train_full[x_train_full['Metadata_Radiated'] == "control"].drop(['Metadata_Radiated'], axis=1))

test_scaler = StandardScaler().fit(x_test_full[x_test_full['Metadata_Radiated'] == "control"].drop(['Metadata_Radiated'], axis=1))

# drop metadata

x_train = x_train_full.copy().drop(['Metadata_Radiated'], axis=1)
x_test = x_test_full.copy().drop(['Metadata_Radiated'], axis=1)

#scale
x_test = test_scaler.transform(x_test)
x_train = train_scaler.transform(x_train)

print("Shape of training data")
print(x_train.shape)

print("Shape of testing data")
print(x_train_full.shape)

# SVM
# Test and train on sensesent and non-sen

# filter data for cells with only a score in either senescence or no senescence columns
x_2_c = data_for_pca.copy()
x_2_catagories = x_2_c[(x_2_c['Senescent'] == 1) | (x_2_c['Not Senescent'] == 1)]
y_2 = x_2_catagories["Senescent"]

x_2_rest = x_2_c[(x_2_c['Senescent'] != 1) & (x_2_c['Not Senescent'] != 1)]
y_2_rest = x_2_rest["Senescent"]


x_2_catagories = x_2_catagories[features_to_keep]
x_2_rest = x_2_rest[features_to_keep]

to_drop = []
for column in x_2_catagories.columns:
    split_cols = column.split('_')
    if len(split_cols) > 2:
        if split_cols[2] == 'CorrLaminB1' or split_cols[2] == 'CorrP21':
            to_drop.append(column)
x_2_catagories = x_2_catagories.drop(to_drop, axis=1)
x_2_rest = x_2_rest.drop(to_drop, axis=1)

# split into test and train
x_train_2_full, x_test_2_full, y_train_2, y_test_2 = train_test_split(x_2_catagories, y_2, test_size=fraction_to_test)

x_rest_test = pd.concat([x_train_2_full, x_2_rest])
y_rest_test = pd.concat([y_train_2, y_2_rest])

# train scaler based on control
train_scaler_2 = StandardScaler().fit(x_train_2_full[x_train_2_full['Metadata_Radiated'] == "control"].drop(['Metadata_Radiated'], axis=1))

test_scaler_2 = StandardScaler().fit(x_test_2_full[x_test_2_full['Metadata_Radiated'] == "control"].drop(['Metadata_Radiated'], axis=1))

rest_scaler_2 = StandardScaler().fit(x_rest_test[x_rest_test['Metadata_Radiated'] == "control"].drop(['Metadata_Radiated'], axis=1))

# drop metadata

x_train_2 = x_train_2_full.copy().drop(['Metadata_Radiated'], axis=1)
x_test_2 = x_test_2_full.copy().drop(['Metadata_Radiated'], axis=1)
x_rest_2 = x_rest_test.copy().drop(['Metadata_Radiated'], axis=1)

#scale
x_test_2 = test_scaler_2.transform(x_test_2)
x_train_2 = train_scaler_2.transform(x_train_2)
x_rest_2 = rest_scaler_2.transform(x_rest_2)

# train on the 2 subtypes, test on the two subtypes
clf_svm_2 = svm.SVC(kernel='rbf')
clf_svm_2.fit(x_train_2, y_train_2)
y_pred_svm_2 = clf_svm_2.predict(x_test_2)
pred_probs_svm_2 = clf_svm_2.decision_function(x_test_2)

print("Train on very senescent and very non-senescent, test on the same")
print("Accuracy:", metrics.accuracy_score(y_test_2, y_pred_svm_2))
print("Precision:", metrics.precision_score(y_test_2, y_pred_svm_2))
print("Recall:", metrics.recall_score(y_test_2, y_pred_svm_2))

# train on the 2 subtypes, test on all
y_pred_svm_3 = clf_svm_2.predict(x_rest_2)
pred_probs_svm_3 = clf_svm_2.decision_function(x_rest_2)
print("Accuracy:", metrics.accuracy_score(y_rest_test, y_pred_svm_3))
print("Precision:", metrics.precision_score(y_rest_test, y_pred_svm_3))
print("Recall:", metrics.recall_score(y_rest_test, y_pred_svm_3))

fpr_2, tpr_2, thresholds_2 = roc_curve(y_test_2, pred_probs_svm_2)
roc_auc_2 = auc(fpr_2, tpr_2)

fpr_3, tpr_3, thresholds_3 = roc_curve(y_rest_test, pred_probs_svm_3)
roc_auc_3 = auc(fpr_3, tpr_3)

display_2 = RocCurveDisplay(fpr=fpr_2, tpr=tpr_2, roc_auc=roc_auc_2,
                            estimator_name='Train on very senescent and very non-senescent cells, test on the same')
display_3 = RocCurveDisplay(fpr=fpr_3, tpr=tpr_3, roc_auc=roc_auc_3,
                            estimator_name='Train on subset, test on all')
display_2.plot()
display_3.plot()
plt.show()

results_df = pd.DataFrame([pred_probs_svm_3, y_rest_test]).T
results_df = results_df.sort_values(by=[0])
plot_ordered_classifier_score(results_df, "E55", "SVM")

# pickle

filename = 'E55_SVM_model.sav'
pickle.dump(clf_svm_2, open(filename, 'wb'))

# find important parameters, can copy out, takes a while to run

# perm_importance = permutation_importance(clf_svm_2, x_test, y_test)
#
# feature_names = x.columns
# features = np.array(feature_names)
#
# sorted_idx = perm_importance.importances_mean.argsort()
#
# fig = go.Figure()
# fig.add_trace(go.Bar(
#     y=features[sorted_idx],
#     x=perm_importance.importances_mean[sorted_idx],
#     name='SF Zoo',
#     orientation='h',
#     marker=dict(
#         color='blue',
#         line=dict(color='darkblue', width=3), opacity = 0.6
#     )
# ))
# fig.update_layout(
#         font=dict(
#             size=18,
#         ),
#     title = "E57 feature importance in SVM",
#     )
# fig.show()