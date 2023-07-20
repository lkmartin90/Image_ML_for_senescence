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

################################################################
# start by processing our E31 data and using it to train the machine learning model
################################################################

#  Read in data and pre-process

E_31_NucleiObject = pd.read_csv("../E31_LaminB1_P21_data/E31_050423_P21_LaminB1NucleiObject.csv")
E_31_NucleiObject = data_processing_Nuclei(E_31_NucleiObject)
E_31_DilatedNuclei = pd.read_csv("../E31_LaminB1_P21_data/E31_050423_P21_LaminB1DilatedNuclei_1.csv")
E_31_DilatedNuclei = data_processing_Dilated(E_31_DilatedNuclei)

# create one data object containing all the data
data = pd.concat([E_31_NucleiObject, E_31_DilatedNuclei], axis=1)

# Rescale intensity measures based on background levels

E_31_image = pd.read_csv("../E31_LaminB1_P21_data/E31_050423_P21_LaminB1Image.csv")

# subtract background from each image
data = rescale_from_background(data, E_31_image)

# check columns
# data.columns[0:100]

# Remove columns that are entirely NaN

data = data.dropna(axis='columns')

# Remove cells that are an outlier in many catagories

data = find_outliers(data, 70)

# Filter based on cell size

data = data[data['AreaShape_Area'] > 180]

# Filter based on Std dev

data = data[data['Intensity_StdIntensity_CorrNuclei'] > 0.001]

# Create new features

data = create_new_features(data)

# Use this to add a column labelling those with low LaminB1 and high P21 as senescent

Lam_cutoff = 0.006
P21_cutoff = 0.0115

data['Senescent'] = 0
data.loc[(data.Intensity_MeanIntensity_CorrLaminB1 < Lam_cutoff) & (
            data.Intensity_MeanIntensity_CorrP21 > P21_cutoff), 'Senescent'] = 1

Lam_cutoff_1 = 0.0063
P21_cutoff_1 = 0.01

data['Not Senescent'] = 0
data.loc[(data.Intensity_MeanIntensity_CorrLaminB1 > Lam_cutoff_1) & (
            data.Intensity_MeanIntensity_CorrP21 < P21_cutoff_1), 'Not Senescent'] = 1

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
    ['Metadata_CellLine', 'ImageNumber', 'ObjectNumber', 'Metadata_Radiated', 'Number_Object_Number', 'Senescent',
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
                             "Intensity_StdIntensity", "Intensity_UpperQuartileIntensity", "cell_line", "x_proj",
                             "y_proj"]

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

x_train_full, x_test_full, y_train, y_test = train_test_split(x, y, test_size=fraction_to_test)


x_train = x_train_full.copy().drop(['x_proj', 'y_proj'], axis=1)
x_test = x_test_full.copy().drop(['x_proj', 'y_proj'], axis=1)

# scale

x_test = StandardScaler().fit_transform(x_test)
x_train = StandardScaler().fit_transform(x_train)

print("Shape of training data")
print(x_train.shape)

print("Shape of testing data")
print(x_train_full.shape)

# SVM
# Test and train on sensesent and non-sen

# filter data for cells with only a score in either senescence or no senescence columns
x_2_catagories = data_for_pca.copy()
x_2_catagories = x_2_catagories[(x_2_catagories['Senescent'] == 1) | (x_2_catagories['Not Senescent'] == 1)]
y_2 = x_2_catagories["Senescent"]

x_2_catagories = x_2_catagories[features_to_keep]

to_drop = ["x_proj", "y_proj"]
for column in x_2_catagories.columns:
    split_cols = column.split('_')
    if len(split_cols) > 2:
        if split_cols[2] == 'CorrLaminB1' or split_cols[2] == 'CorrP21':
            to_drop.append(column)
x_2_catagories = x_2_catagories.drop(to_drop, axis=1)

x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_2_catagories, y_2, test_size=fraction_to_test)

x_train_2 = StandardScaler().fit_transform(x_train_2)
x_test_2 = StandardScaler().fit_transform(x_test_2)

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
print("Train on very senescent and very non-senescent, test on all")
y_pred_svm_3 = clf_svm_2.predict(x_test)
pred_probs_svm_3 = clf_svm_2.decision_function(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_svm_3))
print("Precision:", metrics.precision_score(y_test, y_pred_svm_3))
print("Recall:", metrics.recall_score(y_test, y_pred_svm_3))

fpr_2, tpr_2, thresholds_2 = roc_curve(y_test_2, pred_probs_svm_2)
roc_auc_2 = auc(fpr_2, tpr_2)

fpr_3, tpr_3, thresholds_3 = roc_curve(y_test, pred_probs_svm_3)
roc_auc_3 = auc(fpr_3, tpr_3)

display_2 = RocCurveDisplay(fpr=fpr_2, tpr=tpr_2, roc_auc=roc_auc_2, estimator_name='example estimator')
display_3 = RocCurveDisplay(fpr=fpr_3, tpr=tpr_3, roc_auc=roc_auc_3, estimator_name='example estimator')
display_2.plot()
display_3.plot()
plt.show()

# pickle

filename = 'E31_SVM_model.sav'
pickle.dump(clf_svm_2, open(filename, 'wb'))
