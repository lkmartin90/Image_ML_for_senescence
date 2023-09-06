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

E31 = pd.read_csv('./E31_drug_discovery/E31_senescence_data.csv')
E57 = pd.read_csv('./E57_drug_discovery/E57_senescence_data.csv')
E55 = pd.read_csv('./E55/E55_senescence_data.csv')
E53 = pd.read_csv('./E53/E53_senescence_data.csv')

E55['Not Senescent'] = np.where(E55['Senescent']==1, 0, 1)


data = pd.concat([E31, E57, E55, E53], ignore_index=True)

# Split data into test and train

y = data["Senescent"]

# remove everything not DAPI related
x = data.copy()
x = x.drop(
    ['ImageNumber', 'ObjectNumber', 'Number_Object_Number',
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
                             "Intensity_StdIntensity", "Intensity_UpperQuartileIntensity", 'Metadata_Radiated']

# keep only features that we have drug discovery data for

features_to_keep = ["Metadata_CellLine", "Senescent"]
for feature in x.columns:
    split_array = feature.split('_')
    if len(split_array)> 1:
        double_name = split_array[0] + "_" + split_array[1]
        if double_name in features_to_keep_template:
            features_to_keep.append(feature)

x = x[features_to_keep]

# do not delete - deals with intensities that arn't DAPI
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
x = x.drop(['Metadata_CellLine'], axis=1)
x = x.drop(['Senescent'], axis=1)

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
x_2_c = data.copy()
x_2_catagories = x_2_c[(x_2_c['Senescent'] == 1) | (x_2_c['Not Senescent'] == 1)]
y_2 = x_2_catagories["Senescent"]

x_2_rest = x_2_c[(x_2_c['Senescent'] != 1) & (x_2_c['Not Senescent'] != 1)]
y_2_rest = x_2_rest["Senescent"]

# restrict to features we want, plus Metadata_Radiated
x_2_catagories = x_2_catagories[features_to_keep]
x_2_rest = x_2_rest[features_to_keep]

for column in x_2_catagories.columns:
    split_cols = column.split('_')
    if len(split_cols) > 2:
        if split_cols[2] == 'CorrLaminB1' or split_cols[2] == 'CorrP21':
            to_drop.append(column)

x_2_catagories = x_2_catagories.drop(to_drop, axis=1)
x_2_rest = x_2_rest.drop(to_drop, axis=1)

# split into test and train
x_train_2_full, x_test_2_full, y_train_2, y_test_2 = train_test_split(x_2_catagories, y_2, test_size=fraction_to_test)

x_rest_test = pd.concat([x_test_2_full, x_2_rest])
y_rest_test = pd.concat([y_test_2, y_2_rest])
cell_lines = x_rest_test["Metadata_CellLine"]
sen = x_rest_test["Senescent"]

x_rest_test = x_rest_test.drop(['Metadata_CellLine'], axis=1)
x_train_2_full = x_train_2_full.drop(['Metadata_CellLine'], axis=1)
x_test_2_full = x_test_2_full.drop(['Metadata_CellLine'], axis=1)
x_rest_test = x_rest_test.drop(['Senescent'], axis=1)
x_train_2_full = x_train_2_full.drop(['Senescent'], axis=1)
x_test_2_full = x_test_2_full.drop(['Senescent'], axis=1)

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


print("Train on very senescent and very non-senescent, test on all")
y_pred_svm_3 = clf_svm_2.predict(x_rest_2)
pred_probs_svm_3 = clf_svm_2.decision_function(x_rest_2)
print("Accuracy:", metrics.accuracy_score(y_rest_test, y_pred_svm_3))
print("Precision:", metrics.precision_score(y_rest_test, y_pred_svm_3))
print("Recall:", metrics.recall_score(y_rest_test, y_pred_svm_3))

fpr_2, tpr_2, thresholds_2 = roc_curve(y_test_2, pred_probs_svm_2)
roc_auc_2 = auc(fpr_2, tpr_2)

fpr_3, tpr_3, thresholds_3 = roc_curve(y_rest_test, pred_probs_svm_3)
roc_auc_3 = auc(fpr_3, tpr_3)

display_2 = RocCurveDisplay(fpr=fpr_2, tpr=tpr_2, roc_auc=roc_auc_2, estimator_name='Train and test on subset')
display_3 = RocCurveDisplay(fpr=fpr_3, tpr=tpr_3, roc_auc=roc_auc_3, estimator_name='Train on subset, test on all')
display_2.plot()
display_3.plot()
plt.show()

results_df = pd.DataFrame([pred_probs_svm_3, y_rest_test]).T
results_df = results_df.sort_values(by=[0])
plot_ordered_classifier_score(results_df, "All", "SVM")
# pickle

filename = 'AllLines_SVM_model.sav'
pickle.dump(clf_svm_2, open(filename, 'wb'))

# calculate fraction senescent of each cell line

x_rest_test["sen_prediction"] = y_pred_svm_3
x_rest_test["sen_prob"] = pred_probs_svm_3
x_rest_test["Metadata_CellLine"] = cell_lines
x_rest_test["Senescent"] = sen
print(x_rest_test.groupby(by=["Metadata_CellLine"]).sum()["sen_prediction"])
print(x_rest_test.groupby(by=["Metadata_CellLine"]).count()["sen_prediction"])

for cell_line in ["E31", "E57", "E55", "E53"]:
    print(cell_line)
    temp = x_rest_test[x_rest_test["Metadata_CellLine"] == cell_line]

    fpr, tpr, thresholds = roc_curve(temp["Senescent"], temp["sen_prob"])
    roc_auc = auc(fpr, tpr)

    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=cell_line + ' Train on subset, test on all')
    display.plot()
    plt.show()

    print("Accuracy:", metrics.accuracy_score(temp["Senescent"], temp["sen_prediction"]))
    print("Precision:", metrics.precision_score(temp["Senescent"], temp["sen_prediction"]))
    print("Recall:", metrics.recall_score(temp["Senescent"], temp["sen_prediction"]))

