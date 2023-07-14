import sys
sys.path.append("..")
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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

##  Read in data and pre-process

E_57_control_NucleiObject = pd.read_csv("../E57_LaminB1_P21_data/E57_control_220323_P21_LaminB1NucleiObject.csv")
E_57_radiated_NucleiObject = pd.read_csv("../E57_LaminB1_P21_data/E57_rad_220323_P21_LaminB1NucleiObject.csv")

E_57_control_NucleiObject = data_processing_Nuclei(E_57_control_NucleiObject)
E_57_radiated_NucleiObject = data_processing_Nuclei(E_57_radiated_NucleiObject)

E_57_control_DilatedNuclei = pd.read_csv("../E57_LaminB1_P21_data/E57_control_220323_P21_LaminB1DilatedNuclei_1.csv")
E_57_control_DilatedNuclei = data_processing_Dilated(E_57_control_DilatedNuclei)
E_57_radiated_DilatedNuclei = pd.read_csv("../E57_LaminB1_P21_data/E57_rad_220323_P21_LaminB1DilatedNuclei_1.csv")
E_57_radiated_DilatedNuclei = data_processing_Dilated(E_57_radiated_DilatedNuclei)

control_data = pd.concat([E_57_control_NucleiObject, E_57_control_DilatedNuclei], axis=1)
radiated_data = pd.concat([E_57_radiated_NucleiObject, E_57_radiated_DilatedNuclei], axis=1)

control_data['Metadata_Radiated'] = "control"
radiated_data['Metadata_Radiated'] = "radiated"

control_data["CELL_ID"] = control_data['Metadata_CellLine'].astype(str) +"_"+ control_data['ImageNumber'].astype(str) +"_"+  control_data['ObjectNumber'].astype(str) +"_"+  control_data['Metadata_Radiated'].astype(str)
radiated_data["CELL_ID"] = radiated_data['Metadata_CellLine'].astype(str) +"_"+ radiated_data['ImageNumber'].astype(str) +"_"+  radiated_data['ObjectNumber'].astype(str) +"_"+  radiated_data['Metadata_Radiated'].astype(str)
control_data = control_data.set_index("CELL_ID")
radiated_data = radiated_data.set_index("CELL_ID")

# create one data object containing all the data
data = pd.concat([control_data, radiated_data])
data['Rad_number'] = data['Metadata_Radiated'].astype(str) +"_"+ data['ImageNumber'].astype(str)

## Rescale intensity measures based on background levels

E_57_image_control = pd.read_csv("../E57_LaminB1_P21_data/E57_control_220323_P21_LaminB1Image.csv")
E_57_image_radiated = pd.read_csv("../E57_LaminB1_P21_data/E57_rad_220323_P21_LaminB1Image.csv")

fig = px.histogram(data, x='Intensity_MeanIntensity_CorrP21', color='ImageNumber', nbins=2000)
fig.update_layout(
    font=dict(
        size=16,
    )
)
fig.update_layout(
    title="P21 in each cell before normalisation",
    xaxis_title="Mean P21 intensity",
    yaxis_title="Count",
    barmode='overlay')
fig.update_traces(opacity=0.75)

fig.show()

# subtract background from each image
data = rescale_from_background_E55(data, E_57_image_control, E_57_image_radiated)

fig = px.histogram(data, x='Intensity_MeanIntensity_CorrP21', color='ImageNumber', nbins=1000)
fig.update_layout(
    font=dict(
        size=16,
    )
)
fig.update_layout(
    title="P21 in each cell after normalisation",
    xaxis_title="Mean P21 intensity",
    yaxis_title="Count",
    barmode='overlay')
fig.update_traces(opacity=0.75)

fig.show()

# check columns
# data.columns[0:100]

## Remove columns that are entirely NaN

data = data.dropna(axis='columns')

## Remove cells that are an outlier in many catagories

data = find_outliers_E55(data, 70)

## Filter based on cell size

data = data[data['AreaShape_Area'] > 150]

## Filter based on Std dev

data = data[data['Intensity_StdIntensity_CorrNuclei'] > 0.001]

## Create new features

data = create_new_features(data)

## Use this to add a column labelling those with low LaminB1 and high P21 as senescent

Lam_cutoff = 0.0042
P21_cutoff = 0.0022

data['Senescent'] = 0
data.loc[(data.Intensity_MeanIntensity_CorrLaminB1 < Lam_cutoff) & (
            data.Intensity_MeanIntensity_CorrP21 > P21_cutoff), 'Senescent'] = 1

Lam_cutoff_1 = 0.005
P21_cutoff_1 = 0.0019

data['Not Senescent'] = 0
data.loc[(data.Intensity_MeanIntensity_CorrLaminB1 > Lam_cutoff_1) & (
            data.Intensity_MeanIntensity_CorrP21 < P21_cutoff_1), 'Not Senescent'] = 1

print('Number of cells defined as senescent:')
print(sum(data['Senescent']))

print('Number of cells defined as not senescent:')
print(sum(data['Not Senescent']))

## Project onto senescence axis

# data = project_onto_line(data, 'RadialDistribution_MeanFrac_CorrLaminB1_max', 'Intensity_MeanIntensity_CorrP21', projection_line)
data = project_onto_line_pca(data, 'Intensity_MeanIntensity_CorrLaminB1', 'Intensity_MeanIntensity_CorrP21')

## Create data_for_umap

data_for_umap = data.dropna(axis='columns')
data_for_umap = data_for_umap.drop(
    ['Metadata_CellLine', 'ImageNumber', 'ObjectNumber', 'Metadata_Radiated', 'Number_Object_Number', 'Senescent',
     'Not Senescent', 'x_proj', 'y_proj', 'Rad_number'], axis=1)

## Include option to remove granularity

# to_drop = []
# for column in data_for_umap.columns:
#     split_cols = column.split('_')
#     if split_cols[0] == 'Granularity':
#         to_drop.append(column)
# data_for_umap = data_for_umap.drop(to_drop, axis = 1)

## Put a threshold in the data to remove features with low variance

filter_threshold = 0.2
data_for_umap_filtered, filtered_columns = variance_threshold(data_for_umap, filter_threshold)

# Drop column with nan as all of the entries
data_for_pca = data.dropna(axis='columns')
# replace an infinities with nan, then drop cells with nan
data_for_pca.replace([np.inf, -np.inf], np.nan, inplace=True)
data_for_pca = data_for_pca.dropna()
# drop metedata
data_for_pca_coldrop = data_for_pca.drop(
    ['Metadata_CellLine', 'ImageNumber', 'ObjectNumber', 'Metadata_Radiated', 'Number_Object_Number', 'Senescent',
     'x_proj', 'y_proj', 'Rad_number'], axis=1)

####################################################################
# Machine learning
####################################################################

## Split data into test and train

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

## SVM

# selesct "balanced" option as have far fewer positively identified senescenet cells
clf_svm = svm.SVC(kernel='rbf', class_weight='balanced')

clf_svm.fit(x_train, y_train)

y_pred_svm = clf_svm.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred_svm))
# Model Precision:
# What proportion of positive identifications were actually correct?
print("Precision:", metrics.precision_score(y_test, y_pred_svm))
# Model Recall:
# What proportion of actual positives were identified correctly?
print("Recall:", metrics.recall_score(y_test, y_pred_svm))

pred_probs_svm = clf_svm.decision_function(x_test)
results_df = pd.DataFrame([pred_probs_svm, y_test]).T
results_df = results_df.sort_values(by=[0])

px.histogram(x=pred_probs_svm)

plot_ordered_classifier_score(results_df, "E57", "SVM")

x_test_plot = x_test.copy()

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results_test = tsne.fit_transform(x_test_plot)

plot_projection("E57 TSNE on PCA coloured by SVM classifier score ", tsne_results_test, pred_probs_svm)

plot_projection("E57 TSNE on PCA coloured by p21 senescence classification ", tsne_results_test, y_test)

## Test and train on sensesent and non-sen

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

print("Accuracy:", metrics.accuracy_score(y_test_2, y_pred_svm_2))
print("Precision:", metrics.precision_score(y_test_2, y_pred_svm_2))
print("Recall:", metrics.recall_score(y_test_2, y_pred_svm_2))

# train on the 2 subtypes, test on all
y_pred_svm_3 = clf_svm_2.predict(x_test)
pred_probs_svm_3 = clf_svm_2.decision_function(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_svm_3))
print("Precision:", metrics.precision_score(y_test, y_pred_svm_3))
print("Recall:", metrics.recall_score(y_test, y_pred_svm_3))

fpr, tpr, thresholds = roc_curve(y_test, pred_probs_svm)
roc_auc = auc(fpr, tpr)

fpr_2, tpr_2, thresholds_2 = roc_curve(y_test_2, pred_probs_svm_2)
roc_auc_2 = auc(fpr_2, tpr_2)

fpr_3, tpr_3, thresholds_3 = roc_curve(y_test, pred_probs_svm_3)
roc_auc_3 = auc(fpr_3, tpr_3)

display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
display_2 = RocCurveDisplay(fpr=fpr_2, tpr=tpr_2, roc_auc=roc_auc_2, estimator_name='example estimator')
display_3 = RocCurveDisplay(fpr=fpr_3, tpr=tpr_3, roc_auc=roc_auc_3, estimator_name='example estimator')
display.plot()
display_2.plot()
display_3.plot()
plt.show()

## Vary test and train split

frac_test_array = np.arange(0.05, 1.0, 0.01)
frac_acc = []
frac_prec = []
frac_rec = []

for frac in frac_test_array:
    x_train_vary, x_test_vary, y_train_vary, y_test_vary = train_test_split(x_2_catagories, y_2, test_size=frac)
    # train on the 2 subtypes, test on the two subtypes
    x_train_vary = StandardScaler().fit_transform(x_train_vary)
    x_test_vary = StandardScaler().fit_transform(x_test_vary)
    clf_svm_vary = svm.SVC(kernel='rbf')
    clf_svm_vary.fit(x_train_vary, y_train_vary)
    y_pred_vary = clf_svm_vary.predict(x_test_vary)
    pred_probs_vary = clf_svm_vary.decision_function(x_test_vary)
    frac_acc.append(metrics.accuracy_score(y_test_vary, y_pred_vary))
    frac_prec.append(metrics.precision_score(y_test_vary, y_pred_vary))
    frac_rec.append(metrics.recall_score(y_test_vary, y_pred_vary))

frac_to_plot = pd.DataFrame(np.array([frac_test_array, frac_acc, frac_prec, frac_rec]).T,
                            columns=["frac to test", "accuracy", "precision", "recall"])

fig = go.Figure()
fig.add_trace(go.Scatter(x=frac_to_plot["frac to test"], y=frac_to_plot["accuracy"], mode="markers", name="Accuracy"))
fig.add_trace(go.Scatter(x=frac_to_plot["frac to test"], y=frac_to_plot["precision"], mode="markers", name="Precision"))
fig.add_trace(go.Scatter(x=frac_to_plot["frac to test"], y=frac_to_plot["recall"], mode="markers", name="Recall"))
fig.update_layout(
    font=dict(
        size=22,
    )
)
fig.update_layout(
    title="E31 test train split",
    xaxis_title="Fraction of cells tested on",
    yaxis_title="Metric")

## pickle

filename = 'E57_SVM_model.sav'
pickle.dump(clf_svm_2, open(filename, 'wb'))