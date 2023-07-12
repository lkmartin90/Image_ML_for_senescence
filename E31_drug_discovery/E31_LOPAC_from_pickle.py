import numpy as np
import pandas as pd
import os
import umap
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import hdbscan
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.inspection import permutation_importance
from functions import *
import scipy.cluster.hierarchy as sch
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import glob
import pickle


################################################################################################
## Test on drug discovery data
#################################################################################################
# load the data and the model
drug_data_tot = pd.read_csv("E31_LOPAC_data.csv", index_col=False)
drug_data_tot = drug_data_tot.drop("Unnamed: 0", axis = 1)

clf_svm_2 = pickle.load(open('E31_SVM_model.sav', 'rb'))

# AT this stage, before irrelevant features are deleted, we need to correct the intensity of different plates
# against each other. To start with we will not correct features that will vary by cell density, as later cell no.
# varies as drugs kill off cells, but correction would still be applied.

LOPAC_meta = pd.read_csv("/home/lucymartin/Documents/XDF/Cell_image_processing_code/drug_screen_metadata/GCGR-Lpc-Metadata.csv")
LOPAC_meta['Barcode'] = LOPAC_meta['Barcode'].str.replace('.', '-')
drug_data_temp = drug_data_tot.copy()

drug_data_temp = drug_data_temp.set_index(["Metadata_platename", "Metadata_well"])
compound = (LOPAC_meta.set_index(['Barcode', "Well Name"])['chemical_name'])

drug_data_temp["compound"] = compound
drug_data_tot["compound"] = list(drug_data_temp["compound"])

DMSO_grouped_means = pd.DataFrame(drug_data_tot[drug_data_tot["compound"] == "DMSO"].groupby(["Metadata_platename"]).mean())
# DMSO_grouped_means now contains the mean of DMSO samples in 01 and 02 wells, for each plate
plate_names = list(DMSO_grouped_means.index)

print(DMSO_grouped_means)

# test to ensure normaliseation is doing what we expect it to
#test_well = drug_data_tot[(drug_data_tot["Metadata_platename"] == "E31L458-0-5") & (drug_data_tot["Metadata_well"] == "B11")]["Intensity_MaxIntensity_W1"]
#test_well_corrected = (test_well.copy()/DMSO_grouped_means.loc[('E31L458-0-5')]["Intensity_MaxIntensity_W1"]) * DMSO_grouped_means.loc[plate_names[0]]["Intensity_MaxIntensity_W1"]

# want to divide by the mean DMSO intensity for that plate, and multiply by the mean DMSO intensity for plate 1
for measure in drug_data_tot.columns:
    # split the column names at each "_"
    split_measure = measure.split('_')
    if split_measure[0] == "Intensity":
        # if the column is an intensity measure then want to scale by the above masked metrics.
        if split_measure[1][0] != 'S' and split_measure[1] != 'MassDisplacement':
            for plate in plate_names:
                drug_data_tot.loc[(drug_data_tot['Metadata_platename'] == plate), measure] = \
                    drug_data_tot.loc[(drug_data_tot['Metadata_platename'] == plate), measure].copy() \
                    / DMSO_grouped_means.loc[plate][measure] * DMSO_grouped_means.loc[plate_names[0]][measure]


#test_well_after = drug_data_tot[(drug_data_tot["Metadata_platename"] == "E31L458-0-5") & (drug_data_tot["Metadata_well"] == "B11")]["Intensity_MaxIntensity_W1"]
#print("normalisation test")
#print(test_well_corrected)
#print(test_well_after)

# drop columns not needed for ML

drug_data_reduced = drug_data_tot.copy().drop(["Metadata_platename", "Metadata_well", "ImageNumber", "ObjectNumber", "compound"], axis=1)

# scale data
# Implicit assumption here that we have the same proportion os senescent cells as in the training data? YES
# Not sure that this is a problem if we use the relative senescence score?

drug_data_scaled = StandardScaler().fit_transform(drug_data_reduced)

# apply the classifier to the drug discovery data
drug_pred = clf_svm_2.predict(drug_data_scaled)
drug_pred_probs = clf_svm_2.decision_function(drug_data_scaled)

# we produce a senescence score for each cell
# want to scale that score by the minimum so that we have no negative values

print("sen score of 0 mapped to ", min(drug_pred_probs))

drug_data_tot["sen_score"] = drug_pred_probs + -1 * min(drug_pred_probs)

drug_data_tot["sen_prediction"] = drug_pred

drug_data_tot = drug_data_tot.dropna(axis=1)

# group the data so we know what the senescence score is for each well of each plate

grouped_dat = drug_data_tot.groupby(["Metadata_platename", "Metadata_well"])
# find the mean cell senescence score per well
mean_sen_score = grouped_dat.mean()["sen_score"]
# find the standard deviation in senescence score per well
std_sen_score = grouped_dat.std()["sen_score"]

tot_sen = grouped_dat.sum()["sen_prediction"]

index_sen_score = grouped_dat.mean().index

px.scatter(y=drug_pred_probs, x=np.arange(len(drug_pred_probs)))

px.scatter(y=mean_sen_score, error_y=std_sen_score, x=list([x[0] + "_" + x[1] for x in index_sen_score]))

# find the number of cells in each well

cell_no = pd.DataFrame(
    drug_data_tot.groupby(["Metadata_platename", "Metadata_well", "ImageNumber"]).max()["ObjectNumber"].groupby(["Metadata_platename", "Metadata_well"]).sum())

# create summary data for each well and save to file

output_data = pd.DataFrame(mean_sen_score)

output_data["std_sen_score"] = std_sen_score
output_data["cell_no"] = cell_no
output_data["number_sen"] = tot_sen

output_data.to_csv('E31_senscore_LOPAC.csv')
drug_data_tot["ID"] = drug_data_tot["Metadata_platename"] + "_" + drug_data_tot["Metadata_well"]
drug_data_tot.to_csv('E31_LOPAC_full_data.csv')