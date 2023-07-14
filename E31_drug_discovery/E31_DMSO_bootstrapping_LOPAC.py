import sys
sys.path.append("..")
import pandas as pd
from functions import *
from random import choices

########################################################
## Import full data
########################################################

drug_data_tot = pd.read_csv('E31_LOPAC_full_data.csv')
drug_data_tot = drug_data_tot.set_index("ID")

########################################################
## Import senscore data
########################################################

all_data = pd.read_csv("E31_senscore_LOPAC.csv")

all_data["ID"] = all_data["Metadata_platename"] + "_" + all_data["Metadata_well"]

all_data["fraction_sen"] = all_data["number_sen"]/ all_data["cell_no"]

all_data["conc"] = all_data["Metadata_platename"].map({'E31L458-0-5': 0.5, 'E31L458-3': 3, 'E31L459-0-5': 0.5, 'E31L459-3': 3,'E31L461-0-5': 0.5, 'E31L461-3': 3, 'E31L460-0-5': 0.5, 'E31L460-3': 3,})

all_data["row"] = all_data["Metadata_well"].str[:1]

all_data["well_ending"] = all_data["Metadata_well"].str[1:]

# create a list of the compound names matching the compound IDs.
compound = [list(drug_data_tot.loc[x]["compound"])[0] for x in all_data["ID"]]

print(compound)
# asign the compunds to a new column
all_data["compound"] = compound
drug_data_tot = drug_data_tot.reset_index()

########################################################
## Bootstrapping
########################################################

# Here want to only keep the DMSO columns

DMSO_data = drug_data_tot.copy().loc[(drug_data_tot["compound"] == "DMSO")]

# Then want to bootstrap from each DMSO well
# For each well, want to take a subset of the cells, and take the mean of the senescence score

all_dfs = []

for plate in set(all_data["Metadata_platename"]):
    print(plate)

    # want to actually pool the DMSO wells on one plate
    DMSO_on_plate = DMSO_data[(DMSO_data["Metadata_platename"] == plate)].copy()

    num_DMSO_cells_on_plate = len(DMSO_on_plate)

    # for each well on the plate
    for well in set(drug_data_tot[(drug_data_tot["Metadata_platename"] == plate)]["Metadata_well"]):
        print(well)
        # find the compound in that well
        compound = list(drug_data_tot[(drug_data_tot["Metadata_platename"] == plate) & (drug_data_tot["Metadata_well"] == well)]["compound"])[0]

        # if not DMSO do the bootstrapping
        if compound != "DMSO":

            # find the number of cells in the well
            num_cells_for_bootstrap = len(drug_data_tot[(drug_data_tot["Metadata_platename"] == plate) & (drug_data_tot["Metadata_well"] == well)])

            # create empty arrays to store the summary data
            subset_mean_senscore = []
            subset_num_sencells = []
            subset_std_senscore = []
            subset_num_cells = []

            #  Bootstrap 50 times, finding the mean of a subset of the DMSO cells

            for i in range(50):
                # chose random subset so cells, want this to be WITH replacement
                chosen_cells = choices(DMSO_on_plate.index, k=num_cells_for_bootstrap)
                chosen_cell_data = DMSO_on_plate.loc[chosen_cells]
                #print(len(chosen_cell_data))

                # find, mean, standard deviation, and number of senescent cells per well
                mean_sen_score = np.mean(list(chosen_cell_data["sen_score"]))
                std_sen_score = np.std(list(chosen_cell_data["sen_score"]))
                tot_sen = sum(list(chosen_cell_data["sen_prediction"]))

                # add to summary array
                subset_num_cells.append(num_cells_for_bootstrap)
                subset_mean_senscore.append(mean_sen_score)
                subset_num_sencells.append(tot_sen)
                subset_std_senscore.append(std_sen_score)


            # create df of data from this well
            subset_df = pd.DataFrame([subset_mean_senscore, subset_num_sencells, subset_std_senscore, subset_num_cells])
            subset_df = subset_df.T
            subset_df = subset_df.rename({0: 'mean_senscore', 1: 'num_sencells', 2: 'std_senscore', 3: 'num_cells'}, axis=1)


            subset_df["ID"] = plate + "_" + well
            all_dfs.append(subset_df)

# create df with all data and read out to a file
bootstrapped_data = pd.concat(all_dfs)

print(bootstrapped_data.groupby("ID").std())
# find the standard deviation of the bootstrapped means
std_senscore_from_bootstrapping = bootstrapped_data.groupby("ID").std()["mean_senscore"]
std_numsen_from_bootstrapping =  bootstrapped_data.groupby("ID").std()["num_sencells"]
all_data = all_data.set_index("ID")
all_data["boostrapped_senscore_mean_std"] = std_senscore_from_bootstrapping
all_data["boostrapped_numcells_mean_std"] = std_numsen_from_bootstrapping

print(all_data)
all_data.to_csv("E31_LOPAC_with_bootstrap.csv")