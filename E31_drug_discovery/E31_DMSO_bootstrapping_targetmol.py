import sys

sys.path.append("..")
import pandas as pd
from functions import *
from random import choices

########################################################
# Import full data
########################################################

drug_data_tot = pd.read_csv('E31_targetmol_full_data.csv')
drug_data_tot["well_ending"] = drug_data_tot["Metadata_well"].str[1:]

########################################################
# Import senscore data
########################################################

all_data = pd.read_csv("E31_senscore_targetmol.csv")

all_data["ID"] = all_data["Metadata_platename"] + "_" + all_data["Metadata_well"]

all_data["fraction_sen"] = all_data["number_sen"] / all_data["cell_no"]

all_data["conc"] = all_data["Metadata_platename"].map(
    {'E31-TM-0-01': 0.01, 'E31-TM-0-1': 0.1, 'E31-TM-1-0': 1, 'E31-TM-10': 10})

all_data["row"] = all_data["Metadata_well"].str[:1]

all_data["well_ending"] = all_data["Metadata_well"].str[1:]

########################################################
# Bootstrapping
########################################################

# Here want to only keep the DMSO columns

DMSO_data = drug_data_tot.copy().loc[(drug_data_tot["well_ending"] == "01") | (drug_data_tot["well_ending"] == "02")]

# Then want to bootstrap from each DMSO well
# For each well, want to take a subset of the cells, and take the mean of the senescence score

all_dfs = []

for plate in set(DMSO_data["Metadata_platename"]):
    print(plate)
    for row in set(DMSO_data["row"]):

        print(row)
        # want to actually pool the wells 01 and 02 with the same starting letter
        DMSO_on_plate = DMSO_data[(DMSO_data["Metadata_platename"] == plate)].copy()

        num_DMSO_cells_on_plate = len(DMSO_on_plate)

        # find the number of cells in that sample
        for well in set(all_data["well_ending"]):
            print(well)
            if well != "01" and well != "02":

                num_cells_for_bootstrap = len(drug_data_tot[(drug_data_tot["Metadata_platename"] == plate) &
                                                            (drug_data_tot["row"] == row) &
                                                            (drug_data_tot["well_ending"] == well)])

                # create empty arrays to store the summary data
                subset_mean_senscore = []
                subset_num_sencells = []
                subset_std_senscore = []
                subset_num_cells = []

                # Do 50 times. Is this enough?

                for i in range(50):
                    # chose random subset so cells, want this to be WITH replacement
                    chosen_cells = choices(DMSO_on_plate.index, k=num_cells_for_bootstrap)
                    chosen_cell_data = DMSO_on_plate.loc[chosen_cells]
                    # print(len(chosen_cell_data))

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
                subset_df = pd.DataFrame(
                    [subset_mean_senscore, subset_num_sencells, subset_std_senscore, subset_num_cells])
                subset_df = subset_df.T
                subset_df = subset_df.rename({0: 'mean_senscore', 1: 'num_sencells', 2: 'std_senscore', 3: 'num_cells'},
                                             axis=1)

                subset_df["ID"] = plate + "_" + row + well
                all_dfs.append(subset_df)

# create df with all data and read out to a file
bootstrapped_data = pd.concat(all_dfs)

print(bootstrapped_data.groupby("ID").std())

std_senscore_from_bootstrapping = bootstrapped_data.groupby("ID").std()["mean_senscore"]
std_numsen_from_bootstrapping = bootstrapped_data.groupby("ID").std()["num_sencells"]
all_data = all_data.set_index("ID")
all_data["boostrapped_senscore_mean_std"] = std_senscore_from_bootstrapping
all_data["boostrapped_numcells_mean_std"] = std_numsen_from_bootstrapping

print(all_data)
all_data.to_csv("E31_targetmol_with_bootstrap.csv")
