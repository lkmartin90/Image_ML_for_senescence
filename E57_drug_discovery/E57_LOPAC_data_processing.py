import pandas as pd
import glob

# This code reads in the drug discovery data for E57 and saves it in a much more managable and publishable format

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

drug_data_list = []
count = 0
for file in glob.glob("/mnt/CELL_PAINTING_DATA/LOPAC_raw_data/E57/E57*"):
    if count < 500:
        print(file)
        path = file + "/Nuclei.csv"
        drug_data = pd.read_csv(path, skiprows=[0])

        # drop features in loop to minimise space usage
        features_to_keep = []
        for feature in drug_data.columns:
            split_array = feature.split('_')
            if len(split_array) > 1:
                double_name = split_array[0] + "_" + split_array[1]
                if double_name in features_to_keep_template:
                    features_to_keep.append(feature)
        features_to_keep.append("Metadata_platename")
        features_to_keep.append("Metadata_well")
        features_to_keep.append("ImageNumber")
        features_to_keep.append("ObjectNumber")

        drug_data = drug_data.copy()[features_to_keep]

        to_drop = []
        for column in drug_data.columns:
            split_cols = column.split('_')
            if len(split_cols) > 2:
                if split_cols[2] != 'W1':
                    to_drop.append(column)
        drug_data = drug_data.drop(to_drop, axis=1)

        # append to list of dfs
        drug_data_list.append(drug_data)
        count = count + 1

drug_data_tot = pd.concat(drug_data_list)

drug_data_tot.to_csv("E57_LOPAC_data.csv")
