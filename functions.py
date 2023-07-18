import numpy as np
import plotly.express as px
from skspatial.objects import Line, Point
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from numpy.linalg import eig

def data_processing_Nuclei(data_nucleiobject):
    """Takes a dataNucleiObject and process it, removing columns which relate to the location of the cell.
    We only want to keep the data for the DAPI stain and the nuclear morphology from these images, therefore we remove
    location data and columns relating to P21 and LaminB1.

        Parameters
        -------
        data_nucleiobject: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline

        Returns
        -------
        data_nucleiobject: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline with unneeded columns removed
        """

    # drop some irelevant metadata at the start and the end
    data_nucleiobject = data_nucleiobject.drop(list(data_nucleiobject.columns[4:26]), axis=1)
    to_drop = []
    # search for and drop LaminB1 and P21
    for name in data_nucleiobject.columns:
        split_array = name.split('_')
        if len(split_array) > 2:
            stain_name = split_array[2]
            if stain_name == 'CorrLaminB1' or stain_name == 'CorrP21':
                to_drop.append(name)
                
    data_nucleiobject = data_nucleiobject.drop(to_drop, axis=1)
    data_nucleiobject = data_nucleiobject.drop(['Metadata_C'], axis=1)
    data_nucleiobject["cell_ID"] = data_nucleiobject['Metadata_CellLine'].astype(str) + "_"\
                                   + data_nucleiobject['ImageNumber'].astype(str) + "_"\
                                   + data_nucleiobject['ObjectNumber'].astype(str)
    data_nucleiobject = data_nucleiobject.set_index("cell_ID")
    data_nucleiobject = data_nucleiobject.drop(['Number_Object_Number'], axis=1)
    # search and drop anything to do with location 
    to_drop = []
    for name in data_nucleiobject.columns:
        split_array = name.split('_')
        if len(split_array)>1:
            if split_array[0] == 'Location':
                to_drop.append(name)
            elif split_array[1] == 'BoundingBoxMaximum' or split_array[1] == 'BoundingBoxMinimum' \
                    or split_array[1] == 'Center':
                to_drop.append(name)
    data_nucleiobject = data_nucleiobject.drop(to_drop, axis=1)
    return data_nucleiobject


def data_processing_Dilated(data_dilatednuclei):
    """Takes a dataDilatedObject and process it, removing columns which relate to the location of the cell.
    We don't want to take the morphology data anymore as we've expanded the cell size to capture the P21 and LaminB1,
    so remove all data relating to nuclear DAPI stain.

        Parameters
        -------
        data_dilatedobject: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline

        Returns
        -------
        data_dilatedobject: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline with unneeded columns removed
        """

    data_dilatednuclei["cell_ID"] = data_dilatednuclei['Metadata_CellLine'].astype(str) + "_"\
                                    + data_dilatednuclei['ImageNumber'].astype(str) + "_"\
                                    + data_dilatednuclei['ObjectNumber'].astype(str)
    data_dilatednuclei = data_dilatednuclei.set_index("cell_ID")
    # search for and drop the nuclear stain
    to_drop = []
    for name in data_dilatednuclei.columns:
        split_array = name.split('_')
        if len(split_array) < 3 and name != 'Metadata_Radiated':
            to_drop.append(name)
        elif len(split_array) > 2:
            stain_name = split_array[2]
            if stain_name == 'CorrNuclei':
                to_drop.append(name)
    data_dilatednuclei = data_dilatednuclei.drop(to_drop, axis=1)
    to_drop = []
    for name in data_dilatednuclei.columns:
        split_array = name.split('_')
        if len(split_array)>1:
            if split_array[0] == 'Location' or split_array[0] == 'Texture':# or split_array[0] == 'Granularity':
                to_drop.append(name)
    data_dilatednuclei = data_dilatednuclei.drop(to_drop, axis=1)

    return data_dilatednuclei


def data_processing_Dilated_EDU(data_dilatednuclei):
    """Takes a dataDilatedObject and process it, removing columns which relate to the location of the cell.
    We don't want to take the morphology data anymore as we've expanded the cell size to capture the P21 and LaminB1,
    so remove all data relating to nuclear DAPI stain.

        Parameters
        -------
        data_dilatedobject: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline

        Returns
        -------
        data_dilatedobject: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline with unneeded columns removed
        """

    data_dilatednuclei["cell_ID"] = data_dilatednuclei['Metadata_CellLine'].astype(str) + "_"\
                                    + data_dilatednuclei['ImageNumber'].astype(str) + "_"\
                                    + data_dilatednuclei['ObjectNumber'].astype(str)
    data_dilatednuclei = data_dilatednuclei.set_index("cell_ID")
    # search for and drop the nuclear stain
    to_drop = []
    for name in data_dilatednuclei.columns:
        split_array = name.split('_')
        if len(split_array) < 3 and name != 'Metadata_Radiated':
            to_drop.append(name)
        elif len(split_array) > 2:
            stain_name = split_array[2]
            if stain_name == 'CorrNuclei' or stain_name == "CorrEDU":
                to_drop.append(name)
    data_dilatednuclei = data_dilatednuclei.drop(to_drop, axis=1)
    to_drop = []
    for name in data_dilatednuclei.columns:
        split_array = name.split('_')
        if len(split_array)>1:
            if split_array[0] == 'Location' or split_array[0] == 'Texture':# or split_array[0] == 'Granularity':
                to_drop.append(name)
    data_dilatednuclei = data_dilatednuclei.drop(to_drop, axis=1)

    return data_dilatednuclei


def rescale_from_background(data, image_data):
    """Rescales the intensity data for each stain based on the background intensity for each stain in each image, as
    ouputted by cell profiler.

        Parameters
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline
        image_data: pandas dataframe
            dataframe containing the "Image" output from a cell profiler pipeline, including background intensities

        Returns
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline with intensity data rescaled
        """

    background_LaminB1 = image_data['Intensity_MeanIntensity_MaskLaminB1']
    background_P21 = image_data['Intensity_MeanIntensity_MaskP21']
    background_DAPI = image_data['Intensity_MeanIntensity_MaskNuclei']

    mean_P21_control = image_data.loc[image_data['Metadata_Radiated'] == "control"]['Intensity_MeanIntensity_MaskP21'].mean()
    mean_LamB1_control = image_data.loc[image_data['Metadata_Radiated'] == "control"]['Intensity_MeanIntensity_MaskLaminB1'].mean()
    mean_DAPI_control = image_data.loc[image_data['Metadata_Radiated'] == "control"]['Intensity_MeanIntensity_MaskNuclei'].mean()
    mean_P21_radiated = image_data.loc[image_data['Metadata_Radiated'] == "radiated"]['Intensity_MeanIntensity_MaskP21'].mean()
    mean_LamB1_radiated = image_data.loc[image_data['Metadata_Radiated'] == "radiated"]['Intensity_MeanIntensity_MaskLaminB1'].mean()
    mean_DAPI_radiated = image_data.loc[image_data['Metadata_Radiated'] == "radiated"]['Intensity_MeanIntensity_MaskNuclei'].mean()


    print('Mean DAPI in background of radiated images: ', mean_DAPI_radiated)
    print('Mean P21 in background of radiated images: ', mean_P21_radiated)
    print('Mean LaminB1 in background of radiated images: ', mean_LamB1_radiated)
    print('Mean DAPI in background of control images: ', mean_DAPI_control)
    print('Mean P21 in background of control images: ', mean_P21_control)
    print('Mean LaminB1 in background of control images: ', mean_LamB1_control)

    for measure in data.columns:
        # split the column names at each "_"
        split_measure = measure.split('_')
        if split_measure[0] == "Intensity":
            # if the column is an intensity measure then want to scale by the above masked metrics.

            if split_measure[2] == 'CorrNuclei':
                # select those columns which are intensity based but not standard deviations
                if split_measure[1][0] != 'S' and split_measure[1] != 'MassDisplacement':
                    for ImNum in list(image_data.loc[image_data['Metadata_Radiated'] == "radiated"]["ImageNumber"]):
                        # print(ImNum)
                        # print(data.loc[(data['Metadata_Radiated'] == "control")&(data['ImageNumber']== ImNum), measure])
                        data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_DAPI.loc[ImNum - 1])
                        # print(data.loc[(data['Metadata_Radiated'] == "control")&(data['ImageNumber']== ImNum), measure])
                    for ImNum in list(image_data.loc[image_data['Metadata_Radiated'] == "control"]["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_DAPI.loc[ImNum - 1])

            elif split_measure[2] == 'CorrP21':
                if split_measure[1][0] != 'S' and split_measure[1] != 'MassDisplacement':
                    for ImNum in list(image_data.loc[image_data['Metadata_Radiated'] == "radiated"]["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_P21.loc[ImNum - 1])
                    for ImNum in list(image_data.loc[image_data['Metadata_Radiated'] == "control"]["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_P21.loc[ImNum - 1])

            elif split_measure[2] == 'CorrLaminB1':
                if split_measure[1][0] != 'S' and split_measure[1] != 'MassDisplacement':
                    for ImNum in list(image_data.loc[image_data['Metadata_Radiated'] == "radiated"]["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_LaminB1.loc[ImNum - 1])
                    for ImNum in list(image_data.loc[image_data['Metadata_Radiated'] == "control"]["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_LaminB1.loc[ImNum - 1])

    return data


def rescale_from_background_EDU(data, image_data):
    """Rescales the intensity data for each stain based on the background intensity for each stain in each image, as
    ouputted by cell profiler.

        Parameters
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline
        image_data: pandas dataframe
            dataframe containing the "Image" output from a cell profiler pipeline, including background intensities

        Returns
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline with intensity data rescaled
        """

    background_LaminB1 = image_data['Intensity_MeanIntensity_MaskLaminB1']
    background_P21 = image_data['Intensity_MeanIntensity_MaskP21']
    background_DAPI = image_data['Intensity_MeanIntensity_MaskNuclei']
    background_EDU = image_data['Intensity_MeanIntensity_MaskEDU']

    mean_P21_control = image_data.loc[image_data['Metadata_Radiated'] == "control"]['Intensity_MeanIntensity_MaskP21'].mean()
    mean_LamB1_control = image_data.loc[image_data['Metadata_Radiated'] == "control"]['Intensity_MeanIntensity_MaskLaminB1'].mean()
    mean_DAPI_control = image_data.loc[image_data['Metadata_Radiated'] == "control"]['Intensity_MeanIntensity_MaskNuclei'].mean()
    mean_EDU_control = image_data.loc[image_data['Metadata_Radiated'] == "control"][
        'Intensity_MeanIntensity_MaskEDU'].mean()
    mean_P21_radiated = image_data.loc[image_data['Metadata_Radiated'] == "radiated"]['Intensity_MeanIntensity_MaskP21'].mean()
    mean_LamB1_radiated = image_data.loc[image_data['Metadata_Radiated'] == "radiated"]['Intensity_MeanIntensity_MaskLaminB1'].mean()
    mean_DAPI_radiated = image_data.loc[image_data['Metadata_Radiated'] == "radiated"]['Intensity_MeanIntensity_MaskNuclei'].mean()
    mean_EDU_radiated = image_data.loc[image_data['Metadata_Radiated'] == "radiated"][
        'Intensity_MeanIntensity_MaskEDU'].mean()


    print('Mean DAPI in background of radiated images: ', mean_DAPI_radiated)
    print('Mean P21 in background of radiated images: ', mean_P21_radiated)
    print('Mean LaminB1 in background of radiated images: ', mean_LamB1_radiated)
    print('Mean EDU in background of radiated images: ', mean_EDU_radiated)
    print('Mean DAPI in background of control images: ', mean_DAPI_control)
    print('Mean P21 in background of control images: ', mean_P21_control)
    print('Mean LaminB1 in background of control images: ', mean_LamB1_control)
    print('Mean EDU in background of control images: ', mean_EDU_control)

    for measure in data.columns:
        # split the column names at each "_"
        split_measure = measure.split('_')
        if split_measure[0] == "Intensity":
            # if the column is an intensity measure then want to scale by the above masked metrics.

            if split_measure[2] == 'CorrNuclei':
                # select those columns which are intensity based but not standard deviations
                if split_measure[1][0] != 'S' and split_measure[1] != 'MassDisplacement':
                    for ImNum in list(image_data.loc[image_data['Metadata_Radiated'] == "radiated"]["ImageNumber"]):
                        # print(ImNum)
                        # print(data.loc[(data['Metadata_Radiated'] == "control")&(data['ImageNumber']== ImNum), measure])
                        data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_DAPI.loc[ImNum - 1])
                        # print(data.loc[(data['Metadata_Radiated'] == "control")&(data['ImageNumber']== ImNum), measure])
                    for ImNum in list(image_data.loc[image_data['Metadata_Radiated'] == "control"]["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_DAPI.loc[ImNum - 1])

            elif split_measure[2] == 'CorrP21':
                if split_measure[1][0] != 'S' and split_measure[1] != 'MassDisplacement':
                    for ImNum in list(image_data.loc[image_data['Metadata_Radiated'] == "radiated"]["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_P21.loc[ImNum - 1])
                    for ImNum in list(image_data.loc[image_data['Metadata_Radiated'] == "control"]["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_P21.loc[ImNum - 1])

            elif split_measure[2] == 'CorrLaminB1':
                if split_measure[1][0] != 'S' and split_measure[1] != 'MassDisplacement':
                    for ImNum in list(image_data.loc[image_data['Metadata_Radiated'] == "radiated"]["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_LaminB1.loc[ImNum - 1])
                    for ImNum in list(image_data.loc[image_data['Metadata_Radiated'] == "control"]["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_LaminB1.loc[ImNum - 1])

            elif split_measure[2] == 'CorrEDU':
                if split_measure[1][0] != 'S' and split_measure[1] != 'MassDisplacement':
                    for ImNum in list(image_data.loc[image_data['Metadata_Radiated'] == "radiated"]["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_EDU.loc[ImNum - 1])
                    for ImNum in list(image_data.loc[image_data['Metadata_Radiated'] == "control"]["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_EDU.loc[ImNum - 1])

    return data


def rescale_from_background_E55(data, image_data_control, image_data_radiated):
    """Rescales the intensity data for each stain based on the background intensity for each stain in each image, as
    ouputted by cell profiler.

        Parameters
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline
        image_data_control: pandas dataframe
            dataframe containing the "Image" output from a cell profiler pipeline, including background intensities
            for the control data
        image_data_radiated: pandas dataframe
            dataframe containing the "Image" output from a cell profiler pipeline, including background intensities
            for the control data

        Returns
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline with intensity data rescaled
        """

    background_LaminB1_control = image_data_control['Intensity_MeanIntensity_MaskLaminB1']
    background_P21_control = image_data_control['Intensity_MeanIntensity_MaskP21']
    background_DAPI_control = image_data_control['Intensity_MeanIntensity_MaskNuclei']

    background_LaminB1_radiated = image_data_radiated['Intensity_MeanIntensity_MaskLaminB1']
    background_P21_radiated = image_data_radiated['Intensity_MeanIntensity_MaskP21']
    background_DAPI_radiated = image_data_radiated['Intensity_MeanIntensity_MaskNuclei']

    mean_P21_control = background_P21_control.mean()
    mean_P21_radiated = background_P21_radiated.mean()
    mean_LamB1_control = background_LaminB1_control.mean()
    mean_LamB1_radiated = background_LaminB1_radiated.mean()
    mean_DAPI_control = background_DAPI_control.mean()
    mean_DAPI_radiated = background_DAPI_radiated.mean()

    print('Mean DAPI in background of radiated images: ', mean_DAPI_radiated)
    print('Mean P21 in background of radiated images: ', mean_P21_radiated)
    print('Mean LaminB1 in background of radiated images: ', mean_LamB1_radiated)
    print('Mean DAPI in background of control images: ', mean_DAPI_control)
    print('Mean P21 in background of control images: ', mean_P21_control)
    print('Mean LaminB1 in background of control images: ', mean_LamB1_control)

    for measure in data.columns:
        # split the column names at each "_"
        split_measure = measure.split('_')
        if split_measure[0] == "Intensity":
            # if the column is an intensity measure then want to scale by the above masked metrics.

            if split_measure[2] == 'CorrNuclei':
                # select those columns which are intensity based but not standard deviations
                if split_measure[1][0] != 'S' and split_measure[1] != 'MassDisplacement':
                    for ImNum in list(image_data_control["ImageNumber"]):
                        # print(ImNum)
                        # print(data.loc[(data['Metadata_Radiated'] == "control")&(data['ImageNumber']== ImNum), measure])
                        data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_DAPI_control.loc[ImNum - 1])
                        # print(data.loc[(data['Metadata_Radiated'] == "control")&(data['ImageNumber']== ImNum), measure])
                    for ImNum in list(image_data_radiated["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_DAPI_radiated.loc[ImNum - 1])

            elif split_measure[2] == 'CorrP21':
                if split_measure[1][0] != 'S' and split_measure[1] != 'MassDisplacement':
                    for ImNum in list(image_data_control["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_P21_control.loc[ImNum - 1])
                    for ImNum in list(image_data_radiated["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_P21_radiated.loc[ImNum - 1])

            elif split_measure[2] == 'CorrLaminB1':
                if split_measure[1][0] != 'S' and split_measure[1] != 'MassDisplacement':
                    for ImNum in list(image_data_control["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_LaminB1_control.loc[ImNum - 1])
                    for ImNum in list(image_data_radiated["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_LaminB1_radiated.loc[ImNum - 1])

    return data


def rescale_from_background_E55_EDU(data, image_data_control, image_data_radiated):
    """Rescales the intensity data for each stain based on the background intensity for each stain in each image, as
    ouputted by cell profiler.

        Parameters
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline
        image_data_control: pandas dataframe
            dataframe containing the "Image" output from a cell profiler pipeline, including background intensities
            for the control data
        image_data_radiated: pandas dataframe
            dataframe containing the "Image" output from a cell profiler pipeline, including background intensities
            for the control data

        Returns
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline with intensity data rescaled
        """

    background_LaminB1_control = image_data_control['Intensity_MeanIntensity_MaskLaminB1']
    background_P21_control = image_data_control['Intensity_MeanIntensity_MaskP21']
    background_DAPI_control = image_data_control['Intensity_MeanIntensity_MaskNuclei']
    background_EDU_control = image_data_control['Intensity_MeanIntensity_MaskEDU']

    background_LaminB1_radiated = image_data_radiated['Intensity_MeanIntensity_MaskLaminB1']
    background_P21_radiated = image_data_radiated['Intensity_MeanIntensity_MaskP21']
    background_DAPI_radiated = image_data_radiated['Intensity_MeanIntensity_MaskNuclei']
    background_EDU_radiated = image_data_radiated['Intensity_MeanIntensity_MaskEDU']

    mean_P21_control = background_P21_control.mean()
    mean_P21_radiated = background_P21_radiated.mean()
    mean_LamB1_control = background_LaminB1_control.mean()
    mean_LamB1_radiated = background_LaminB1_radiated.mean()
    mean_DAPI_control = background_DAPI_control.mean()
    mean_DAPI_radiated = background_DAPI_radiated.mean()
    mean_EDU_control = background_EDU_control.mean()
    mean_EDU_radiated = background_EDU_radiated.mean()

    print('Mean DAPI in background of radiated images: ', mean_DAPI_radiated)
    print('Mean P21 in background of radiated images: ', mean_P21_radiated)
    print('Mean LaminB1 in background of radiated images: ', mean_LamB1_radiated)
    print('Mean DAPI in background of control images: ', mean_DAPI_control)
    print('Mean P21 in background of control images: ', mean_P21_control)
    print('Mean LaminB1 in background of control images: ', mean_LamB1_control)
    print('Mean EDU in background of control images: ', mean_EDU_control)
    print('Mean EDU in background of radiated images: ', mean_EDU_radiated)

    for measure in data.columns:
        # split the column names at each "_"
        split_measure = measure.split('_')
        if split_measure[0] == "Intensity":
            # if the column is an intensity measure then want to scale by the above masked metrics.

            if split_measure[2] == 'CorrNuclei':
                # select those columns which are intensity based but not standard deviations
                if split_measure[1][0] != 'S' and split_measure[1] != 'MassDisplacement':
                    for ImNum in list(image_data_control["ImageNumber"]):
                        # print(ImNum)
                        # print(data.loc[(data['Metadata_Radiated'] == "control")&(data['ImageNumber']== ImNum), measure])
                        data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_DAPI_control.loc[ImNum - 1])
                        # print(data.loc[(data['Metadata_Radiated'] == "control")&(data['ImageNumber']== ImNum), measure])
                    for ImNum in list(image_data_radiated["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_DAPI_radiated.loc[ImNum - 1])

            elif split_measure[2] == 'CorrP21':
                if split_measure[1][0] != 'S' and split_measure[1] != 'MassDisplacement':
                    for ImNum in list(image_data_control["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_P21_control.loc[ImNum - 1])
                    for ImNum in list(image_data_radiated["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_P21_radiated.loc[ImNum - 1])

            elif split_measure[2] == 'CorrLaminB1':
                if split_measure[1][0] != 'S' and split_measure[1] != 'MassDisplacement':
                    for ImNum in list(image_data_control["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "control") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_LaminB1_control.loc[ImNum - 1])
                    for ImNum in list(image_data_radiated["ImageNumber"]):
                        data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure] = \
                            np.array(data.loc[(data['Metadata_Radiated'] == "radiated") & (data['ImageNumber'] == ImNum), measure].copy()
                                     - background_LaminB1_radiated.loc[ImNum - 1])

    return data


def find_outliers(data, cutoff):
    """Gives each cell an outlier score for each feature. If the cell is above the 95 quantile or below the 5 quantile
    for a given feature the outlier score is increased by one. If a cell has an outlier score above the threshold it
    is removed from the data.

        Parameters
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline
        cutoff: int
            outlier score above which the cell is classed a mistake in the cell profiler cell detection

        Returns
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline with cells that do not meet the cutoff
            removed
        """

    # copy the data frame so we don't accidently make changes to the origonal
    outlier_data = data.copy()
    # drop the first 3 columns as they are metadata
    outlier_data = outlier_data.drop(outlier_data.columns[:3], axis=1)
    outlier_data = outlier_data.drop(['Metadata_Radiated'], axis=1)
    # calculate the 95% quantile
    quantile_95 = outlier_data.quantile(q=0.95)
    # calculate the 5% quantile
    quantile_5 = outlier_data.quantile(q=0.05)
    outlier_data_95 = outlier_data.copy()
    outlier_data_5 = outlier_data.copy()

    # subtract the 95% and 5% quantile respectively
    for i, col in enumerate(outlier_data.columns):
        outlier_data_95[col] = outlier_data_95[col] - quantile_95[i]
        outlier_data_5[col] = outlier_data_5[col] - quantile_5[i]

    # replace colum value by a 1 if it is an outlier, or a 0 if not
    outlier_data_95[outlier_data_95 > 0] = 1
    outlier_data_95[outlier_data_95 < 0] = 0
    outlier_data_5[outlier_data_5 < 0] = 1
    outlier_data_5[outlier_data_5 > 0] = 0

    # sum the number of times a cell appears as an outlier
    outlier_metric_95 = outlier_data_95.sum(axis=1)
    outlier_metric_5 = outlier_data_5.sum(axis=1)
    outlier_metric = outlier_metric_5 + outlier_metric_95

    # plot as histogram
    fig = px.histogram(outlier_metric)
    fig.show()

    # remove those with an outlier score above 60
    data = data.drop(list(outlier_metric[outlier_metric > cutoff].index), axis=0)

    return data


def find_outliers_E55(data, cutoff):
    """Gives each cell an outlier score for each feature. If the cell is above the 95 quantile or below the 5 quantile
    for a given feature the outlier score is increased by one. If a cell has an outlier score above the threshold it
    is removed from the data. slightly different to find_outliers as E55 has data has different structure.

        Parameters
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline
        cutoff: int
            outlier score above which the cell is classed a mistake in the cell profiler cell detection

        Returns
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline with cells that do not meet the cutoff
            removed
        """

    # copy the data frame so we don't accadently make changes to the origonal
    outlier_data = data.copy()
    # drop the first 3 columns as they are metadata
    outlier_data = outlier_data.drop(outlier_data.columns[:3], axis=1)
    outlier_data = outlier_data.drop(['Metadata_Radiated', 'Rad_number'], axis=1)
    # calculate the 95% quantile
    quantile_95 = outlier_data.quantile(q=0.95)
    # calculate the 5% quantile
    quantile_5 = outlier_data.quantile(q=0.05)
    outlier_data_95 = outlier_data.copy()
    outlier_data_5 = outlier_data.copy()

    # subtract the 95% and 5% quantile respectively
    for i, col in enumerate(outlier_data.columns):
        outlier_data_95[col] = outlier_data_95[col] - quantile_95[i]
        outlier_data_5[col] = outlier_data_5[col] - quantile_5[i]

    # replace colum value by a 1 if it is an outlier, or a 0 if not
    outlier_data_95[outlier_data_95 > 0] = 1
    outlier_data_95[outlier_data_95 < 0] = 0
    outlier_data_5[outlier_data_5 < 0] = 1
    outlier_data_5[outlier_data_5 > 0] = 0

    # sum the number of times a cell appears as an outlier
    outlier_metric_95 = outlier_data_95.sum(axis=1)
    outlier_metric_5 = outlier_data_5.sum(axis=1)
    outlier_metric = outlier_metric_5 + outlier_metric_95

    # plot as histogram
    fig = px.histogram(outlier_metric)
    fig.show()

    # remove those with an outlier score above 60
    data = data.drop(list(outlier_metric[outlier_metric > cutoff].index), axis=0)

    return data


def create_new_features(data):
    """Engineer new features of interest from existing data.

        Parameters
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline

        Returns
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline with additional features
        """

    # Create new columns for rescaled P21 and LaminB1 based on cell cycle/ DAPI
    data['CorrectedIntensity_MeanIntensity_CorrP21'] = data['Intensity_MeanIntensity_CorrP21'] \
                                                       / data['Intensity_IntegratedIntensity_CorrNuclei']
    data['CorrectedIntensity_IntegratedIntensity_CorrLaminB1'] = data['Intensity_MaxIntensity_CorrLaminB1'] \
                                                                 / data['Intensity_IntegratedIntensity_CorrNuclei']

    data['CorrectedIntensity_MeanIntensity_CorrP21'] = data['Intensity_MeanIntensity_CorrP21'] \
                                                       / data['Intensity_IntegratedIntensity_CorrNuclei']
    data['CorrectedIntensity_IntegratedIntensity_CorrLaminB1'] = data['Intensity_MaxIntensity_CorrLaminB1'] \
                                                                 / data['Intensity_IntegratedIntensity_CorrNuclei']

    data['CorrectedIntensity_MeanIntensityArea_CorrP21'] = data['Intensity_MeanIntensity_CorrP21'] * data['AreaShape_Area']
    data['CorrectedIntensity_MeanIntensityArea_CorrLaminB1'] = data['Intensity_MeanIntensity_CorrLaminB1'] * data['AreaShape_Area']

    # Create new column for the maximum LaminB1 fraction
    meanfrac_lamB1 = []
    frac_lamB1 = []

    for measure in data.columns:
        # split the column names at each "_"
        split_measure = measure.split('_')
        if split_measure[0] == 'RadialDistribution' and split_measure[2] == 'CorrLaminB1':
            if split_measure[1] == 'MeanFrac':
                meanfrac_lamB1.append(measure)
            elif split_measure[1] == 'FracAtD':
                frac_lamB1.append(measure)

    data['RadialDistribution_MeanFrac_CorrLaminB1_max'] = data[meanfrac_lamB1].max(axis=1)
    # Mean fractional intensity at a given radius; calculated as fraction of total intensity normalized by
    # fraction of pixels at a given radius
    data['RadialDistribution_FracAtD_CorrLaminB1_max'] = data[frac_lamB1].max(axis=1)

    data['RadialDistribution_MaxIntensFrac_CorrLaminB1_max'] = data['RadialDistribution_FracAtD_CorrLaminB1_max'] * \
                                                               data['Intensity_IntegratedIntensity_CorrLaminB1']
    # Create column for upper quantile on lam B1 scaled by lower quartile
    data['Intensity_ScaledQuartileIntensity_CorrLaminB1'] = data['Intensity_UpperQuartileIntensity_CorrLaminB1'] / data[
        'Intensity_LowerQuartileIntensity_CorrLaminB1']

    return data


def plot_hist_with_extras(data, x_column, y_column, line, shaded, titles):
    """Plots a scatter plot of two data columns, with marignal histograms on the x and y axis, coloured by whether the
    cells were radiated or not, with the option to plot a black line or shaded region.

        Parameters
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline
        x_column: string
            name of the data column to plot on the x axis
        y_column: string
            name of the data column to plot on the y axis
        line: list
            list of coordinates of the line to be plotted [x0, y0, x1, y1]
        shaded: list
            list of coordinates of the shaded region to be plotted [x0, y0, x1, y1]
        titles: list
            list of the titles and for the plot [title, x axis title, y axis title]

        """

    fig = px.scatter(data, x=x_column, y=y_column,
                     opacity=0.2, color='Metadata_Radiated', marginal_x="histogram", marginal_y="histogram")
    fig.update_layout(
        font=dict(
            size=16,
        )
    )

    if len(line) == 4:
        fig.add_shape(type="line",
                      x0=line[0],
                      y0=line[1],
                      x1=line[2],
                      y1=line[3])

    if len(shaded) == 4:
        fig.add_shape(type="rect",
                      x0=shaded[0],
                      y0=shaded[1],
                      x1=shaded[2],
                      y1=shaded[3], fillcolor="grey", opacity=0.4, line={'width': 0})

    fig.update_layout(
        title=titles[0],
        xaxis_title=titles[1],
        yaxis_title=titles[2])

    fig.show()

    return


def plot_hist_with_extras_2(data, x_column, y_column, line, shaded, shaded_2, titles):
    """Plots a scatter plot of two data columns, with marignal histograms on the x and y axis, coloured by whether the
    cells were radiated or not, with the option to plot a black line or shaded region.

        Parameters
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline
        x_column: string
            name of the data column to plot on the x axis
        y_column: string
            name of the data column to plot on the y axis
        line: list
            list of coordinates of the line to be plotted [x0, y0, x1, y1]
        shaded: list
            list of coordinates of the shaded region to be plotted [x0, y0, x1, y1]
        titles: list
            list of the titles and for the plot [title, x axis title, y axis title]

        """

    fig = px.scatter(data, x=x_column, y=y_column,
                     opacity=0.2, color='Metadata_Radiated', marginal_x="histogram", marginal_y="histogram")
    fig.update_layout(
        font=dict(
            size=16,
        )
    )

    if len(line) == 4:
        fig.add_shape(type="line",
                      x0=line[0],
                      y0=line[1],
                      x1=line[2],
                      y1=line[3])

    if len(shaded) == 4:
        fig.add_shape(type="rect",
                      x0=shaded[0],
                      y0=shaded[1],
                      x1=shaded[2],
                      y1=shaded[3], fillcolor="grey", opacity=0.4, line={'width': 0})

    if len(shaded_2) == 4:
        fig.add_shape(type="rect",
                      x0=shaded_2[0],
                      y0=shaded_2[1],
                      x1=shaded_2[2],
                      y1=shaded_2[3], fillcolor="grey", opacity=0.4, line={'width': 0})

    fig.update_layout(
        title=titles[0],
        xaxis_title=titles[1],
        yaxis_title=titles[2])

    fig.show()

    return


def project_onto_line(data, x_column, y_column, line):
    """Projects datapoints from the specified x and y columns onto a given line. Adds columns containing this info
    to the dataframe.

        Parameters
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline
        x_column: string
            name of the data column to plot on the x axis
        y_column: string
            name of the data column to plot on the y axis
        line: list
            list of coordinates of the line to be plotted [x0, y0, x1, y1]

        Returns
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline, with added columns containing
            the projection data

        """

    x_proj = []
    y_proj = []

    line = Line(point=[line[0], line[1]], direction=[line[2] - line[0], line[3] - line[1]])
    for i in range(len(data)):
        x_coord = data.iloc[i][x_column]
        y_coord = data.iloc[i][y_column]
        point = Point([x_coord, y_coord])
        point_projected = line.project_point(point)
        x_proj.append(point_projected[0])
        y_proj.append(point_projected[1])

    data['x_proj'] = x_proj
    data['y_proj'] = y_proj

    return data


def project_onto_line_pca_return(data, x_column, y_column):
    """Projects datapoints from the specified x and y columns onto a PC1. Adds columns containing this info
    to the dataframe.

        Parameters
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline
        x_column: string
            name of the data column to plot on the x axis
        y_column: string
            name of the data column to plot on the y axis

        Returns
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline, with added columns containing
            the projection data

        """

    for_pca = data[[x_column, y_column]].copy()

    # # number of PCA components
    # pca = PCA(n_components=1)
    # # do the PCA reduction
    # components = pca.fit_transform(for_pca)
    #
    # data['x_proj'] = [0 for x in components]
    # data['y_proj'] = [x[0] for x in components]


    # or
    A = np.array(for_pca)
    print(A)
    # calculate the mean of each column
    M = np.mean(A.T, axis=1)
    print(M)
    # center columns by subtracting column means
    C = A - M
    print(C)
    # calculate covariance matrix of centered matrix
    V = np.cov(C.T)
    print(V)
    # eigendecomposition of covariance matrix
    values, vectors = eig(V)
    print("vectors", vectors)
    print("values", values)
    ##Step-4
    sorted_index = np.argsort(values)[::-1]
    sorted_eigenvalue = values[sorted_index]
    sorted_eigenvectors = vectors[:,sorted_index]
    num_components = 1
    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]

    #Step-6
    X_reduced = np.dot(eigenvector_subset.transpose() , C.transpose() ).transpose()

    print(X_reduced)

    grad = vectors[0][1]/vectors[0][0]
    c = M[1] - grad*M[0]

    data['x_proj'] = [0 for x in X_reduced]
    data['y_proj'] = [x[0] for x in X_reduced]

    return data, grad, c


def project_onto_line_pca(data, x_column, y_column):
    """Projects datapoints from the specified x and y columns onto a PC1. Adds columns containing this info
    to the dataframe.

        Parameters
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline
        x_column: string
            name of the data column to plot on the x axis
        y_column: string
            name of the data column to plot on the y axis

        Returns
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline, with added columns containing
            the projection data

        """

    for_pca = data[[x_column, y_column]].copy()

    # number of PCA components
    pca = PCA(n_components=1)
    # do the PCA reduction
    components = pca.fit_transform(for_pca)

    data['x_proj'] = [0 for x in components]
    data['y_proj'] = [x[0] for x in components]

    return data


def variance_threshold(data, threshold):
    """Filter features, removing those which do not have a large variance over all cells.

        Parameters
        -------
        data: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline
        threshold: float

        Returns
        -------
        data_filtered: pandas dataframe
            dataframe containing the output data from a cell profiler pipeline, with the features with small
            variance removed
        filtered_columns: list
            list of the features that passed the filter

        """

    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    sel = VarianceThreshold(threshold)
    sel.fit(data)
    idx = np.where(sel.variances_ > threshold)[0]

    filtered_columns = data.columns[idx]

    data_filtered = sel.fit_transform(data)

    print("Shape of filtered data")
    print(data_filtered.shape)

    return data_filtered, filtered_columns


def plot_projection(projection_type, embedding, color_by_column):
    """Plots a scatter plot of projection data

        Parameters
        -------
        projection_type: string
            string describing type of projection, eg, "TSNE"
        embedding: array
            array dta containing the embedding
        color_by_column: string
            name of the data column to colour the data by


        """

    fig = px.scatter(x=embedding[:, 0], y=embedding[:, 1], title=projection_type + " of cell image data", opacity=0.5,
                     render_mode="svg", color=color_by_column)
    fig.update_traces(marker_size=5)
    fig.update_layout(
        font=dict(
            size=20,
        )
    )
    fig.show()
    return


def plot_ordered_classifier_score(classifier_results, cell_line, classifier):
    """Plots a scatter plot of the ordered probability score assigned to each cell

        Parameters
        -------
        classifier_results: dataframe
            dataframe containing the classifier prediction probability and the "ground truth" of the test data, eg:
            pd.DataFrame([pred_probs, y_test]).T
        cell_line: string
            name of the cell line for the plot title
        classifier: string
            name of the classifier for plot title

        """

    fig = px.scatter(y=classifier_results[0], x=np.arange(len(classifier_results)), color=classifier_results[1], opacity=0.2)
    fig.update_layout(
        font=dict(
            size=20,
        )
    )
    fig.update_layout(
        title=cell_line + " " + classifier,
        xaxis_title="Ordered cells",
        yaxis_title="SVM prediction score")
    fig.show()

    return


def plot_continuous_classifier_comparison(data, x_column, classifier):
    """Plots a scatter plot of the probability score assigned to each cell from the classifier, against the projection
    of the data onto the P21/ LaminB1 senescence axis, with the line of best fit for all of the data shown in black.

        Parameters
        -------
        x_column: string
            The name of the column to plot on the x axis (one of the classification methods used)
        classifier: string
            name of the classifier used for plotting title
        """

    fig = px.scatter(data, x=x_column, y='projection', color='senescent', marginal_x="histogram",
                     marginal_y="histogram", opacity=0.2, trendline='ols', trendline_scope='overall',
                     trendline_color_override='black')
    fig.update_layout(
        font=dict(
            size=16,
        )
    )
    fig.update_layout(
        title="Comparison of " + classifier + "classifier to position on P21 and LaminB1 senescence axis",
        xaxis_title=classifier + "prediction score",
        yaxis_title="Position along P21 and LaminB1 senescence axis")

    fig.show()

    return
