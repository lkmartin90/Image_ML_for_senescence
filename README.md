# Image_ML_for_senescence
Machine learning to detect senescent glioblastoma cells and to find compounds which induce senescence.

Functions called by other files are found in "functions.py" and should be commented well enough that they are self-explanatory. 

E31_LaminB1_P21_analysis.ipynb is a notebook with a step-by-step explanation of our analysis of our data for cell line E31. 

## File structure: 

### E31_LaminB1_P21_data:
  * E31_050423_P21_LaminB1DilatedNuclei_1.csv.zip
    - Data from the CellProfiler pipeline, with the slightly dilated nuclear size to capture the LaminB1 ring. Includes features for each cell in the fluorescence microscopy images. 
  * E31_050423_P21_LaminB1NucleiObject.csv.zip
    - Data from the Cellprofiler pipeline, from the undilated cell detection, containing accurate cell morphology info and DAPI stain. Includes features for each cell in the fluorescence microscopy images.
  * E31_050423_P21_LaminB1Image.csv
     - data from the CellProfiler pipeline describing a full image, including background stain intensities. 
### E31_drug_discovery:
  * E31_LOPAC_data_processing.py
    - Code to process the raw output data from the cell Profiler pipeline (not published), removing unnecessary data including stains other than the DAPI. Outputs *E31_LOPAC_data.csv*, which is too large to share via github.  
  * E31_LOPAC_from_pickle.py
    - Takes *E31_LOPAC_data.csv* and *E31_SVM_model.sav*, to apply our SVM to the LOPAC drug discovery data. Produces *E31_senscore_LOPAC.csv* (contains the senescence score associated with each compound in the LOPAC dataset, and the fraction of cells predicted to be senescent) and *E31_LOPAC_full_data.csv* (too big to share on GitHub, contains *E31_LOPAC_data.csv*, including senescence prediction for each individual cell). 
  * E31_SVM_model.sav
    - Pickled SVM model trained on the LaminB1 and P21 data.
  * E31_train_SVM.py
    - Code to train SVM on the LaminB1 and P21 data, producing *E31_SVM_model.sav*.
  * E31_senscore_LOPAC.csv
    - File containing the senescence score associated with each compound in the LOPAC dataset, and the fraction of cells predicted to be senescent.
  * E31_DMSO_bootstrapping_LOPAC.py
    - Takes as input *E31_senscore_LOPAC.csv* and *E31_LOPAC_full_data.csv*. Bootstraps the DMSO controls for each plate to find significant senescence scores at each cell number. Outputs "E31_LOPAC_with_bootstrap.csv".

# Pipeline:

* Run *E31_train_SVM.py*, which takes as input the data in file *E31_drug_discovery*. This produces *E31_SVM_model.sav*, a pickled version of the SVM model trained on our LaminB1 and P21 data. For a detailed version of the analysis in this script, see *E31_LaminB1_P21_analysis.ipynb*.
* Run *E31_LOPAC_from_pickle*, must have input files *E31_LOPAC_data.csv* and *E31_SVM_model.sav*, outputs *E31_senscore_LOPAC.csv* and *E31_LOPAC_full_data.csv*.
* Run *E31_DMSO_bootstrapping_LOPAC.py*, must have input files *E31_senscore_LOPAC.csv* and *E31_LOPAC_full_data.csv*, outputs "E31_LOPAC_with_bootstrap.csv".
* Run *E31_LOPAC_datavis.ipynb* to visualise data, must have input file *E31_LOPAC_with_bootstrap.csv*.
