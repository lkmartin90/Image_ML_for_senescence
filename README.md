# Image_ML_for_senescence
Machine learning to detect senescent glioblastoma cells and to find compounds which induce senescence.

Functions called by other files are found in "functions.py" and should be commented well enough that they are self-explanatory. 

E31_LaminB1_P21_analysis.ipynb is a notebook with a step-by-step explanation of our analysis of our data for cell line E31. 

## File structure: 

### E31_drug_discovery:
  * E31_050423_P21_LaminB1DilatedNuclei_1.csv.zip
  * E31_050423_P21_LaminB1NucleiObject.csv.zip
  * E53_220323_P21_LaminB1Image.csv
### E31_LaminB1_P21_data:
  * E31_LOPAC_data_processing.py
    - Code to process the raw output data from the cell Profiler pipeline (not published), removing unnecessary data including stains other than the DAPI
  * E31_LOPAC_from_pickle.py
    - Takes the output of "E31_LOPAC_data_processing.py" and "E31_SVM_model.sav", to apply our SVM to the LOPAC drug discovery data.
  * E31_SVM_model.sav
    - Pickled SVM model trained on the LaminB1 and P21 data.
  * E31_train_SVM.py
    - Code to train SVM on the LaminB1 and P21 data, producing "E31_SVM_model.sav".
