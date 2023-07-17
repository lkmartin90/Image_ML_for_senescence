#!/bin/bash

#change to correct directory

echo Running with trained SVM and processed drug discovery data

cd ./E31_drug_discovery/
python E31_LOPAC_from_pickle.py
echo Ran E31_LOPAC_from_pickle
python E31_DMSO_bootstrapping_LOPAC.py
echo Ran E31_DMSO_bootstrapping_LOPAC

python E31_targetmol_from_pickle.py
echo Ran E31_targetmol_from_pickle
python E31_DMSO_bootstrapping_targetmol.py
echo Ran E31_DMSO_bootstrapping_targetmol

cd ../E57_drug_discovery/
python E57_LOPAC_from_pickle.py
echo Ran E57_LOPAC_from_pickle
python E57_DMSO_bootstrapping_LOPAC.py
echo Ran E57_DMSO_bootstrapping_LOPAC

python E57_targetmol_from_pickle.py
echo Ran E57_targetmol_from_pickle
python E57_DMSO_bootstrapping_targetmol.py
echo Ran E57_DMSO_bootstrapping_targetmol
