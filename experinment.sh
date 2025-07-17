#!/bin/bash

# Re-create tartanair train/test splits
python data_splits/tartanair/get_tartanair.py

# Experinment depth anthing v2 with melo freeze after mono training
sbatch train_dav2_melo_freeze.sh