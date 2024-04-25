#!/bin/bash

# Pull Intenet-available images
#docker pull r-base:4.1.0
docker pull maven:3.8.3-jdk-11-slim
docker pull numenta/nupic

# Build base image
cd 0-base-images
docker build -t registry.gitlab.hpi.de/akita/i/python3-base:0.2.5 ./python3-base
docker build -t registry.gitlab.hpi.de/akita/i/python3-torch:0.2.5 ./python3-torch
docker build -t registry.gitlab.hpi.de/akita/i/pyod:0.2.5 ./pyod
docker build -t registry.gitlab.hpi.de/akita/i/r-base ./r-base
docker build -t registry.gitlab.hpi.de/akita/i/tsmp:0.2.5 ./tsmp
docker build -t registry.gitlab.hpi.de/akita/i/java-base:0.2.5 ./java-base
docker build -t registry.gitlab.hpi.de/akita/i/python2-base:0.2.5 ./python2-base
docker build -t registry.gitlab.hpi.de/akita/i/python36-base:0.2.5 ./python36-base
docker build -t registry.gitlab.hpi.de/akita/i/r4-base:0.2.5 ./r4-base
docker build -t registry.gitlab.hpi.de/akita/i/rust-base:0.2.5 ./rust-base
docker build -t registry.gitlab.hpi.de/akita/i/timeeval-test-algorithm:0.2.5 ./timeeval-test-algorithm
cd ..

# For R-based programs
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
apt update
apt install r-base
Rscript -e 'install.packages("jsonlite")'
Rscript -e 'install.packages("tsmp")'
Rscript -e 'install.packages("PST")'
Rscript -e 'install.packages("TraMineR")'
Rscript -e 'install.packages("arules")'
Rscript -e 'install.packages("BBmisc")'

# Define the list of subfolders to ignore
ignored_folders=("01-data" "02-results" ".git" "0-base-images" "3-scripts")
univariable_algs=("novelty_svr" "phasespace_svm" "ensemble_gi" "grammarviz3" "hotsax" "ts_bitmap" "norma" "sand" "series2graph" "stamp" "stomp" "valmod" "left_stampi" "ssa" "pst" "numenta_htm" "subsequence_lof" "subsequence_if" "dwt_mlead" "fft" "sr" "s_h_esd" "dspot" "arima" "median_method" "sarima" "triple_es" "pci" "generic_rf" "generic_xgb" "tarzan" "health_esn" "ocean_wnn" "bagel" "donut" "img_embedding_cae" "sr_cnn" "subsequence_fast_mcd")

# Get the list of sub-folders within the current folder, excluding the ones to ignore
subfolders=$(find . -mindepth 1 -maxdepth 1 -type d)

# Traverse through the list of sub-folders
for folder in $subfolders; do
	# Extract the name of the sub-folder
    folder_name=$(basename "$folder")
	# Check if the folder name is the list of supported algorithms
	if [[ "${univariable_algs[*]}" =~ $folder_name ]]; then
		# Pass the sub-folder name to the desired command
		echo "$folder_name"
		docker build -t adapad_$folder_name ./$folder_name || true
		# your_command "$folder_name"  # Replace "your_command" with the actual command you want to run	
	fi
	
done
