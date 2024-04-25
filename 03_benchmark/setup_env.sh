#!/bin/bash

git clone https://github.com/HPI-Information-Systems/TimeEval-algorithms
cp 00_used_for_setup/* TimeEval-algorithms/
cd TimeEval-algorithms
chmod +x setup.sh
./setup.sh