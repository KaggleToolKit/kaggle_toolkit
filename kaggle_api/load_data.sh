#!/bin/bash
kaggle competitions download titanic
mv titanic.zip ./data
cd ./data
unzip titanic.zip
rm titanic.zip