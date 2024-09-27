#!/bin/zsh

echo "hello"

cp -r ../../drools_py/src/drools_py ./drools_py_copied
cp -r ../../python_di/src/python_di ./python_di_copied
cp -r ../../python_util/src/python_util ./python_util_copied
cp -r ../../data/metadata_extractor/src ./metadata_extractor_copied
mkdir -p src/model_server || true
cp -r ../src/model_server ./src/model_server
mkdir resources || true
cp application-docker.yml ./resources/application.yml
cp ../docker.env ./docker.env
cp ../requirements.txt ./requirements.txt