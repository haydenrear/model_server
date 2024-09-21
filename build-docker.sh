cp -r ../drools_py/src/drools_py ./drools_py_copied
cp -r ../python_di/src/python_di ./python_di_copied
cp -r ../python_util/src/python_util ./python_util_copied
cp -r ../data/metadata_extractor/src ./metadata_extractor_copied
docker build -t localhost:5001/model-server-made .
docker push localhost:5001/model-server-made
rm -rf ./drools_py_copied
rm -rf ./python_di_copied
rm -rf ./python_util_copied
rm -rf ./metadata_extractor_copied
