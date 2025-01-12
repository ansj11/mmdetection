

python demo/demo.py

pushd /gemini/data-1/evalmodels

mv /gemini/data-1/mmdetection/mask2former_accuracy_contour_* json_files/

bash run_test.sh

popd



