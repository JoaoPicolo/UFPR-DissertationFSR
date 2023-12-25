echo '32x32'
echo 'Open Set Quiscampi'
python get_results.py quiscampi-open-set/gallery-256.bin quiscampi-open-set/downgrade-32/fold_1/ 1
python get_results.py quiscampi-open-set/gallery-256.bin quiscampi-open-set/bicubic-interpolation-32/fold_1/ 1
python get_results.py quiscampi-open-set/gallery-256.bin quiscampi-open-set/no-interpolation-32/fold_1/ 1
python get_results.py quiscampi-open-set/gallery-256.bin quiscampi-open-set/sparnet-quiscampi-lr32-sr256-scale8/fold_1/ 1
python get_results.py quiscampi-open-set/gallery-256.bin quiscampi-open-set/dic-celebA-scale8-Quiscampi-32/fold_1/ 1
python get_results.py quiscampi-open-set/gallery-256.bin quiscampi-open-set/dic-helen-scale8-Quiscampi-32/fold_1/ 1
python get_results.py quiscampi-open-set/gallery-256.bin quiscampi-open-set/dicgan-celebA-scale8-Quiscampi-32/fold_1/ 1
python get_results.py quiscampi-open-set/gallery-256.bin quiscampi-open-set/dicgan-helen-scale8-Quiscampi-32/fold_1/ 1

echo 'Closed Set Quiscampi'
python get_results.py quiscampi-closed-set/gallery-256.bin quiscampi-closed-set/downgrade-32/fold_1/ 1
python get_results.py quiscampi-closed-set/gallery-256.bin quiscampi-closed-set/bicubic-interpolation-32/fold_1/ 1
python get_results.py quiscampi-closed-set/gallery-256.bin quiscampi-closed-set/no-interpolation-32/fold_1/ 1
python get_results.py quiscampi-closed-set/gallery-256.bin quiscampi-closed-set/sparnet-quiscampi-lr32-sr256-scale8/fold_1/ 1
python get_results.py quiscampi-closed-set/gallery-256.bin quiscampi-closed-set/dic-celebA-scale8-Quiscampi-32/fold_1/ 1
python get_results.py quiscampi-closed-set/gallery-256.bin quiscampi-closed-set/dic-helen-scale8-Quiscampi-32/fold_1/ 1
python get_results.py quiscampi-closed-set/gallery-256.bin quiscampi-closed-set/dicgan-celebA-scale8-Quiscampi-32/fold_1/ 1
python get_results.py quiscampi-closed-set/gallery-256.bin quiscampi-closed-set/dicgan-helen-scale8-Quiscampi-32/fold_1/ 1

echo 'Open Set SCFace'
python get_results.py scface-open-set/gallery-256.bin scface-open-set/downgrade-32/fold_1/ 0
python get_results.py scface-open-set/gallery-256.bin scface-open-set/bicubic-interpolation-32/fold_1/ 0
python get_results.py scface-open-set/gallery-256.bin scface-open-set/no-interpolation-32/fold_1/ 0
python get_results.py scface-open-set/gallery-256.bin scface-open-set/sparnet-scface-lr32-sr256-scale8/fold_1/ 0
python get_results.py scface-open-set/gallery-256.bin scface-open-set/dic-celebA-scale8-SCFace-32/fold_1/ 0
python get_results.py scface-open-set/gallery-256.bin scface-open-set/dic-helen-scale8-SCFace-32/fold_1/ 0
python get_results.py scface-open-set/gallery-256.bin scface-open-set/dicgan-celebA-scale8-SCFace-32/fold_1/ 0
python get_results.py scface-open-set/gallery-256.bin scface-open-set/dicgan-helen-scale8-SCFace-32/fold_1/ 0

echo 'Closed Set SCFace'
python get_results.py scface-closed-set/gallery-256.bin scface-closed-set/downgrade-32/fold_1/ 0
python get_results.py scface-closed-set/gallery-256.bin scface-closed-set/bicubic-interpolation-32/fold_1/ 0
python get_results.py scface-closed-set/gallery-256.bin scface-closed-set/no-interpolation-32/fold_1/ 0
python get_results.py scface-closed-set/gallery-256.bin scface-closed-set/sparnet-scface-lr32-sr256-scale8/fold_1/ 0
python get_results.py scface-closed-set/gallery-256.bin scface-closed-set/dic-celebA-scale8-SCFace-32/fold_1/ 0
python get_results.py scface-closed-set/gallery-256.bin scface-closed-set/dic-helen-scale8-SCFace-32/fold_1/ 0
python get_results.py scface-closed-set/gallery-256.bin scface-closed-set/dicgan-celebA-scale8-SCFace-32/fold_1/ 0
python get_results.py scface-closed-set/gallery-256.bin scface-closed-set/dicgan-helen-scale8-SCFace-32/fold_1/ 0

echo '64x64'
echo 'Open Set Quiscampi'
python get_results.py quiscampi-open-set/gallery-256.bin quiscampi-open-set/downgrade-64/fold_1/ 1
python get_results.py quiscampi-open-set/gallery-256.bin quiscampi-open-set/bicubic-interpolation-64/fold_1/ 1
python get_results.py quiscampi-open-set/gallery-256.bin quiscampi-open-set/no-interpolation-64/fold_1/ 1
python get_results.py quiscampi-open-set/gallery-512.bin quiscampi-open-set/sparnet-quiscampi-lr64-sr512-scale8/fold_1/ 1
python get_results.py quiscampi-open-set/gallery-512.bin quiscampi-open-set/dic-celebA-scale8-Quiscampi-64/fold_1/ 1
python get_results.py quiscampi-open-set/gallery-512.bin quiscampi-open-set/dic-helen-scale8-Quiscampi-64/fold_1/ 1
python get_results.py quiscampi-open-set/gallery-512.bin quiscampi-open-set/dicgan-celebA-scale8-Quiscampi-64/fold_1/ 1
python get_results.py quiscampi-open-set/gallery-512.bin quiscampi-open-set/dicgan-helen-scale8-Quiscampi-64/fold_1/ 1

echo 'Closed Set Quiscampi'
python get_results.py quiscampi-closed-set/gallery-256.bin quiscampi-closed-set/downgrade-64/fold_1/ 1
python get_results.py quiscampi-closed-set/gallery-256.bin quiscampi-closed-set/bicubic-interpolation-64/fold_1/ 1
python get_results.py quiscampi-closed-set/gallery-256.bin quiscampi-closed-set/no-interpolation-64/fold_1/ 1
python get_results.py quiscampi-closed-set/gallery-512.bin quiscampi-closed-set/sparnet-quiscampi-lr64-sr512-scale8/fold_1/ 1
python get_results.py quiscampi-closed-set/gallery-512.bin quiscampi-closed-set/dic-celebA-scale8-Quiscampi-64/fold_1/ 1
python get_results.py quiscampi-closed-set/gallery-512.bin quiscampi-closed-set/dic-helen-scale8-Quiscampi-64/fold_1/ 1
python get_results.py quiscampi-closed-set/gallery-512.bin quiscampi-closed-set/dicgan-celebA-scale8-Quiscampi-64/fold_1/ 1
python get_results.py quiscampi-closed-set/gallery-512.bin quiscampi-closed-set/dicgan-helen-scale8-Quiscampi-64/fold_1/ 1

echo 'Open Set SCFace'
python get_results.py scface-open-set/gallery-256.bin scface-open-set/downgrade-64/fold_1/ 0
python get_results.py scface-open-set/gallery-256.bin scface-open-set/bicubic-interpolation-64/fold_1/ 0
python get_results.py scface-open-set/gallery-256.bin scface-open-set/no-interpolation-64/fold_1/ 0
python get_results.py scface-open-set/gallery-512.bin scface-open-set/sparnet-scface-lr64-sr512-scale8/fold_1/ 0
python get_results.py scface-open-set/gallery-512.bin scface-open-set/dic-celebA-scale8-SCFace-64/fold_1/ 0
python get_results.py scface-open-set/gallery-512.bin scface-open-set/dic-helen-scale8-SCFace-64/fold_1/ 0
python get_results.py scface-open-set/gallery-512.bin scface-open-set/dicgan-celebA-scale8-SCFace-64/fold_1/ 0
python get_results.py scface-open-set/gallery-512.bin scface-open-set/dicgan-helen-scale8-SCFace-64/fold_1/ 0

echo 'Closed Set SCFace'
python get_results.py scface-closed-set/gallery-256.bin scface-closed-set/downgrade-64/fold_1/ 0
python get_results.py scface-closed-set/gallery-256.bin scface-closed-set/bicubic-interpolation-64/fold_1/ 0
python get_results.py scface-closed-set/gallery-256.bin scface-closed-set/no-interpolation-64/fold_1/ 0
python get_results.py scface-closed-set/gallery-512.bin scface-closed-set/sparnet-scface-lr64-sr512-scale8/fold_1/ 0
python get_results.py scface-closed-set/gallery-512.bin scface-closed-set/dic-celebA-scale8-SCFace-64/fold_1/ 0
python get_results.py scface-closed-set/gallery-512.bin scface-closed-set/dic-helen-scale8-SCFace-64/fold_1/ 0
python get_results.py scface-closed-set/gallery-512.bin scface-closed-set/dicgan-celebA-scale8-SCFace-64/fold_1/ 0
python get_results.py scface-closed-set/gallery-512.bin scface-closed-set/dicgan-helen-scale8-SCFace-64/fold_1/ 0