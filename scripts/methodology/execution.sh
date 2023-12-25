# QUISCAMPI
# Separates sets
echo 'Quiscampi'
python protocol_split.py references/quiscampi/ quiscampi-scale8-example.txt quiscampi 1

# Calculates gallery for each set
python gallery_setup.py quiscampi-closed-set/reference-256/fold_1/ 256 1
python gallery_setup.py quiscampi-closed-set/reference-512/fold_1/ 512 1

python gallery_setup.py quiscampi-open-set/reference-256/fold_1/ 256 1
python gallery_setup.py quiscampi-open-set/reference-512/fold_1/ 512 1

# Calculates bin for each set
echo 'Quiscampi closed 256'
insightfacepaddle --rec_model ArcFace --build_index ./quiscampi-closed-set/gallery-256.bin --img_dir ./quiscampi-closed-set/reference-256/fold_1/ --label ./quiscampi-closed-set/gallery-256.txt
echo 'Quiscampi closed 512'
insightfacepaddle --rec_model ArcFace --build_index ./quiscampi-closed-set/gallery-512.bin --img_dir ./quiscampi-closed-set/reference-512/fold_1/ --label ./quiscampi-closed-set/gallery-512.txt

echo 'Quiscampi open 256'
insightfacepaddle --rec_model ArcFace --build_index ./quiscampi-open-set/gallery-256.bin --img_dir ./quiscampi-open-set/reference-256/fold_1/ --label ./quiscampi-open-set/gallery-256.txt
echo 'Quiscampi closed 512'
insightfacepaddle --rec_model ArcFace --build_index ./quiscampi-open-set/gallery-512.bin --img_dir ./quiscampi-open-set/reference-512/fold_1/ --label ./quiscampi-open-set/gallery-512.txt

# SCFACE
# Separates sets
echo 'SCFace'
python protocol_split.py references/scface/ scface-scale8-example.txt scface 0

# Calculates gallery for each set
python gallery_setup.py scface-closed-set/reference-256/fold_1/ 256 0
python gallery_setup.py scface-closed-set/reference-512/fold_1/ 512 0

python gallery_setup.py scface-open-set/reference-256/fold_1/ 256 0
python gallery_setup.py scface-open-set/reference-512/fold_1/ 512 0

# Calculates bin for each set
echo 'SCFace closed 256'
insightfacepaddle --rec_model ArcFace --build_index ./scface-closed-set/gallery-256.bin --img_dir ./scface-closed-set/reference-256/fold_1/ --label ./scface-closed-set/gallery-256.txt
echo 'SCFace closed 512'
insightfacepaddle --rec_model ArcFace --build_index ./scface-closed-set/gallery-512.bin --img_dir ./scface-closed-set/reference-512/fold_1/ --label ./scface-closed-set/gallery-512.txt

echo 'SCFace open 256'
insightfacepaddle --rec_model ArcFace --build_index ./scface-open-set/gallery-256.bin --img_dir ./scface-open-set/reference-256/fold_1/ --label ./scface-open-set/gallery-256.txt
echo 'SCFace closed 512'
insightfacepaddle --rec_model ArcFace --build_index ./scface-open-set/gallery-512.bin --img_dir ./scface-open-set/reference-512/fold_1/ --label ./scface-open-set/gallery-512.txt