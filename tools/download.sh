## Script for downloading data
mkdir -p data
cd data
# GloVe Vectors
~/bins/axel -n 32  http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip

# Questions
~/bins/axel -n 32  http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip v2_Questions_Train_mscoco.zip 
rm v2_Questions_Train_mscoco.zip

~/bins/axel -n 32  http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip v2_Questions_Val_mscoco.zip
rm v2_Questions_Val_mscoco.zip

~/bins/axel -n 32  http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip v2_Questions_Test_mscoco.zip
rm v2_Questions_Test_mscoco.zip

# Annotations
~/bins/axel -n 32  http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Train_mscoco.zip
rm v2_Annotations_Train_mscoco.zip

~/bins/axel -n 32  http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip
rm v2_Annotations_Val_mscoco.zip

# Image Features
~/bins/axel -n 32  https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
unzip trainval_36.zip
rm trainval_36.zip
