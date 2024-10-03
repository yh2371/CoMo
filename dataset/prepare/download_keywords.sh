cd ./dataset/HumanML3D

echo -e "Downloading HumanML3D keywords..."
gdown 12Hvo11kBD_gI-EpHG9iYfsBenX78o-t4
unzip keywords.zip
echo -e "Cleaning\n"
rm keywords.zip

echo -e "Downloading HumanML3D embeddings..."
gdown 14VrliBsOMXUiSIfkRq6EQ6JRMMdHrVlf
unzip keyword_embeddings.zip
echo -e "Cleaning\n"
rm keyword_embeddings.zip

echo -e "Downloading done!"