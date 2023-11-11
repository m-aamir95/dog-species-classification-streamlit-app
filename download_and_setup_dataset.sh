
# Download, ext
if [ ! -d "./Images" ]
then
  echo "Images dir not found, downloading dataset";
  #Download Dataset from stanford
  wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar -O images.tar;
  # Uncompressing tar ball
  tar -xvf images.tar &> /dev/null;

  mv ./Images* ./Raw_Data;
  mv ./images* ./Raw_Data;
fi