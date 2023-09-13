clear all

%% Carregar la imatge
img = imread('17.jpg');
imshow(img)

%% Assignar les coordenades d'interès i retallar.
img_cropped = imcrop(img, [48 24 625 531]);

%% Mostrar la nova imatge
imshow(img_cropped)