clear all

%% Carregar la imatge
img = imread('17.jpg');
imshow(img);

%% Convertir la imatge a blanc i negre i emplenar els forats en les àrees brillants

bw = im2bw(img, 0.8);
bw_clean = imfill(bw, 'holes'); 
imshow(bw_clean)

%% Aplicar una màscara a la imatge original per a eliminar les àrees blanques. 

img_clean = bsxfun(@times, img, cast(~bw_clean, class(img))); % aplica la máscara a la imagen original para eliminar las áreas blancas

%% Emplenar els forats en cadascún dels canals de color i combinar-ho en una sola imatge. 

img = img_clean;


red_filled = imfill(img(:,:,1), 'holes');
green_filled = imfill(img(:,:,2), 'holes');
blue_filled = imfill(img(:,:,3), 'holes');


img_filled = cat(3, red_filled, green_filled, blue_filled);

%% Retallar els marcs negres. 
img_cropped = imcrop(img_filled, [48 24 625 531]);

%% Mostrar la imatge sense reflexe especular i sense marc negre
imshow(img_cropped);


