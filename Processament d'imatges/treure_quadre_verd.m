clear all

%% Carregar la imatge
img = imread('6.jpg');
imshow(img);

%% Segmentar la imatge per detectar els píxels verds
mask = zeros(size(img,1), size(img,2)); % Inicializar la máscara con ceros
for i = 1:size(img,1)
    for j = 1:size(img,2)
        if img(i,j,2) > 90 && img(i,j,1) < 150 && img(i,j,3) > 100 % Condición para el verde turquesa
            mask(i,j) = 1;% Asignar valor 1 a la máscara en la posición (i,j)
        else 
            mask(i,j) = 0;
        end
    end
end


%% Eliminar forats i petits objectes a la màscara
mask_clean = imfill(mask, 'holes');
mask_clean = bwareaopen(mask_clean, 1000);

%% Trobar les coordenades dels píxels verds a la màscara
[row, col] = find(mask_clean);

%% Trobar les coordenades del rectangle que encercla els píxels verds
x = min(col);
y = min(row);
w = max(col) - x;
h = max(row) - y;
a = max(col);
b = max(row);

%%  Eliminar el rectangle dels píxels verds.
img_sin_azul = bsxfun(@times, img, cast(~mask,class(img)));

%% Eliminar el marc negre.
img_cropped = imcrop(img_sin_azul, [48 24 625 531]);

%% Mostrar la imatge final sense píxels verds ni marc negre.
imshow(img_cropped)
