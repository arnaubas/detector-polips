clear all

%%
% Ruta de la carpeta donde guardarás las imágenes recortadas
%outputFolder = 'C:\Users\basso\OneDrive - Universitat Politècnica de Catalunya\documents\documents\UNI 4t any\2n SEMESTRE\TFG\TREBALL FOTOS\recuadre polips';
%% 

% Lee la imagen
imagen = imread(['675.jpg']);

% Muestra la imagen
imshow(imagen);

% Agrega ejes con valores de píxeles
axis on;

% Configura los valores mínimos y máximos de los ejes X e Y
% según el tamaño de la imagen
[x, y, ~] = size(imagen);
axis([1, y, 1, x]);

% Etiqueta los ejes X e Y con los valores de píxeles
xlabel('X (píxeles)');
ylabel('Y (píxeles)');
%% 

% Coordenadas del cuadro delimitador
x1 = 360;
x2 = 540;
y1 = 330;
y2 = 520;

ancho1=x2-x1
alto1=y2-y1
% Dibuja el cuadro delimitador en la imagen
imagen_con_cuadro = insertShape(imagen, 'Rectangle', [x1 y1 x2-x1 y2-y1], 'LineWidth', 4, 'Color', 'yellow');

imshow(imagen_con_cuadro);
%%

% Guarda la imagen recortada en la carpeta de salida
%outputFileName = fullfile(outputFolder, '57.jpg');
%imwrite(imagen_con_cuadro, outputFileName);
