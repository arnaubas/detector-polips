clear all
%%
%etiquetasbo = {};  % Inicializar la matriz de etiquetas vacía

%%
% Define las coordenadas de la caja delimitadora en formato [x y ancho alto].
x = 360;       % Coordenada x de la esquina superior izquierda de la caja delimitadora
y = 330;       % Coordenada y de la esquina superior izquierda de la caja delimitadora
ancho = 180;   % Ancho de la caja delimitadora
alto = 190;    % Alto de la caja delimitadora

% Crea la ruta de archivo completa de la imagen
imageFilename = fullfile(['imatges\675.jpg']);

% Crea una matriz de etiquetas que contenga la ruta de archivo y las coordenadas de la caja delimitadora.
nuevasEtiquetas = {imageFilename, [x y ancho alto]};

% Concatena las nuevas etiquetas con las existentes utilizando la función vertcat.
etiquetasbo = vertcat(etiquetasbo, nuevasEtiquetas);

% Guarda las etiquetas en un archivo, por ejemplo, 'etiquetas_polipos.mat'.
save('etiquetas_polipos_bo20.mat', 'etiquetasbo');