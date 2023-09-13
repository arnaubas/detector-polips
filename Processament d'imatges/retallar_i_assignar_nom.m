clear all
%% Ruta de la carpeta que conté les imatges
folder = 'C:\Users\basso\OneDrive - Universitat Politècnica de Catalunya\documents\documents\UNI 4t any\2n SEMESTRE\TFG\TREBALL FOTOS\colon_normal';

%% Ruta de la carpeta on es guardaran les imatges retallades
outputFolder = 'C:\Users\basso\OneDrive - Universitat Politècnica de Catalunya\documents\documents\UNI 4t any\2n SEMESTRE\TFG\TREBALL FOTOS\colon_normal_treball';

%% Anomenar les fotos amb un número de l'1 al 1000. 
fileNames = dir(fullfile(folder, '*.jpg'));

% crear un contador
contador = 1;

% Iteració sobre cada arxiu i aplicació de la funció de retall.
for i = 1:numel(fileNames)

    [~, name, extension] = fileparts(fileNames(i).name);
    % Carregar la imatge
    img = imread(fullfile(folder, fileNames(i).name));
    
    % Aplicar la funció de retall a la imatge
    imgRecortada = editar_img_3(fullfile(folder, fileNames(i).name));
    
    % Guardar la imatge retallada a la carpeta de sortida.
    [~, baseFileName, ~] = fileparts(fileNames(i).name);
    outputFileName = fullfile(outputFolder, [num2str(contador), extension]);
    imwrite(imgRecortada, outputFileName);

    % Incrementar el contador
    contador = contador + 1;
end