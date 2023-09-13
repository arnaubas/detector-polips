clear all
%% Unzip les imatges i les cel·les amb les coordenades del recuadre groc.
unzip imatges.zip
data = load('etiquetas_polipos_120.mat');
etiquetasbo = data.etiquetasbo;

%% passar les cel·les a taula
etiquetasbo = cell2table(etiquetasbo, 'VariableNames', {'imageFilename', 'boundingBox'});

% Concatenar els valors de [x, y, amplada, altura] en una mateixa cel·la.
etiquetasbo.boundingBox = mat2cell(etiquetasbo.boundingBox, ones(size(etiquetasbo, 1), 1), 4);
etiquetasbo.imageFilename = fullfile(pwd,etiquetasbo.imageFilename);



%% dividir 70% - 30%

rng(0);
shuffledIndices = randperm(height(etiquetasbo));
idx = floor(0.7 * length(shuffledIndices) );
trainingDataTbl = etiquetasbo(shuffledIndices(1:idx),:);
testDataTbl = etiquetasbo(shuffledIndices(idx+1:end),:);

%% Crear magatzem de dades

imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'boundingBox'));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'boundingBox'));

%% Combinar la informació. 

trainingData = combine(imdsTrain,bldsTrain);
testData = combine(imdsTest,bldsTest);

%% Ajustar la mida de la imatge
inputSize = [224 224 3];

%% Transformar les dades de prova. 
testData = transform(testData, @(data) yolo_preprocessData(data, inputSize));

%% Definir el número d'objectes a detectar
numClasses = width(etiquetasbo)-1;

%% Estimar les 'anchorBoxes'
trainingDataForEstimation = transform(trainingData,@(data)yolo_preprocessData(data,inputSize));
numAnchors = 5;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);

%% Escollir el model de CNN per entrenar
featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';

%% Yolo v2 detection model
lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

%% Ajustar la mida de la imatge
inputSize = [224 224 3];

%% augmentar les dades d'entrenament
augmentedTrainingData = transform(trainingData,@yolo_augmentData);

%% preprocessar les dades d'entrenament
preprocessedTrainingData = transform(augmentedTrainingData,@(data)yolo_preprocessData(data,inputSize));
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%% opcions d'entrenament
options = trainingOptions('sgdm', ...
        'MiniBatchSize', 10, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',20,...
        'CheckpointPath', tempdir, ...
        'Shuffle','never');

%% Entrenar el mètode YOLO v2. 
[detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);

% Iniciar una llista per emmagatzemar les puntuacions de confiança. 
confidenceScores = [];

% Llegir les imatges de prova.
numTestImages = numel(testDataTbl.imageFilename);
for i = 1:numTestImages
    % Llegir una imatge de prova i redimensionar-la.
    I = imread(testDataTbl.imageFilename{i});
    I = imresize(I, inputSize(1:2));
    
     % Detectar pòlips a la imatge.
    [bboxes, scores] = detect(detector, I);
    
    % Agregar les puntuacions de confiança a la llista.
    confidenceScores = [confidenceScores; scores];
    
   % Mostrar la imatge amb els quadrats i les puntuacions només si es detecten objectes
    I_new = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
    figure
    imshow(I_new)
end

confidenceScores

%% Resultats
% Calcular la mediana de les puntuacions de confiança
medianConfidence = median(confidenceScores);

%Gráfica para las primeras 18 imágenes
figure('Position', [100, 100, 1200, 400]);

numRows = 2;
numCols = 2;
spacing = 0.02;

for i = 1:4
    I = imread(testDataTbl.imageFilename{i});
    I = imresize(I, inputSize(1:2));
    
    [bboxes, scores] = detect(detector, I);
    
    confidenceScores = [confidenceScores; scores];
    
    I_new = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
    
    ax = subplot(numRows, numCols, i);
    imshow(I_new);
    ax.Position(3:4) = ax.Position(3:4) - spacing;
end

% Gràfica per les últimes 18 imatges

confidenceScores

medianConfidence = median(confidenceScores);

%%
figure;

numTestImages = numel(testDataTbl.imageFilename);
numRows = 6;
numCols = 6;

for i = 1:numTestImages
    I = imread(testDataTbl.imageFilename{i});
    I = imresize(I, inputSize(1:2));
    
    [bboxes, scores] = detect(detector, I);
    
    confidenceScores = [confidenceScores; scores];
    
    I_new = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
    
    subplot(numRows, numCols, i);
    imshow(I_new);
end

confidenceScores

medianConfidence = median(confidenceScores);

%%
% Crear una cel·la per emmagatzemar les imatges resultat.
resultImages = cell(numTestImages, 1);

% Llegir les imatges de prova i processar-les.
for i = 1:numTestImages
    I = imread(testDataTbl.imageFilename{i});
    I = imresize(I, inputSize(1:2));
    
    [bboxes, scores] = detect(detector, I);
    
    I_new = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
    
    resultImages{i} = I_new;
end

% Mostrar totes les imatges resultants en un collaige.
figure;
montage(resultImages, 'Size', [6, 6]);