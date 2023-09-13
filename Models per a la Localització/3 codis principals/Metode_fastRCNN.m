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

%%

data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%% Ajustar la mida de la imatge
inputSize = [224 224 3];

%% Nombre d'objectes a detectar.
numClasses = width(etiquetasbo)-1;

%% Estimar les 'anchorboxes'.
trainingDataForEstimation = transform(trainingData,@(data)yolo_preprocessData(data,inputSize));
numAnchors = 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);

%% Escollir el model de CNN per a l'entrenament.
featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';

%% Escollir el model de detecció
lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

%% Augmentar les dades.
augmentedTrainingData = transform(trainingData,@yolo_augmentData);


%% Preprocessament de les dades d'entrenament.
trainingData = transform(augmentedTrainingData,@(data)yolo_preprocessData(data,inputSize));
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%% Opcions d'entrenament
options = trainingOptions('sgdm',...
    'MaxEpochs',10,...
    'MiniBatchSize',2,...
    'InitialLearnRate',1e-3,...
    'CheckpointPath',tempdir);

%% Entrenament del model.
[detector, info] = trainFasterRCNNObjectDetector(trainingData,lgraph,options, ...
        'NegativeOverlapRange',[0 0.3], ...
        'PositiveOverlapRange',[0.6 1]);

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

if ~isempty(bboxes)
    I_new = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
    figure
    imshow(I_new)
else
    disp('No se detectaron objetos en la imagen.')
end
end

%% Resultats

confidenceScores
% Calcular la mediana de las puntuacions de confiança
medianConfidence = median(confidenceScores);

% Crear una cel·la per emmagatzemar totes les imatges resultat.
resultImages = cell(numTestImages, 1);

% Llegir les imatges de prova i processar-les. 
for i = 1:numTestImages
    I = imread(testDataTbl.imageFilename{i});
    I = imresize(I, inputSize(1:2));
    
    [bboxes, scores] = detect(detector, I);
    
    % Verificar si s'han detectat objectes a la imatge.
    if ~isempty(bboxes)
        I_new = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
    else
        I_new = I; % Mantenir la imatge original si no es detecten objectes.
        disp('No se detectaron objetos en la imagen.');
    end
    
    resultImages{i} = I_new;
end

% Mostrar totes les imatges resultat en un collaige. 
figure;
montage(resultImages, 'Size', [6, 6]);
