clear all

%% Unzip les imatges i la taula amb les coordenades dels quadrats grocs.
unzip imatges.zip
data = load('etiquetas_polipos_120.mat');
etiquetasbo = data.etiquetasbo;

%% passar les cel·les a taula.
% Convierte la celda en una tabla
etiquetasbo = cell2table(etiquetasbo, 'VariableNames', {'imageFilename', 'boundingBox'});

% Concatena los valores de [x, y, ancho, altura] en una misma celda
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

%% Llegir i mostrar un exemple d'imatge

data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%% Escollir el model de detecció
net = resnet50();
lgraph = layerGraph(net);

%% Ajustar la mida de la imatge
inputSize = [224 224 3];
classNames = {'boundingBox'};

%% Trobar la capa 'activation_40_relu'
idx = find(ismember({lgraph.Layers.Name},'activation_40_relu'));

% Eliminar totes les capes després de 'activation_40_relu'
removedLayers = {lgraph.Layers(idx+1:end).Name};
ssdLayerGraph = removeLayers(lgraph,removedLayers);

weightsInitializerValue = 'glorot';
biasInitializerValue = 'zeros';

% Afegir capes extra a sobre de la xarxa bàsica.
extraLayers = [];

% Afegir la capa conv6_1
filterSize = 1;
numFilters = 256;
numChannels = 1024;
conv6_1 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Name = 'conv6_1', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu6_1 = reluLayer(Name = 'relu6_1');
extraLayers = [extraLayers; conv6_1; relu6_1];

% Afegir la capa conv6_2
filterSize = 3;
numFilters = 512;
numChannels = 256;
conv6_2 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Padding = iSamePadding(filterSize), ...
    Stride = [2, 2], ...
    Name = 'conv6_2', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu6_2 = reluLayer(Name = 'relu6_2');
extraLayers = [extraLayers; conv6_2; relu6_2];

% Afegir la capa conv7_1
filterSize = 1;
numFilters = 128;
numChannels = 512;
conv7_1 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Name = 'conv7_1', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu7_1 = reluLayer(Name = 'relu7_1');
extraLayers = [extraLayers; conv7_1; relu7_1];

% Afegir la capa conv7_2
filterSize = 3;
numFilters = 256;
numChannels = 128;
conv7_2 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Padding = iSamePadding(filterSize), ...
    Stride = [2, 2], ...
    Name = 'conv7_2', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu7_2 = reluLayer(Name = 'relu7_2');
extraLayers = [extraLayers; conv7_2; relu7_2];

% Afegir la capa conv8_1 
filterSize = 1;
numFilters = 128;
numChannels = 256;
conv8_1 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Name = 'conv8_1', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu8_1 = reluLayer(Name = 'relu8_1');
extraLayers = [extraLayers; conv8_1; relu8_1];

% Afegir la capa conv8_2
filterSize = 3;
numFilters = 256;
numChannels = 128;
conv8_2 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Name = 'conv8_2', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu8_2 = reluLayer(Name ='relu8_2');
extraLayers = [extraLayers; conv8_2; relu8_2];

% Afegir la capa conv9_1
filterSize = 1;
numFilters = 128;
numChannels = 256;
conv9_1 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Padding = iSamePadding(filterSize), ...
    Name = 'conv9_1', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu9_1 = reluLayer('Name', 'relu9_1');
extraLayers = [extraLayers; conv9_1; relu9_1];

if ~isempty(extraLayers)
    lastLayerName = ssdLayerGraph.Layers(end).Name;
    ssdLayerGraph = addLayers(ssdLayerGraph, extraLayers);
    ssdLayerGraph = connectLayers(ssdLayerGraph, lastLayerName, extraLayers(1).Name);
end

detNetworkSource = ["activation_22_relu", "activation_40_relu", "relu6_2", "relu7_2", "relu8_2"];

%% Definir les 'anchorBoxes'
anchorBoxes = {[60,30;30,60;60,21;42,30];...
               [111,60;60,111;111,35;64,60;111,42;78,60];...
               [162,111;111,162;162,64;94,111;162,78;115,111];...
               [213,162;162,213;213,94;123,162;213,115;151,162];...
               [264,213;213,264;264,151;187,213]};

%% Definir el mètode SSD
detector = ssdObjectDetector(ssdLayerGraph,classNames,anchorBoxes,DetectionNetworkSource=detNetworkSource,InputSize=inputSize,ModelName='ssdVehicle'); 

%% Augmentar les dades d'entrenament
augmentedTrainingData = transform(trainingData, @yolo_augmentData);
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},rectangle = data{2});
    reset(augmentedTrainingData);
end

figure
montage(augmentedData,BorderSize = 10)
%% Assignar les dades d'entrenament preprocessades. 
preprocessedTrainingData = transform(augmentedTrainingData,@(data)yolo_preprocessData(data,inputSize));
data = read(preprocessedTrainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%% Opcions d'entrenament
options = trainingOptions('sgdm', ...
        MiniBatchSize = 10, ....
        InitialLearnRate = 1e-3, ...
        LearnRateSchedule = 'piecewise', ...
        LearnRateDropPeriod = 30, ...
        LearnRateDropFactor =  0.8, ...
        MaxEpochs = 20, ...
        VerboseFrequency = 50, ...        
        CheckpointPath = tempdir, ...
        Shuffle = 'every-epoch');

%% Entrenar el programa SSD
 % Entrenar el detector SSD.
 [detector, info] = trainSSDObjectDetector(preprocessedTrainingData,detector,options);
 
 %% Anotar el resultat
% Iniciar una llista per a les puntuacions de confiança. 
confidenceScores = [];

% Llegir les imatges de prova. 
numTestImages = numel(testDataTbl.imageFilename);
for i = 1:numTestImages
    % Llegir una imatge dde prova i redimensionar-la
    I = imread(testDataTbl.imageFilename{i});
    I = imresize(I, inputSize(1:2));
    
    % Detectar polips a la imatge
    [bboxes, scores] = detect(detector, I);
    
    % Agregar les puntuacions de confiança a la llista
    confidenceScores = [confidenceScores; scores];
    
    % Mostrar la imatge amb els recuadres i els valors de confiança. 
    I_new = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
    figure
    imshow(I_new)
end

confidenceScores
% Calcular la mitjana dels valors de confiança.
medianConfidence = median(confidenceScores);

%% Mostrar les imatges resultat en un collaige.
% Crear una cel·la per emmagatzemar totes les imatges resultat. 
resultImages = cell(numTestImages, 1);

% Llegir les imatges de prova i processar-les
for i = 1:numTestImages
    I = imread(testDataTbl.imageFilename{i});
    I = imresize(I, inputSize(1:2));
    
    [bboxes, scores] = detect(detector, I);
    
    I_new = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
    
    resultImages{i} = I_new;
end

% Mostrar totes les imatges en un collaige. 
figure;
montage(resultImages, 'Size', [6, 6]);
