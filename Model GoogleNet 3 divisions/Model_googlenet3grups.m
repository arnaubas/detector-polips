clear all

%% Carregar les imatges de la carpeta "Polips"
polips_imds = imageDatastore('C:\Users\basso\OneDrive - Universitat Politècnica de Catalunya\documents\documents\UNI 4t any\2n SEMESTRE\TFG\TREBALL FOTOS\Pòlips\', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%% Carregar les imatges de la carpeta "no polips"
no_polips_imds = imageDatastore('C:\Users\basso\OneDrive - Universitat Politècnica de Catalunya\documents\documents\UNI 4t any\2n SEMESTRE\TFG\TREBALL FOTOS\No_pòlips\', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%% Combinar les dues carpetes en un sol dataset
imds = imageDatastore(cat(1,polips_imds.Files,no_polips_imds.Files), ...
    'LabelSource','foldernames');

%% Dividir el conjunt de dades en entrenament (70%) i la resta (30%)
[trainImds,remainingImds] = splitEachLabel(imds,0.7);

%% Dividir el conjunt restant en validació (15%) i prova (15%)
[valImds,testImds] = splitEachLabel(remainingImds,0.5);

%% Carregar la xarxa pre-entrenada
net = googlenet;

%% Modificar la última capa per a que es puguin connectar totes les parts correctament.
capes = layerGraph(net);
[learnableLayer,classLayer] = findLayersToReplace(capes);

numClasses = numel(categories(trainImds.Labels)); % Número de classes (pòlips, no pòlips)

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

capes = replaceLayer(capes,learnableLayer.Name,newLearnableLayer);

noutipusdecapa = classificationLayer('Name','new_classoutput');
capes = replaceLayer(capes,classLayer.Name,noutipusdecapa);

%% Redimensionar les imatges a 224x224
inputSize = net.Layers(1).InputSize(1:2);
newimdsTrain = augmentedImageDatastore(inputSize(1:2),trainImds);
newimdsValidation = augmentedImageDatastore(inputSize(1:2),valImds);
newimdsTest = augmentedImageDatastore(inputSize(1:2),testImds);

%% Preparar les imatges per l'entrenament.
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
newimdsTrain = augmentedImageDatastore(inputSize(1:2),trainImds, ...
    'DataAugmentation',imageAugmenter);
newimdsTest = augmentedImageDatastore(inputSize(1:2), testImds);
newimdsValidation = augmentedImageDatastore(inputSize(1:2),valImds);

%% Definir les propietats del programa.
miniBatchSize = 10;
valFrequency = floor(numel(newimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',newimdsValidation, ... %s'activa la validació
    'ValidationFrequency',valFrequency, ...
    'Verbose',false);

%% Entrenar el programa
net = trainNetwork(newimdsTrain,capes,options);

%% Evaluar el rendiment de la xarxa amb el conjunt de dades de prova.
% Obtenir les prediccions de la xarxa amb el conjunt de dades de prova.
etiquetespredites = classify(net, newimdsTest);
% Obtenir les etiquetes reals del conjunt de dades de prova.
etiquetesreals = testImds.Labels;

%% Matriu de confusió
% Calcular la matriu de confusió
confusionMat = confusionmat(etiquetesreals, etiquetespredites);

%% Calcular exactitud, precisió, sensibilitat, especificitat i nombre F1
Exactitud = (confusionMat(1)+confusionMat(4))/(confusionMat(1)+confusionMat(2)+confusionMat(3)+confusionMat(4));
Precisio = confusionMat(1)/(confusionMat(1)+confusionMat(3));
Sensibilitat = confusionMat(1)/(confusionMat(1)+confusionMat(2));
Especificitat = confusionMat(4)/(confusionMat(4)+confusionMat(3));
NombreF1 = (2*Sensibilitat*Precisio)/(Sensibilitat+Precisio);
%% Mostrar la matriu de confusió
figure
cm = confusionchart(etiquetesreals, etiquetespredites);
cm.Title = 'Matriu de Confusió utilitzant GoogleNet amb 3 divisions';

%% Mostrar les métriques de rendiment
Exactitud
Precisio
Sensibilitat
Especificitat
NombreF1