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

%% Dividir el conjunt de dades en entrenament i prova.
[trainImds,TestImds] = splitEachLabel(imds,0.7,0.3,'randomized');

%% Carregar la xarxa no pre-entrenada
capes = inceptionv3('Weights','none');

%% Modificar la última capa per a que es puguin connectar totes les parts correctament.
numClasses = numel(categories(trainImds.Labels)); % Número de clases (polipos, sin polipos)
newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
newClassLayer = classificationLayer('Name','new_classoutput');
capes = replaceLayer(capes,'fc1000',newLearnableLayer);
capes = replaceLayer(capes,'ClassificationLayer_fc1000',newClassLayer)

%% Redimensionar les imatges a 224x224
inputSize = lgraph.Layers(1).InputSize(1:2);
newimdsTrain = augmentedImageDatastore(inputSize(1:2),trainImds);
newimdsTest = augmentedImageDatastore(inputSize(1:2), TestImds);

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
newimdsTest = augmentedImageDatastore(inputSize(1:2), TestImds);

%% Definir les propietats del programa.
miniBatchSize = 10;
valFrequency = floor(numel(newimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false);

%% Entrenar el programa
net = trainNetwork(newimdsTrain,capes,options);

%% Evaluar el rendiment de la xarxa amb el conjunt de dades de prova.
% Obtenir les prediccions de la xarxa amb el conjunt de dades de prova.
etiquetespredites = classify(net, newimdsTest);
% Obtenir les etiquetes reals del conjunt de dades de prova.
etiquetesreals = TestImds.Labels;

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
cm.Title = 'Matriu de Confusió utilitzant Inception v3 no pre-entrenat';
%% Mostrar les métriques de rendiment
Exactitud
Precisio
Sensibilitat
Especificitat
NombreF1
