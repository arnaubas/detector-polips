data = load('etiquetas_polipos_bobo.mat');
etiquetasbo = data.etiquetasbo;
etiquetasbo = cell2table(etiquetasbo, 'VariableNames', {'imageFilename', 'boundingBox'});

% Concatena los valores de [x, y, ancho, altura] en una misma celda
etiquetasbo.boundingBox = mat2cell(etiquetasbo.boundingBox, ones(size(etiquetasbo, 1), 1), 4);
%%
dataDir = fullfile(toolboxdir('vision'),'visiondata');
etiquetasbo.imageFilename = fullfile(dataDir,etiquetasbo.imageFilename);

%%
summary(etiquetasbo)

%%
allBoxes = vertcat(etiquetasbo.boundingBox{:});
%%
aspectRatio = allBoxes(:,3) ./ allBoxes(:,4);
area = prod(allBoxes(:,3:4),2);

figure
scatter(area,aspectRatio)
xlabel("Box Area")
ylabel("Aspect Ratio (width/height)");
title("Box Area vs. Aspect Ratio")

%%
trainingData = boxLabelDatastore(etiquetasbo(:,2:end));

%%
numAnchors = 5;
[anchorBoxes,meanIoU] = estimateAnchorBoxes(trainingData,numAnchors);
anchorBoxes

%% 
meanIoU
%%
maxNumAnchors = 15;
meanIoU = zeros([maxNumAnchors,1]);
anchorBoxes = cell(maxNumAnchors, 1);
for k = 1:maxNumAnchors
    % Estimate anchors and mean IoU.
    [anchorBoxes{k},meanIoU(k)] = estimateAnchorBoxes(trainingData,k);    
end

figure
plot(1:maxNumAnchors,meanIoU,'-o')
ylabel("Mean IoU")
xlabel("Number of Anchors")
title("Number of Anchors vs. Mean IoU")