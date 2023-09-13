function [bboxes, scores, labels] = yolo_post_proc(predictions, I, anchorBoxes, classnames)
% from yolov2objectdetector class

%default params
params.MinSize=[1 1];
params.MaxSize=[224 224];
params.Threshold=0.5;
params.FractionDownsampling=1;
params.FilterBboxesFunctor = vision.internal.cnn.utils.FilterBboxesFunctor;

PreprocessedImageSize=size(I);
ScaleX=1;
ScaleY=1;

%anchorBoxes
outputPrediction = iPredictUsingFeatureMap(predictions, params.Threshold, PreprocessedImageSize, anchorBoxes, params.FractionDownsampling);
if ~isempty(outputPrediction)
    
    bboxesX1Y1X2Y2 = outputPrediction(:,1:4);
    scorePred = outputPrediction(:,5);
    classPred = outputPrediction(:,6);
    
    % ClipBoxes to boundaries.
    bboxesX1Y1X2Y2 = iClipBBox(bboxesX1Y1X2Y2, PreprocessedImageSize);
    
    % Scale boxes back to size(Iroi).
    bboxesX1Y1X2Y2 = vision.internal.cnn.boxUtils.scaleX1X2Y1Y2(bboxesX1Y1X2Y2, ScaleX, ScaleY);
    
    % Convert [x1 y1 x2 y2] to [x y w h].
    bboxPred = vision.internal.cnn.boxUtils.x1y1x2y2ToXYWH(bboxesX1Y1X2Y2);
    
    % Filter boxes based on MinSize, MaxSize.
    [bboxPred, scorePred, classPred] = filterBBoxes(params.FilterBboxesFunctor,...
        params.MinSize,params.MaxSize,bboxPred,scorePred,classPred);
    
    % Apply NMS.
    
    [bboxes, scores, labels] = selectStrongestBboxMulticlass(bboxPred, scorePred, classPred ,...
            'RatioType', 'Union', 'OverlapThreshold', 0.5);
        
    % Apply ROI offset
%    bboxes(:,1:2) = vision.internal.detector.addOffsetForROI(bboxes(:,1:2), params.ROI, params.UseROI);
    
    % Convert classId to classNames.
    labels = classnames(1,labels);
    labels = categorical(cellstr(labels))';
    
else
    bboxes = zeros(0,4,'single');
    scores = zeros(0,1,'single');
    labels = categorical(cell(0,1),cellstr(classnames));
end

end

function outputPrediction = iPredictUsingFeatureMap(featureMap, threshold, preprocessedImageSize, anchorBoxes, fractionDownsampling)

gridSize = size(featureMap);

featureMap = permute(featureMap,[2 1 3 4]);
featureMap = reshape(featureMap,gridSize(1)*gridSize(2),gridSize(3),1,[]);
featureMap = reshape(featureMap,gridSize(1)*gridSize(2),size(anchorBoxes,1),gridSize(3)/size(anchorBoxes,1),[]);
featureMap = permute(featureMap,[2 3 1 4]);

% This is to maintain backward compatibility with version 1 detectors.
if fractionDownsampling
    downsampleFactor = preprocessedImageSize(1:2)./gridSize(1:2);
else
    downsampleFactor = floor(preprocessedImageSize(1:2)./gridSize(1:2));
end

% Scale anchor boxes with respect to feature map size
anchorBoxes = anchorBoxes./downsampleFactor;

% Extract IoU, class probabilities from feature map.
iouPred = featureMap(:,1,:,:);
sigmaXY = featureMap(:,2:3,:,:);
expWH = featureMap(:,4:5,:,:);
probPred = featureMap(:,6:end,:,:);

% Compute bounding box coordinates [x,y,w,h] with respect to input image
% dimension.
boxOut = nnet.internal.cnn.layer.util.yoloPredictBBox(sigmaXY, expWH, anchorBoxes, gridSize(1:2), downsampleFactor);

boxOut = permute([boxOut,iouPred,probPred],[2 1 3 4]);
boxOut = reshape(boxOut,size(boxOut,1),[]);
boxOut = permute(boxOut,[2 1 3 4]);

% Extract box coordinates, iou, class probabilities.
bboxesX1Y1X2Y2 = boxOut(:,1:4);
iouPred = boxOut(:,5);
probPred = boxOut(:,6:end);
[imax,idx] = max(probPred,[],2);
confScore = iouPred.*imax;
boxOut = [bboxesX1Y1X2Y2,confScore,idx];
outputPrediction = boxOut(confScore>=threshold,:);
end

function clippedBBox = iClipBBox(bbox, imgSize)

clippedBBox  = double(bbox);

x1 = clippedBBox(:,1);
y1 = clippedBBox(:,2);

x2 = clippedBBox(:,3);
y2 = clippedBBox(:,4);

x1(x1 < 1) = 1;
y1(y1 < 1) = 1;

x2(x2 > imgSize(2)) = imgSize(2);
y2(y2 > imgSize(1)) = imgSize(1);

clippedBBox = [x1 y1 x2 y2];
end

function varargout = filterBBoxes(this, minSize, maxSize, varargin)
            assert(nargin - 3 == nargout);

%             if ~isempty(this.PreprocessFunction)
%                 [varargin{1:nargout}] = this.PreprocessFunction(minSize,maxSize,varargin{:});
%             end

            [varargin{1:nargout}] = vision.internal.cnn.utils.FilterBboxesFunctor.filterSmallBBoxes(minSize, varargin{:});
            [varargin{1:nargout}] = vision.internal.cnn.utils.FilterBboxesFunctor.filterLargeBBoxes(maxSize, varargin{:});

           
            varargout = varargin;
end
