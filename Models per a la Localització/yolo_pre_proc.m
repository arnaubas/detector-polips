function Ipreprocessed = yolo_pre_proc(I)

%roi - not used - empty
%useROI = 0
%TrainingImageSize - same as I

%Output of iFindNearestTrainingImageSize
% PreprocessedImageSize: [224 224]
%                    ScaleX: 1
%                    ScaleY: 1

Ipreprocessed = vision.internal.cnn.yolo.yolov2Datastore.normalizeImageAndCastToSingle(I);
