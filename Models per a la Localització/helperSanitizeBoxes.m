%helperSanitizeBoxes Sanitize box data.
% This example helper is used to clean up invalid bounding box data. Boxes
% with values <= 0 are removed and fractional values are rounded to
% integers.
%
% If none of the boxes are valid, this function passes the data through to
% enable downstream processing to issue proper errors.

% Copyright 2020 The Mathworks, Inc.

function boxes = helperSanitizeBoxes(boxes, imageSize)
persistent hasInvalidBoxes
valid = all(boxes > 0, 2);
if any(valid)
    if ~all(valid) && isempty(hasInvalidBoxes)
        % Issue one-time warning about removing invalid boxes.
        hasInvalidBoxes = true;
        warning('Removing ground truth bouding box data with values <= 0.')
    end
    boxes = boxes(valid,:);
    boxes = roundFractionalBoxes(boxes, imageSize);
end

end

