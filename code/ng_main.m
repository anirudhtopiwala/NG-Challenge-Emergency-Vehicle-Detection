%% Anirudh Topiwala
clc; clear all;close all;
%% Setting Directory
% input = '../Input/Police Cars/';
% input = '../Input/Fire Trucks/';
% input = '../Input/Ambulance/';
input = '../Input/Test_Images/';
input = imageDatastore(input,'IncludeSubfolders',true,'LabelSource','foldernames');

%% Loading pretrained network 
net = alexnet;

%% Faster RCNN
% Options for step 1.
optionsStage1 = trainingOptions('sgdm', ... 
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 256, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

% Options for step 2.
optionsStage2 = trainingOptions('sgdm', ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

% Options for step 3.
optionsStage3 = trainingOptions('sgdm', ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 256, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

% Options for step 4.
optionsStage4 = trainingOptions('sgdm', ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

optionsfasterrcnn = [
    optionsStage1
    optionsStage2
    optionsStage3
    optionsStage4
    ];

% load('ambulance.mat')
% load('allLabels.mat')

%  fasterrcnn = trainFasterRCNNObjectDetector(labelall, net, optionsfasterrcnn, ...
%         'NegativeOverlapRange', [0 0.3], ...
%         'PositiveOverlapRange', [0.6 1], ...
%         'NumStrongestRegions',Inf,...
%         'SmallestImageDimension',400,...
%         'BoxPyramidScale', 1.2);
% save('ambulancealexnet.mat','fasterrcnn');
% save('3classalexnet.mat','fasterrcnn');

% load('../net/ambulancealexnet.mat')

load('../net/3classalexnet.mat')

%% Read test image
% load('rcnnall.mat')
filesize=size(input.Files);
resulttable=labelall;
count=0;
for i=1:1:filesize(1)
    % img= (imread('C:\Users\Anirudh\Desktop\2.png'));
    img= imread(input.Files{i});
    imgsize = size(img);
    if (imgsize(1)< 227)
        img= imresize(img,[227 imgsize(2)]);
    elseif(imgsize(2)< 227)
        img= imresize(img,[imgsize(1) 227]);
    else
        img= imresize(img,[227 227]);
    end
    %Expected results
%     n=find(strcmp(input.Files{i},labelall{:,:}));
%         if(n~=0)
%         resulttable{i,:}= labelall{n,:};   
%         else
%             count=count+1;
%         end
    % Detect stop signs
    [bboxes,score,labels] = detect(fasterrcnn,img);
    % Display the detection results
    % [score, idx] = max(score);

    % bbox = bboxes(idx, :);
    outputImage=img;
        for j=1:size(bboxes,1)
            score(j)
            % if (score(i)>0.95)
            annotation = sprintf('%s:', labels(j));
            outputImage = insertObjectAnnotation(outputImage, 'rectangle', bboxes(j,:), annotation);
            % end

        end
    % Results
    resultsStruct(i).Boxes = bboxes;
    resultsStruct(i).Scores = score;
    resultsStruct(i).Labels = labels;
    % figure
    imshow(outputImage) 
%     imwrite(outputImage,strcat(output,num2str(i),'.jpg'));


end
results = struct2table(resultsStruct);
% Extract expected bounding box locations from test data.
expectedResults = resulttable(:, 2:end);

%%


%% Evaluate the object detector using Average Precision metric.
[ap, recall, precision] = evaluateDetectionPrecision(results, expectedResults);
% Plot precision/recall curve
% figure
for i=1:5
    figure
    plot(recall{i},precision{i})
    xlabel('Recall')
    ylabel('Precision')
    grid on
    title(sprintf('Average Precision = %.2f', ap(i)))
end