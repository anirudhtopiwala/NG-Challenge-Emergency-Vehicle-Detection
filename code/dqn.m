%% Anirudh Topiwala
clc; clear all;close all;
%% Setting Directory
input = './Ambulance1/';
% imagelabeldatastoreuncroppedimages= imageDatastore(input,'IncludeSubfolders',true,'LabelSource','foldernames');
% countEachLabel(imagelabeldatastoreuncroppedimages)
% % input = '../Input/final/detection/zoomed out/wounds/';
% output='../Output/detectionwith3classresnet/';
% apath= '../Input/final/detection/for label/';
digitDatasetPath = fullfile('./Ambulance1/');
% digitDatasetPath = fullfile('F:\UMD Summer\Wound Detection\Input\max images\');
imds = imageDatastore(digitDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
input = imageDatastore(input,'IncludeSubfolders',true,'LabelSource','foldernames');

%% Showing the Dataset
% figure;
% perm = randperm(2000,20)';
% for i = 1:20
%     subplot(4,5,i);
%     imshow(imds.Files{perm(i)});
% end
labelCount = countEachLabel(imds);
img = readimage(imds,1);
size(img);

%% Splitting Data
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
% imdsValidation= imageDatastore(apath,'IncludeSubfolders',true,'LabelSource','foldernames');
%% Training Data
numTrainImages = numel(imdsTrain.Labels);
%% Loading Pre trained neural network alexnet
net = alexnet;
% net= resnet101;
% layersTransfer = net.Layers(1:end-3);

%%
% n=numel(gTruth.imageFilename)

% for i=1:numel(gTruth.imageFilename)
% 
%     i
%    
%     img= imread( gTruth.imageFilename{i});
%   
%    if( size(img,3)~=3)
% %        figure
% %        imshow(img);
%        gTruth(i,:)=[];
%    end
%    
%     
%     
% end

%% Data Augmentation
% inputSize = net.Layers(1).InputSize;
% pixelRange = [-30 30];
% imageAugmenter = imageDataAugmenter( 'RandRotation',[0 360],...
%     'RandXReflection',true, ...
%     'RandYReflection',true, ...
%     'RandXScale' ,[0.5 4], ...
%     'RandYScale' ,[0.5 4], ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange);
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
%     'DataAugmentation',imageAugmenter,'ColorPreprocessing','gray2rgb');
% 
% augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%% Defining DQN
% % % numClasses = numel(categories(imdsTrain.Labels));
% % % layers = [
% % %     layersTransfer
% % %     fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
% % %     softmaxLayer
% % %     classificationLayer];
% layers = [
%     imageInputLayer([400 400 3])
%     
%     convolution2dLayer(3,8,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,16,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,32,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     
%     fullyConnectedLayer(2)
%     softmaxLayer
%     classificationLayer];
%% Training Options
% options = trainingOptions('sgdm', ...
%     'MaxEpochs',10, ...
%     'ValidationData',imdsValidation, ...
%     'ValidationFrequency',30, ...
%     'Verbose',false, ...
%     'Plots','training-progress');
options = trainingOptions('sgdm', ...
    'MiniBatchSize',20, ...
    'ExecutionEnvironment','auto', ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');
%% GUI
% netTransfer = trainNetwork(augimdsTrain,layers,options);
% load('transfernet1.mat');
% load('transfernetmoreimages.mat')
% load('myaugmentedimagesmini15.mat')
%  load('a.mat')
%  load('zommedinaugmenteddataminibatch154classes.mat')
%  load('zommedoutminibatch154class.mat')
%  load('zommedinoutminibatch154class.mat')
%   load('binaryclassifierwoundequaldata.mat')
%   load('binaryclassifierbb.mat')
% load('adam4class.mat')
%  load('resnet1014class15epochsadam.mat')
% load('resnet1013class15epochssgdm.mat')
%% Visualizing the First Layers
% Extract the first convolutional layer weights
%  w = netTransfer.Layers(2).Weights;
% 
% % rescale the weights to the range [0, 1] for better visualization
% w = rescale(w);
% 
% figure
% montage(w)
%% Predict
% % % [YPred,scores] = classify(netTransfer,augimdsValidation);
% % % YValidation = imdsValidation.Labels;
% % % % 
% % % accuracy = sum(YPred == YValidation)/numel(YValidation)
% % % idx = randperm(numel(imdsValidation.Files),4);
% % % figure
% % % for i = 1:4
% % %     subplot(2,2,i)
% % %     I = readimage(imdsValidation,idx(i));
% % %     imshow(I)
% % %     label = YPred(idx(i));
% % %     title(string(label));
% % % end
%% Accuracy
% YValidation = imdsValidation.Labels;
% accuracy = mean(YPred == YValidation)
% [c,cm,ind,per] = confusion((YValidation=={'wounds'})',(YPred=={'wounds'})');
% plotconfusion(YValidation,YPred)


%% Taking images and Predicting
% filesize=size(input.Files);
% figure
% for i=1:30:filesize(1)
% 
% img= imread(input.Files{i});
% img= imresize(img,[227 227]);
% imshow(img);
% naming= {'bandage','bellybutton','not','wounds'};  
% Ypre = predict(netTransfer,img);
% [~,id]= max(Ypre);
% subplot(4,4,i)
% imshow(img)
% title(naming{id});

% Detecting
% % % [windows,windowimg,numwindows]= makewindow(img);
% % % disp('Number of Windows');
% % % numwindows
% % % bandageimg= img;
% % % bellybuttonimg= img;
% % % woundimg= img;
% % % notwound= img;
% % % % figure
% % % % imshow(img);
% % % for j=1:size(windows,1)
% % %     checkimg= imcrop(img,windows(j,:));
% % %     if (size(checkimg,1)==0)
% % %         continue
% % %     end
% % %     checkimg= imresize(checkimg,[227 227]);
% % % %     imshow(checkimg);
% % %     pre(j,:)= predict(netTransfer,checkimg);
% % %     if( max(pre(j,4))>0.99)
% % % %         disp('in');
% % %         woundimg = insertShape(woundimg,'Rectangle',windows(j,:),'Color','black','LineWidth',1);
% % % %      rectangle('Position',windows(i,:));
% % %     end
% % %     if( max(pre(j,3))>0.99)
% % %         bellybuttonimg = insertShape(bellybuttonimg,'Rectangle',windows(j,:),'Color','black');
% % %     end
% % %     if( max(pre(j,2))>0.99)
% % %             bandageimg = insertShape(bandageimg,'Rectangle',windows(j,:),'Color','black');
% % %     end   
% % % %     if( max(pre(j,2))>0.99)
% % % %             notwound = insertShape(notwound,'Rectangle',windows(j,:),'Color','black');
% % % %     end
% % % 
% % % end
% % % subplot(1,4,1)
% % % imshow(windowimg);
% % % title('Window Size (20,20), overlap (0.8)');
% % % subplot(1,4,2)
% % % imshow(woundimg);
% % % title('Finding wound with score above 0.99');
% % % subplot(1,4,3)
% % % imshow(bellybuttonimg);
% % % title('Finding  BB with score above 0.99');
% % % subplot(1,4,4)
% % % imshow(bandageimg);
% % % title('Finding bandage button with score above 0.99');
% % % % subplot(1,4,4)
% % % % imshow(bandageimg);
% % % % title('Finding bandage with score above 0.96');
% % % 
% % % 
% % % 
% end









%% Doing Feature Extraction
% featureLayer = 'fc1000';
% trainingFeatures = activations(netTransfer, augimdsTrain, featureLayer, ...
%     'MiniBatchSize', 32, 'OutputAs', 'columns');

%% Defining Options for rcnn
% %     optionsrcnn = trainingOptions('sgdm', ...
% %         'MiniBatchSize', 128, ...
% %         'InitialLearnRate', 1e-3, ...
% %         'LearnRateSchedule', 'piecewise', ...
% %         'LearnRateDropFactor', 0.1, ...
% %         'LearnRateDropPeriod', 100, ...
% %         'MaxEpochs', 200, ...
% %         'Verbose', true);
%     
%% Loading Labels    
%   load('bandagelabels.mat')
% load('labelall.mat')  
% load('uncroppedlabelall.mat')
% load('labelalluncroppedimaged3class.mat')
% load('numberedlabeall3class.mat')
% load('labelallbbphantom.mat')
% RCNN
% rcnn = trainRCNNObjectDetector(labelall, netTransfer, optionsrcnn, ...
%     'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1]);
% load('rcnnbandage.mat')
% load('rcnnall.mat')
% load('rcnnallmoreimages.mat')
% load('b.mat')
% load('fasterrcnnadam4class.mat')
% load('fasterrcnnadamalexnet.mat')
% load('fasterrcnnalex50epochsgdm.mat')
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

 fasterrcnn = trainFasterRCNNObjectDetector(labelall, netTransfer, optionsfasterrcnn, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.6 1], ...
        'NumStrongestRegions',Inf,...
        'SmallestImageDimension',400,...
        'BoxPyramidScale', 1.2);
% save('fasterrcnn3classalexnet.mat','fasterrcnn');
load('fasterrcnn3classalexnet.mat')
% save('fasterrcnnadamvggnet19.mat','fasterrcnn');
%% Read test image
% load('rcnnall.mat')
filesize=size(input.Files);
resulttable=labelall;
count=0;
for i=1:1:filesize(1)
    % img= (imread('C:\Users\Anirudh\Desktop\2.png'));
    img= imread(input.Files{i});
    img= imresize(img,[227 227]);

    %Expected results
    n=find(strcmp(input.Files{i},labelall{:,:}));
        if(n~=0)
        resulttable{i,:}= labelall{n,:};   
        else
            count=count+1;
        end
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
    % imshow(outputImage) 
    imwrite(outputImage,strcat(output,num2str(i),'.jpg'));

    % Debugging
    % % featureMap = activations(rcnn.Network, img, 'softmax');
    % % ban = featureMap(:, :, 1);
    % % % Resize stopSignMap for visualization
    % % [height, width, ~] = size(img);
    % % ban = imresize(ban, [height, width]);
    % % 
    % % % Visualize the feature map superimposed on the test image. 
    % % featureMapOnImage = imfuse(img, ban); 
    % % 
    % % % classifier
    % % % img= imread(input.Files{i});
    % % predi = predict(netTransfer,img);
    % % [~,id]= max(predi);


    % figure
    % imshow(featureMapOnImage)
    % subplot(1,3,1)
    % imshow(outputImage)      
    % title('Detection');
    % subplot(1,3,2)
    % imshow(featureMapOnImage) 
    % title('Rough Estimate');
    % subplot(1,3,3)
    % imshow(img) 
    % title(string(id));



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