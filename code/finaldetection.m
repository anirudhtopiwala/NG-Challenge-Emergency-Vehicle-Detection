clc;clear all;close all;
% output= '../Input/final/Phantom/bellybutton/bb';
% input='../Input/final/Phantom/bellybutton/';
% input = imageDatastore(input);

%% Loading Networks
load('adam4class.mat')
% load('fasterrcnn3classalexnet.mat')
% load('fasterrcnnbellybuttonphantom.mat')
% load('fasterrcnn3classaugmentedimagesfinal.mat')
 %% Switch Between These Two
 % For Trumph
     % load('fasterrcnntrumph3class.mat')
 % For Ultrasound Phantom
    load('fasterrcnnbellybuttonphantommorerobust.mat')

imagesize = netTransfer.Layers(1).InputSize;

%% Initializing ROs Node
% rosinit('http://160.69.69.106:11311');
sub = rossubscriber('/camera/color/image_raw');
% sub1=rossubscriber('/phantom/joint_states');
msg = receive(sub);
pub = rospublisher('/xybb','geometry_msgs/Twist');
pubimg= rospublisher('/imgbb','sensor_msgs/Image');
msgpub = rosmessage(pub);
msgimg=rosmessage(pubimg);
msgimg.Encoding = 'rgb8';
% msg1=receive(sub);
% msg=1;
i=1;
%% Taking Images
while ~(isempty(msg))
    tic
% filesize=size(input.Files);
% figure
% for i=1:1:filesize(1)
%     img= imread('6.jpeg');
    msg = receive(sub);
    img= readImage(msg);
%     img=imread(input.Files{i});
%     imshow(img);
%     img= imresize(img,imagesize(1,1:2));
%     imwrite(img,strcat(output,num2str(i),'.jpg'));i=i+1;
%     pause(2)

   % Detecting Using Faster RCNN
    outputImage=img;
    [bbox,score,label] = detect(fasterrcnn,img);
    [bboxes,scores,labels,index] = selectStrongestBboxMulticlass(bbox,score,label);
%     score=double(score);
%     for m=1:size(bboxes,1)
%     annotation = sprintf('%s:', labels(m));
%     outputImage = insertObjectAnnotation(outputImage, 'rectangle', bboxes(m,:), annotation);
%     x(m)=bboxes(1)+bboxes(3)/2;
%     y(m)=bboxes(2)+bboxes(4)/2;
%     end
% 
%     imshow(outputImage);
   if( ~isempty(scores))
       
       %% To plot max score box
        [maxi,ind]=max(scores);
        box= bboxes(ind,:);
        annotation = sprintf('%s:', labels(ind));
        outputImage = insertObjectAnnotation(outputImage, 'rectangle', box, annotation);
        x=box(1)+box(3)/2;
        y=box(2)+box(4)/2;
        outputImage=insertMarker(outputImage,[x,y],'star','size',4);
    %     meanx= mean(x);
    %     meany=mean(y);
    %% To plot multiple boxes
%         for m=1:size(bboxes,1)
%         annotation = sprintf('%s:', labels(m));
%         outputImage = insertObjectAnnotation(outputImage, 'rectangle', bboxes(m,:), annotation);
%         end
%         hold on
%         plot (x,y,'o','MarkerSize',5,'LineWidth',4);
        msgpub.Linear.X= x;
        msgpub.Linear.Y= y;
        writeImage(msgimg,outputImage);
        send(pub,msgpub);

%         score
   else  
%      msgpub.Linear.X= 0;
%     msgpub.Linear.Y= 0;
    writeImage(msgimg,img);
   end

%     send(pub,msgpub);
    send(pubimg,msgimg);
    imshow(outputImage);

%             clear score
toc

end