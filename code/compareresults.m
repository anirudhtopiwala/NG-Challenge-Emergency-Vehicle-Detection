%% Comparing Results
clc;clear all;close all;
%% Setting Input Directory
input = '../Input/ICRA/test/bellybutton/';
% input='../Input/final/Phantom/trumph/';
% input2='../Input/ICRA/augmented/bellybutton/';
input = imageDatastore(input,'IncludeSubfolders',true,'LabelSource','foldernames');
% input2 = imageDatastore(input2,'IncludeSubfolders',true,'LabelSource','foldernames');
output='../Input/ICRA/output/bellybutton/';
%% Loading all Networks
load('adam4class.mat')
% load('resnet1013class15epochssgdm.mat')
%  load('resnet1014class15epochsadam.mat')
% load('fasterrcnn3classalexnet.mat')
load('fasterrcnn3classaugmentedimagesfinal.mat')
imagesize = netTransfer.Layers(1).InputSize;
%% Taking Images and Making Boxes
filesize=size(input.Files);
boxbb=[];
for i=1:1:filesize(1)
    
img= imread(input.Files{i});
% img=imread('test4_Color.png');
img= imresize(img,imagesize(1,1:2));
% Detecting Using Faster RCNN
outputImage=img;
% fasterbandagebinary=uint8(zeros(imagesize(1,1:2)));fasterwoundbinary=uint8(zeros(imagesize(1,1:2)));fasterbellybuttonbinary=uint8(zeros(imagesize(1,1:2)));
[bboxe,scores,label] = detect(fasterrcnn,img);
[bboxes,score,labels] = selectStrongestBboxMulticlass(bboxe,scores,label);

for m=1:size(bboxes,1)
presentwindow=bboxes(m,:);
% score(j)
% if (score(i)>0.95)
annotation = sprintf('%s:', labels(m));
outputImage = insertObjectAnnotation(outputImage, 'rectangle', bboxes(m,:), annotation);
% % %     if(labels(m)=='bandage')
% % %       fasterbandagebinary(presentwindow(2):presentwindow(2)+presentwindow(3),presentwindow(1):presentwindow(1)+presentwindow(4))=255; 
% % %     else
% % %       fasterbandagebinary= 255;
% % %     end
% % %     if(labels(m)=='bb')
% % %       fasterbellybuttonbinary(presentwindow(2):presentwindow(2)+presentwindow(3),presentwindow(1):presentwindow(1)+presentwindow(4))=255; 
% % %     else
% % %         fasterbellybuttonbinary=255;
% % %     end
% % %     if(labels(m)=='wound')
% % %       fasterwoundbinary(presentwindow(2):presentwindow(2)+presentwindow(3),presentwindow(1):presentwindow(1)+presentwindow(4))=255; 
% % %     else
% % %         fasterwoundbinary=255;
% % %     end
% end
end
% figure
% imshow(outputImage) 
% % Results
boxbb=[boxbb;bboxes];
resultsStruct(i).Boxes = bboxes;
resultsStruct(i).Scores = score;
% resultsStruct(i).Labels = labels;

% imwrite(outputImage,strcat(output,num2str(i),'.jpg'));

% % % % Detecting
% % % [windows,windowimg,numwindows]= makewindow(img);
% % % disp(num2str(numwindows));
% % % bandageimg= img;
% % % bellybuttonimg= img;
% % % woundimg= img;
% % % notwound= img;
% % % % heatwound=img;heatbandage=img;heatbellybutton=img;
% % % heatwound= uint8(zeros(imagesize(1,1:2)));heatbandage= uint8(zeros(imagesize(1,1:2)));heatbellybutton= uint8(zeros(imagesize(1,1:2)));
% % % % figure
% % % % imshow(windowimg);
% % % for j=1:size(windows,1)
% % %     presentwindow=windows(j,:);
% % %     checkimg= imcrop(img,presentwindow);
% % %     if (size(checkimg,1)==0)
% % %         continue
% % %     end
% % %     checkimg= imresize(checkimg,imagesize(1,1:2));
% % %     pre(j,:)= predict(netTransfer,checkimg);
% % %     if( max(pre(j,3))>0.9)
% % %         woundimg = insertShape(woundimg,'Rectangle',presentwindow,'Color','black','LineWidth',1);
% % %         heatwound(presentwindow(2):presentwindow(2)+presentwindow(3),presentwindow(1):presentwindow(1)+presentwindow(4))= heatwound(presentwindow(2):presentwindow(2)+presentwindow(3),presentwindow(1):presentwindow(1)+presentwindow(4))+5;
% % %     end
% % %     if( max(pre(j,2))>0.98)
% % %         bellybuttonimg = insertShape(bellybuttonimg,'Rectangle',presentwindow,'Color','black');
% % %         heatbellybutton(presentwindow(2):presentwindow(2)+presentwindow(3),presentwindow(1):presentwindow(1)+presentwindow(4))= heatbellybutton(presentwindow(2):presentwindow(2)+presentwindow(3),presentwindow(1):presentwindow(1)+presentwindow(4))+5;
% % % 
% % %     end
% % %     if( max(pre(j,1))>0.98)
% % %         bandageimg = insertShape(bandageimg,'Rectangle',presentwindow,'Color','black');
% % %         heatbandage(presentwindow(2):presentwindow(2)+presentwindow(3),presentwindow(1):presentwindow(1)+presentwindow(4))= heatbandage(presentwindow(2):presentwindow(2)+presentwindow(3),presentwindow(1):presentwindow(1)+presentwindow(4))+5;
% % %     end   
% % % %     if( max(pre(j,2))>0.99)
% % % %             notwound = insertShape(notwound,'Rectangle',windows(j,:),'Color','black');
% % % %     end
% % % 
% % % end
% % % bandagethresh=heatbandage>0*0.8*(max(max(heatbandage)));
% % % woundthresh=heatwound>0*0.8*(max(max(heatwound)));
% % % bellybuttonthresh=heatbellybutton>0*0.8*(max(max(heatbellybutton)));
% % % if((sum(sum(bandagethresh==0)))==51529)
% % %     bandageand=bandagethresh | fasterbandagebinary;
% % % else
% % %     bandageand=bandagethresh&fasterbandagebinary;
% % % end
% % % if((sum(sum(woundthresh==0)))==51529)
% % %     woundand=woundthresh | fasterwoundbinary;
% % % else
% % %         woundand=woundthresh & fasterwoundbinary;
% % % end
% % % if((sum(sum(bellybuttonthresh==0)))==51529)
% % %     bellybuttonand=bellybuttonthresh | fasterbellybuttonbinary;
% % % else
% % %         bellybuttonand=bellybuttonthresh & fasterbellybuttonbinary;
% % % end
% % % 
% % % 
% % % 
% % % subplot(3,5,1)
% % % imshow(windowimg);
% % % title('Window Size (20,20), overlap (0.8)');
% % % subplot(3,5,2)
% % % imshow(woundimg);
% % % title('Finding wound with score above 0.99');
% % % subplot(3,5,3)
% % % imshow(heatwound);
% % % title('heatwound');
% % % subplot(3,5,4)
% % % imshow(woundthresh);
% % % title('woundthresh');
% % % subplot(3,5,5)
% % % imshow(woundand);
% % % title('woundand');
% % % subplot(3,5,7)
% % % imshow(bellybuttonimg);
% % % title('Finding  BB with score above 0.99');
% % % subplot(3,5,8)
% % % imshow(heatbellybutton);
% % % title('heatbellybutton');
% % % subplot(3,5,9)
% % % imshow(bellybuttonthresh);
% % % title('bellybuttonthresh');
% % % subplot(3,5,10)
% % % imshow(bellybuttonand);
% % % title('bellybuttonand');
% % % subplot(3,5,12)
% % % imshow(bandageimg);
% % % title('Finding bandage button with score above 0.99');
% % % subplot(3,5,13)
% % % imshow(heatbandage);
% % % title('heatbandage');
% % % subplot(3,5,14)
% % % imshow(bandagethresh);
% % % title('bandagethresh');
% % % subplot(3,5,6)
% % % imshow(outputImage);
% % % title('Fasterrcnn detection');
% % % subplot(3,5,15)
% % % imshow(bandageand);
% % % title('bandageand ');

end
results = struct2table(resultsStruct);
load('labelallbellybuttonfinaltesticra.mat')
expectedResults = labelall(:, 2:end);
%% Evaluate the object detector using Average Precision metric.
[ap, recall, precision] = evaluateDetectionPrecision(results, expectedResults,0.4);
% Plot precision/recall curve
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))
%%
% figure
for i=1:3
figure
plot(recall{i},precision{i})
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap(i)))
end
%%
boxactual=[];
for i=1:size(labelall,1)
    value=labelall{i,2};
    boxactual= [boxactual;value{1}];
end