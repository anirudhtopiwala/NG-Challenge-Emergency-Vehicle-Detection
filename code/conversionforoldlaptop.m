% for i=1: numel(labelall.imageFilename)
%    str= labelall.imageFilename{i};
% %    newStr = strrep(str,'D:\Adventure Island\Study\Extras\Project\Emergency Vehicle Detection\Training\positive\','/home/anirudh/ng challenge1/');
% % 
%     newStr= strrep(str,'\','/');
%    labelall.imageFilename{i}=newStr;
% %     
% end
%% Displaying to check if path is valid
for i=1: numel(labelall.imageFilename)
    i
 img= imread(labelall.imageFilename{i});
%  imshow(img);
%  delete(labelall.imageFilename{i});
end