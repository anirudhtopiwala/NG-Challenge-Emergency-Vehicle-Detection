function [fbox,fscore,flabel]= chooseboxes(bboxes,scores,labels)

if(isempty(labels))
    fbox= bboxes; fscore= scores; flabel= labels;
   return
end
k=1;
  if(sum(labels=='bellybutton'))
      bbindex=scores==max(scores(labels=='bellybutton'));    
      fbox(k,:)= bboxes(bbindex,:);  
      fscore(k)=scores(bbindex);
      flabel(k)=labels(bbindex);
      k=k+1;
  end
  if(sum(labels=='bandage'))
      bbindex=scores==max(scores(labels=='bandage'));    
      fbox(k,:)= bboxes(bbindex,:);  
      fscore(k)=scores(bbindex);
      flabel(k)=labels(bbindex);
      k=k+1;
  end
  if(sum(labels=='wounds'))
      bbindex=scores==max(scores(labels=='wounds'));    
      fbox(k,:)= bboxes(bbindex,:);  
      fscore(k)=scores(bbindex);
      flabel(k)=labels(bbindex);
      k=k+1;
  end
  c=1;ind=0;
  for i=1:size(flabel,2) 
      if (fbox(i,3)/fbox(i,4) < 0.7 || fbox(i,4)/fbox(i,3) < 0.7||fbox(i,3) >200 || fbox(i,4)>200)
           ind(c)= i;c=c+1;
      end
  end
  if (ind~=0)
      fbox(ind,:)=[];
      flabel(ind)=[];
      fscore(ind)=[];
  
  end
  
% catch
%     fbox= bboxes; fscore= scores; flabel= labels;
%     return
% end

end