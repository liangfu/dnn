function mat2xml
    
template=['<img%d type_id="opencv-matrix">' ...
          '<rows>%d</rows>' ...
          '<cols>%d</cols>' ...
          '<dt>f</dt>' ...
          '<data>%s</data></img%d>\n'];
load digitStruct;
fp=fopen('digitStruct.xml','wt');
fprintf(fp,'<?xml version="1.0"?><opencv_storage>\n');

for ii=1:length(digitStruct)
cols=5;
rows=length(digitStruct(ii).bbox);
numbers='';
for jj=1:rows
xx=digitStruct(ii).bbox(jj).left;   % x
yy=digitStruct(ii).bbox(jj).top;    % y
ww=digitStruct(ii).bbox(jj).width;  % w
hh=digitStruct(ii).bbox(jj).height; % h
ll=digitStruct(ii).bbox(jj).label;  % l
numbers=[numbers sprintf('%.0f ',[xx,yy,ww,hh,ll])];
end % jj
fprintf(fp,template,ii,rows,cols,numbers,ii);
if mod(ii,100)==1,disp(sprintf('%d/%d,',ii,length(digitStruct)));end
end % ii

fprintf(fp,'</opencv_storage>\n');
fclose(fp);
  
end

