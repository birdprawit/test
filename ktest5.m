% 30 25 20 15 10 5 3 2
%% i6m32B
k = [30,25,20,15,10,5,3,2];
p = 0.1;
for b=1:length(k)
    
%clear;
load 'c:\GAimpute\GenMissing\ALLAMLMissing\ALL10Miss\dtmiss2.data';
load 'c:\GAimpute\GenMissing\ALLAMLMissing\ALL10Miss\miss2.data';
load 'c:\GAimpute\GenMissing\ALLAMLMissing\ALL10Miss\missvalue2.data';
load 'c:\GAimpute\GenMissing\ALLAMLMissing_N\ALL10Miss\ALLAML_N1\ALLAML_N1_T9.data';

dtmiss=dtmiss2;
missvalue=missvalue2;
miss=miss2;
ALLAML_N1_T1 = ALLAML_N1_T9;
filename = 'ALLAMLMissing_N1_10P_knn_T9.xlsx';

[row,col]=size(dtmiss);


oneP=floor(row*col*p);

EsValue=zeros(oneP,1);%ค่าประมาณ
EsValueKNN=zeros(oneP,1);

%เป็นขั้นตอนการสร้าง Missing Value%

[dtnew]=FKnnEst(dtmiss,miss,oneP,col,row);

%จากนั้นนำทำ Knn โดยใช้ Euclidean distane ในการหาระยะห่าง
dst=0;
dstIdx=zeros(col,2);
cs=0;
csIdx=zeros(row,2);

for r1=1:10

    for i=1:oneP
        RM=miss(i,1);%แถว
        Cl=miss(i,2);%คอลัมน์       
        %หา Distance ของ feature 
        for j=1:col
            dst=Eucli(Cl,RM,j,row,ALLAML_N1_T1); 
            dstIdx(j,1)=j;
            dstIdx(j,2)=dst;
        end        
        
        d=sortrows(dstIdx,sort(2)); %sort feature d คือ คอลัมน์ที่ได้จากการหาdistance จาก missing values%
        
        %-------KNN---------%
        for rc=1:row
            cs=EucliCaseKNN(Cl,RM,col,rc,ALLAML_N1_T1);
            csIdx(rc,1)=rc;
            csIdx(rc,2)=cs;
        end         
        rc30=sortrows(csIdx,sort(2)); 
        sum30=0;
        for s=1:k(b)
            rec=rc30(s+1,1);
            sum30=sum30+dtmiss(rec,Cl);
        end  
       EsValue(i)=sum30/k(b);% พยากรณ์แบบ KNNปกติ
       dtmiss(RM,Cl)=EsValue(i);  

    end


    
    
    
    
      rms_errKNN=sum((missvalue-EsValue).^2);
      accurKNN(b,r1)=rms_errKNN/oneP;
      accurKNN(b,r1)=sqrt(accurKNN(b,r1)/(std(missvalue))^2);

   
end

load 'c:\GAimpute\GenMissing\ALLAMLMissing\ALL10Miss\dtmiss2.data';
load 'c:\GAimpute\GenMissing\ALLAMLMissing\ALL10Miss\miss2.data';
load 'c:\GAimpute\GenMissing\ALLAMLMissing\ALL10Miss\missvalue2.data';
load 'c:\GAimpute\GenMissing\ALLAMLMissing_N\ALL10Miss\ALLAML_N1\ALLAML_N1_T9.data';

dtmiss=dtmiss2;
missvalue=missvalue2;
miss=miss2;
ALLAML_N1_T1 = ALLAML_N1_T9;
% 30 25 20 15 10 5 3 2

[row,col]=size(dtmiss);

oneP=floor(row*col*p);



EsValue1=zeros(oneP,1);%ค่าประมาณ
EsValueKNN1=zeros(oneP,1);
EsValueROW1=zeros(oneP,1);
EsValueKNNW1=zeros(oneP,1);
EsValueRegression1=zeros(oneP,1);

%เป็นขั้นตอนการสร้าง Missing Value%


[dtnew]=FKnnEst(dtmiss,miss,oneP,col,row);

%จากนั้นนำทำ Knn โดยใช้ Euclidean distane ในการหาระยะห่าง
dst=0;
dstIdx=zeros(col,2);
cs=0;
csIdx=zeros(row,2);

    for i=1:oneP
        RM=miss(i,1);%แถว
        Cl=miss(i,2);%คอลัมน์       
        %หา Distance ของ feature 
        for j=1:col
            dst=Eucli(Cl,RM,j,row,ALLAML_N1_T1); 
            dstIdx(j,1)=j;
            dstIdx(j,2)=dst;
        end        
        
        d=sortrows(dstIdx,sort(2)); %sort feature d คือ คอลัมน์ที่ได้จากการหาdistance จาก missing values%
        %---regression--%
        %X=zeros(61,31);        
        %Xp=zeros(1,31);        
        [Xi,Y,Xp1]=Regs(d,RM,Cl,row,dtnew); % fuction เตรียมข้อมูล
        [nr,nc]=size(Xi);
        X(:,:)=[ones(nr,1),Xi];
        B=regress(Y,X)';
        
        [nr1,nc1]=size(Xp1);
        Xp(:,:)=[ones(nr1,1),Xp1];
        Pr=sum(B.*Xp);
        EsValueRegression1(i)=Pr; %พยากรณ์ Regression
    
        %------KNNFS------%
        d=sortrows(dstIdx,sort(2)); %sort ได้ 30 feature        
        for rc=1:row %หาตัวอย่างที่ใกล้
            cs=EucliCase(Cl,RM,d,rc,ALLAML_N1_T1);
            csIdx(rc,1)=rc;
            csIdx(rc,2)=cs;
        end                 
        rc30=sortrows(csIdx,sort(2)); 
        sum30=0;
        for s=1:k(b) % 30 record
            rec=rc30(s+1,1);
            sum30=sum30+dtmiss(rec,Cl);
        end       
       EsValue1(i)=sum30/k(b); %พยากรณ์แบบKNNFS       
        %------NewWeight-----%              
        rcW=[rc30(2:31,1),1./rc30(2:31,2)];
        sumWX=0;
        sumW=0;
        for s=1:k(b) % 30 record
            rec=rcW(s,1);
            sumWX=sumWX+dtmiss(rec,Cl)*rcW(s,2);
            sumW=sumW+rcW(s,2);
        end       
       EsValueKNNW1(i)=sumWX/sumW; % พยากรณ์แบบ KNNFSW
       
       %-------KNN---------%
        for rc=1:row
            cs=EucliCaseKNN(Cl,RM,col,rc,ALLAML_N1_T1);
            csIdx(rc,1)=rc;
            csIdx(rc,2)=cs;
        end         
        rc30=sortrows(csIdx,sort(2)); 
        sum30=0;
        for s=1:k(b)
            rec=rc30(s+1,1);
            sum30=sum30+dtmiss(rec,Cl);
        end  
       EsValueKNN1(i)=sum30/k(b);% พยากรณ์แบบ KNNปกติ
       %----------Rowaverage----%
       sum30=0;
       for s=1:row
           if s~=RM
              sum30=sum30+dtmiss(s,Cl);
           end
        end  
       EsValueROW1(i)=sum30/(row-1); %พยากรณ์แบบ ROW
       
       
    
    end
   
   
      rms_err=sum((missvalue-EsValue1).^2);   
      accurKNN1(b,1)=rms_err/oneP;
      accurKNN1(b,1)=sqrt(accurKNN1(b,1)/(std(missvalue))^2); %การหาค่าความแม่นยำของการพยากรณ์%
  
      rms_errKNN=sum((missvalue-EsValueKNN1).^2);
      accurKNN1(b,2)=rms_errKNN/oneP;
      accurKNN1(b,2)=sqrt(accurKNN1(b,2)/(std(missvalue))^2);
      
      rms_errROW=sum((missvalue-EsValueROW1).^2); 
      accurKNN1(b,5)=rms_errROW/oneP;
      accurKNN1(b,5)=sqrt(accurKNN1(b,5)/(std(missvalue))^2); 
  
      rms_errKNNW=sum((missvalue-EsValueKNNW1).^2);
      accurKNN1(b,3)=rms_errKNNW/oneP;
      accurKNN1(b,3)=sqrt(accurKNN1(b,3)/(std(missvalue))^2);     
      
      rms_errRegression=sum((missvalue-EsValueRegression1).^2);
      accurKNN1(b,4)=rms_errRegression/oneP;
      accurKNN1(b,4)=sqrt(accurKNN1(b,4)/(std(missvalue))^2);     


end

xlswrite(filename,accurKNN1,1,'B3');xlswrite(filename,accurKNN,1,'G3');
    
  
 

