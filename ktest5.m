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

EsValue=zeros(oneP,1);%��һ���ҳ
EsValueKNN=zeros(oneP,1);

%�繢�鹵͹������ҧ Missing Value%

[dtnew]=FKnnEst(dtmiss,miss,oneP,col,row);

%�ҡ��鹹ӷ� Knn ���� Euclidean distane 㹡����������ҧ
dst=0;
dstIdx=zeros(col,2);
cs=0;
csIdx=zeros(row,2);

for r1=1:10

    for i=1:oneP
        RM=miss(i,1);%��
        Cl=miss(i,2);%�������       
        %�� Distance �ͧ feature 
        for j=1:col
            dst=Eucli(Cl,RM,j,row,ALLAML_N1_T1); 
            dstIdx(j,1)=j;
            dstIdx(j,2)=dst;
        end        
        
        d=sortrows(dstIdx,sort(2)); %sort feature d ��� �����������ҡ�����distance �ҡ missing values%
        
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
       EsValue(i)=sum30/k(b);% ��ҡó�Ẻ KNN����
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



EsValue1=zeros(oneP,1);%��һ���ҳ
EsValueKNN1=zeros(oneP,1);
EsValueROW1=zeros(oneP,1);
EsValueKNNW1=zeros(oneP,1);
EsValueRegression1=zeros(oneP,1);

%�繢�鹵͹������ҧ Missing Value%


[dtnew]=FKnnEst(dtmiss,miss,oneP,col,row);

%�ҡ��鹹ӷ� Knn ���� Euclidean distane 㹡����������ҧ
dst=0;
dstIdx=zeros(col,2);
cs=0;
csIdx=zeros(row,2);

    for i=1:oneP
        RM=miss(i,1);%��
        Cl=miss(i,2);%�������       
        %�� Distance �ͧ feature 
        for j=1:col
            dst=Eucli(Cl,RM,j,row,ALLAML_N1_T1); 
            dstIdx(j,1)=j;
            dstIdx(j,2)=dst;
        end        
        
        d=sortrows(dstIdx,sort(2)); %sort feature d ��� �����������ҡ�����distance �ҡ missing values%
        %---regression--%
        %X=zeros(61,31);        
        %Xp=zeros(1,31);        
        [Xi,Y,Xp1]=Regs(d,RM,Cl,row,dtnew); % fuction �����������
        [nr,nc]=size(Xi);
        X(:,:)=[ones(nr,1),Xi];
        B=regress(Y,X)';
        
        [nr1,nc1]=size(Xp1);
        Xp(:,:)=[ones(nr1,1),Xp1];
        Pr=sum(B.*Xp);
        EsValueRegression1(i)=Pr; %��ҡó� Regression
    
        %------KNNFS------%
        d=sortrows(dstIdx,sort(2)); %sort �� 30 feature        
        for rc=1:row %�ҵ�����ҧ������
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
       EsValue1(i)=sum30/k(b); %��ҡó�ẺKNNFS       
        %------NewWeight-----%              
        rcW=[rc30(2:31,1),1./rc30(2:31,2)];
        sumWX=0;
        sumW=0;
        for s=1:k(b) % 30 record
            rec=rcW(s,1);
            sumWX=sumWX+dtmiss(rec,Cl)*rcW(s,2);
            sumW=sumW+rcW(s,2);
        end       
       EsValueKNNW1(i)=sumWX/sumW; % ��ҡó�Ẻ KNNFSW
       
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
       EsValueKNN1(i)=sum30/k(b);% ��ҡó�Ẻ KNN����
       %----------Rowaverage----%
       sum30=0;
       for s=1:row
           if s~=RM
              sum30=sum30+dtmiss(s,Cl);
           end
        end  
       EsValueROW1(i)=sum30/(row-1); %��ҡó�Ẻ ROW
       
       
    
    end
   
   
      rms_err=sum((missvalue-EsValue1).^2);   
      accurKNN1(b,1)=rms_err/oneP;
      accurKNN1(b,1)=sqrt(accurKNN1(b,1)/(std(missvalue))^2); %����Ҥ�Ҥ�������Ӣͧ��þ�ҡó�%
  
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
    
  
 

