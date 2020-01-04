function [out1 out2] = MyRBPN(s_input,s_test,TA,PA,PAT,L)
format long 
% out1 = returns list of matched persons
% out2 = returns recognition rate
% s_input --> row-wise training set example: 700x64 (700 images, each
% having 64 features. Features can be extracted using PCA, DCT, ICA etc.)
% s_test --> row-wise testing set example: 700x64
% TA --> no. of targets
% PA --> no. of images per train person;
% PAT --> no. of images per test person;
% L --> learning rate
%% Variables
% TA = 100;                                   % No. of targets  
% N = 64;                                     % No. of PCs
P = size(s_input,1);                          % No. of train images;        % No. of image (total)
Pt = size(s_test,1);                          % No. of test images;
NTP = Pt/PAT;                                 % No. of test persons
% Hc = 7;                                     % No. of centers in 1st hidden layer per class
% H1 = Hc*TA;                                 % Total no. of centers in 1st hidden layer 
U = 1/PA;
success = 0;
error = 0;

%% Create Target: ------> Size: PxTA
T1 = eye(TA);
val = 1;
for i = 1:TA
    T2(:,val:val+PA-1) = repmat(T1(:,i),1,PA);
    val = val+PA;
end
T = T2';                                    % Row-wised distributed, Size: PxTA-->200x40
clear val
    
%% Center Determination 
% Method 1: Subclustering
val = 1;
valc = 1;
NoC = 0;
for i = 1:TA    
    DC = s_input(val:val+PA-1,:);           % calling images of each class % DC-->data for center    
    [Cx S] = subclust(DC,0.3);              % size(Cx = HcxN; 3x80 
    valr = size(Cx,1);
    C(valc:valc+valr-1,:) = Cx;
    stCntPCls(i) = size(Cx,1);              % stCntPCls --> store no. of centers per class
    NoC = NoC + size(Cx,1);                 % NoC --> Total No. of Centers      
    val = val+PA;
    valc = valc + valr;
end
clear val valc          
% [index,C] = kmeans(s_input,H1);           % Global
% [C S] = subclust(s_input,0.3);
%% Activation Function for 1st hidden layer: ---> size: PxH1
for i = 1:P
    for j = 1:NoC
        d1(i,j) = dist(s_input(i,:),C(j,:)');
    end
end
d2 = d1.*d1;                                % size: PxH1-->200x120

%% Sigma calculation

% % METHOD 1:
% % BB1 = sum(d1)./(P-1);
% % BB2 = BB1;
% BB1 = max(S);
% BB1 = repmat(BB1,[1 NoC]); 
% % BB1 = maxS_St;
% sig = repmat(BB1,[P 1]); 


% METHOD 2:
[r,c] = size(C);
for i = 1:r
    tempC = C;
    tempC(i,:) = [];
    distCi = dist(C(i,:),tempC');
    sumCi = sum(distCi);
    BB1(i) = (1/(r-1))*sumCi;
end
BB1 = max(BB1)
sig = repmat(BB1,[P 1]); 


%% Activation function for 1st hidden layer
G = exp(-d2./(2*(sig.*sig)));               % Act. Function; each raw for each input; size: PxH1

%% Activation Fucntion for 2nd hidden layer: ---> size: PxTA
for i = 1:P
    Ha = G(i,:);                            % Calling A.F. for 1st image, 1xH1-->1x120    
    valc = 1;
    for j = 1:TA
        Hb00(j) = sum(Ha(valc:valc+stCntPCls(j)-1));    % sum A.Fs of each class, 1xTA-->1x40
        Hb(j)=(Hb00(j)); 
        % If you want to normalize, then Hb(j) = U*Hb00(j); Sometimes
        % normalization is found useful. 
        valc = valc + stCntPCls(j);
    end
    clear valc
    H2(i,:) = Hb;                            % Row-wised distributed, size: PxTA-->200x40   
end

%% Go Calculation
for i = 1:NoC
    for j = 1:NoC
        d10(i,j) = dist(C(i,:),C(j,:)');
    end
end
d20 = d10.*d10;                                % size: H1xH1-->120x120

sig0 = repmat(BB1,[NoC 1]);                    % Sigma for G0
G0 = exp(-d20./(2*(sig0.*sig0)));              % Act. Function; each raw for each input; size: H1xH1

%% H0 Calculation
for i = 1:NoC
    Ha0 = G0(i,:);                            % Calling A.F. for 1st image, 1xH1-->1x120    
    valc = 1;
    for j = 1:TA
        Hb0(j) = sum(Ha0(valc:valc+stCntPCls(j)-1));    % sum A.Fs of each class, 1xTA-->1x40
        valc = valc + stCntPCls(j);
    end
    clear valc
    H20(i,:) = Hb0;                           % Row-wised distributed, size: PxTA-->200x40
end

%% Weight Update
%     [Q Rx] = qr(H2);                            % size: Q:PxP-->200x200, Rx:PxTA-->200x40
%     R = Rx(1:TA,1:TA);                          % size: R:TAxTA-->40x40
% 
%     Y2 = Q'*T;                                  % size: PxTA-->200x40
%     Y3 = Y2(1:TA,1:TA);                       % size: (P-TA)xTA-->160x40

%     W = inv(R)*Y3;                             % size: TAxTA-->40x40
% Method 1:
W = inv(H2'*H2 + L*H20'*H20)*H2'*T;

% ----------------------------------------------------------------------- %
% -------------TEST--------------TEST-----------------TEST--------------- %
% ----------------------------------------------------------------------- %
%% Activation Function for 1st hidden layer: ---> size: PxH1    
for i = 1:Pt
    for j = 1:NoC
        d1t(i,j) = dist(s_test(i,:),C(j,:)');
    end
end
d2t = d1t.*d1t;                                 % size: PxH1-->200x120               
sigt = repmat(BB1, [Pt 1]);                     % Simga for test
Gt = exp(-d2t./(2*(sigt.*sigt)));               % Act. Function; each raw for each input; size: PxH1

%% Activation Fucntion for 2nd hidden layer: ---> size: PtxTA
for i = 1:Pt
    Hat = Gt(i,:);                            % Calling A.F. for 1st image, 1xH1-->1x120    
    valc = 1;
    for j = 1:TA
        Hbt11(j) = sum(Hat(valc:valc+stCntPCls(j)-1));    % sum A.Fs of each class, 1xTA-->1x40
        Hbt (j)= Hbt11(j) ; 
        % If you want to normalize, then Hbt(j) = U*Hbt11(j); Sometimes
        % normalization is found useful. 
        valc = valc + stCntPCls(j);
    end
    clear valc
    H2t(i,:) = Hbt;                           % Row-wised distributed, size: PtxTA-->200x40
end

%% Output
Yt = H2t*W;

%% Recognition rate
[~,cnt] = max(Yt');

val = 1;
for i = 1:NTP
    TP(val:val+PAT-1) = i;
    val = val + PAT;
end
clear val

for i = 1:length(cnt)
    if cnt(i) == TP(i)
        success = success+1;
    else
        error = error+1;
    end
end

rate_percent=(success/(success+error))*100;

out1 = cnt;
out2 =  rate_percent;
end

