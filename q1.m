close all; clear ; clc;
n = 2;      % number of feature dimensions
N = 10000;   % number of iid samples
% Class 0 parameters
mu0 = [3 0;0 3];
Sigma0(:,:,1) = [2 0;0 1];
Sigma0(:,:,2) = [1 0;0 2];
w = [0.5 0.5];
% Class 1 parameters
mu1 = [2;2]; 
Sigma1 = [1 0;0 1];
p = [0.65 0.35]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
% Draw samples from each class pdf
x = zeros(n,N);
for i = 1 : N
    if label(i) == 0
        % Class 0 samples for each gaussian based on their distribution
        if rand > w(1)
            x(:,i) = mvnrnd(mu0(:,1),Sigma0(:,:,1),1)';
        else
            x(:,i) = mvnrnd(mu0(:,2),Sigma0(:,:,2),1)';
        end
    end
    if label(i) == 1
        % Class 1 samples for each gaussian based on their distribution
        x(:,i) = mvnrnd(mu1,Sigma1,1)';
    end
end
% Plot with class labels
figure(1),
plot(x(1,label==0),x(2,label==0),'o'), hold on;
plot(x(1,label==1),x(2,label==1),'+'), axis equal;
legend('Class 0','Class 1');
title('Data and their true labels');
xlabel('x_1'), ylabel('x_2');


class0pdf = w(1)*evalGaussian(x,mu0(:,1),Sigma0(:,:,1)) + ...
    w(2)*evalGaussian(x,mu0(:,2),Sigma0(:,:,2));
class1pdf = evalGaussian(x,mu1,Sigma1);
discriminantScore = log(class1pdf)-log(class0pdf);
 
%calculate the threshold
sortDisScr = sort(discriminantScore(discriminantScore >= 0));
l = length(sortDisScr);
threshold = zeros(1,l);
decision = zeros(N,l);
indCount = zeros(4,l);
 
for i = 1 : length(sortDisScr)
    if i == 1
        threshold(i) = sortDisScr(1);   
    else
        threshold(i) = (sortDisScr(i-1) + sortDisScr(i)) / 2;  
    end
    %threshold(i)= threshold(i)*2;
end
% compare score to threshold and make decisions
for i = 1 : length(sortDisScr)
        decision(:,i) = (discriminantScore >= log(threshold(i)));
end
% count the number of each situation
for i = 1 : length(sortDisScr)
    count00 = length(find(logical(decision(:,i)')==0 & label==0)); % number of true negative
    count10 = length(find(logical(decision(:,i)')==1 & label==0)); % number of false positive
    count01 = length(find(logical(decision(:,i)')==0 & label==1)); % number of false negative
    count11 = length(find(logical(decision(:,i)')==1 & label==1)); % number of true positive
    %[p10; p11; threshold;p_error]
    indCount(:,i)=[count10/Nc(1);count11/Nc(2);threshold(i);(count10+count01)/N];
end
[min_error,index] = min(indCount(4,:));
% Plot ROC Curve
figure(2),
plot(indCount(1,:),indCount(2,:),"b-"), axis equal; hold on;
%xlim([0 1]), ylim([0 1])
title('ROC Space');
xlabel('FPR'), ylabel('TPR');
plot(indCount(1,index),indCount(2,index),"ro");
%Print Results
fprintf ( 'Minimum probability of error : %.4f\nThreshold Value: %.4f\n\n',...
min_error,indCount(3,index));
ideal_decision =(discriminantScore >= log(p(1)/p(2)));
ideal_pFA = sum(ideal_decision==1 & label==0)/Nc(1); % False alarm
ideal_pTP = sum(ideal_decision==1 & label==1)/Nc(2); % True Positive
% ideal_error = ideal_pFA*p(1) + (1-ideal_pTP)*p(2);
ideal_pNP = sum(ideal_decision==0 & label==1)/Nc(2);% Miss
ideal_error = ideal_pFA*p(1) + ideal_pNP*p(2);

plot(ideal_pFA,ideal_pTP,"gd");
legend('roc curve','min-P_{error}','ideal m,in-P_{error}');
%Print Results
fprintf ( 'Minimum ideal probability of error : %.4f\nIdeal threshold Value: %.4f\n\n',...
ideal_error,p(1)/p(2));





Sb = (w(1)*mu0(:,1)+w(2)*mu0(:,2) - mu1) * (w(1)*mu0(:,1)+w(2)*mu0(:,2)-mu1)';
Sw =w(1)^2*Sigma0(:,:,1)+w(2)^2*Sigma0(:,:,2) + Sigma1;
[V,D] = eig(Sw\Sb); % alpha w = inv(Sw) Sb w 
[ ~,ind] = sort(diag(D),'descend');
wLDA = V(: , ind (1) ) ; % Fisher LDA projection vector
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA 
% Plot LDA projection
figure (3),
plot (yLDA(  label==0) , zeros (1 ,Nc(1) ) , 'o ' ,yLDA(  label==1) , zeros (1 ,Nc( 2 ) ) , '+ ' ) ;
title('LDA projection of data points and their true labels'); 
xlabel('y_{LDA}');
legend('Class 0','Class 1');
% Sort LDA projection vector and find midpoints
sorted_yLDA = sort(yLDA);
%mid_tau_LDA = [sorted_yLDA(1)-1 sorted_yLDA(1:end-1)+diff(sorted_yLDA)./2 sorted_yLDA(length(sorted_yLDA))+1];
threshold2 = zeros(1,length(sorted_yLDA));
l = length(threshold2);
for i = 1 : l
    if i == 1
        threshold2(i) = sorted_yLDA(1);   
    else
        threshold2(i) = (sorted_yLDA(i-1) + sorted_yLDA(i)) / 2;  
    end
end
% Make decision for every threshold value and find error probabilities
pFA_LDA = zeros(1,l); % False alarm 
pTP_LDA = zeros(1,l); % Correct detection 
pE_LDA = zeros(1,l);
for i = 1: l
    decisionLDA = ( yLDA >= threshold2(i)) ;
    pFA_LDA( i ) = sum(decisionLDA==1 & label==0)/Nc(1) ; % False alarm
    pTP_LDA( i ) = sum(decisionLDA==1 & label==1)/Nc(2) ; % Correct detection 
    pE_LDA(i) = pFA_LDA(i)*p(1) + (1-pTP_LDA(i))*p(2);
end
% Find minimum error and corresponding threshold
[ min_error_LDA , min_index_LDA ] = min(pE_LDA) ;
%min_decision_LDA = (yLDA >= threshold2 ( min_index_LDA ) ) ;
min_FA_LDA = pFA_LDA(min_index_LDA) ; 
min_TP_LDA = pTP_LDA(min_index_LDA) ;

plot(pFA_LDA, pTP_LDA); axis equal; hold on;
%xlim([0 1]), ylim([0 1])
title('Fisher LDA ROC Curve'); 
xlabel('FPR'), ylabel('TPR');
plot(pFA_LDA(min_index_LDA),pTP_LDA(min_index_LDA),"ro");
legend('roc curve','min-P_{error}');
%Print Results
fprintf ('For fisher FDA\n')
fprintf ( 'Minimum probability of error : %.4f\nThreshold Value: %.4f\n\n',...
min_error_LDA, threshold2(min_index_LDA));



