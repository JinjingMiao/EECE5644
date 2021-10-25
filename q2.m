clear all, close all, clc

% Givens
C = 2;
Ntrain = [100, 1000, 10000];
Nvalidation = 20000;
alpha = [0.6, 0.4];
w = [0.5, 0.5];

gmmParameters0.priors = w;
gmmParameters0.meanVectors(:,1) = [5;0];
gmmParameters0.meanVectors(:,2) = [0;4];
gmmParameters0.covMatrices(:,:,1) = [4, 0; 0, 2];
gmmParameters0.covMatrices(:,:,2) = [1, 0; 0, 3];

gmmParameters1.priors = 1;
gmmParameters1.meanVectors = [3;2];
gmmParameters1.covMatrices = [2, 0; 0, 2];

% generate traning dataset
for i=1:length(Ntrain)
    [DATA, LABELS] = myGMM(alpha, Ntrain(i), [gmmParameters0, gmmParameters1]);
    Samples.x{i} = DATA;
    Samples.label{i} =LABELS-1;
end
% generate validation dataset
[Xvalidate, Lvalidate] = myGMM(alpha, Nvalidation, [gmmParameters0, gmmParameters1]);
Lvalidate =Lvalidate-1;

% plot training and validation samples
Samples.x{4} = Xvalidate; Samples.label{4} = Lvalidate;
figuNum = 1;
plotSamples(figuNum, 4, Samples);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PART 1: Theoretically optimal classifer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
g1 = evalGaussian(Xvalidate,gmmParameters1.meanVectors(:,1), gmmParameters1.covMatrices(:,:,1));
g0 = w(1)*evalGaussian(Xvalidate,gmmParameters0.meanVectors(:,1), gmmParameters0.covMatrices(:,:,1))...
    + w(2)*evalGaussian(Xvalidate,gmmParameters0.meanVectors(:,2), gmmParameters0.covMatrices(:,:,2));
% classifer conditional ratio is discriminant score
discriminantScores = log(g1) - log(g0);
y_theoretical = alpha(1)/alpha(2), % theoretical treshhold

% calculate classifer performace with validation samples of 20,000 points
% at ideal treshold value
tau_theoretical = log(y_theoretical)
[Pfp,Ptp,Perror,thresholdList] = ROCcurve(discriminantScores,Lvalidate);
% find theoretical operating point
decision = (discriminantScores >= tau_theoretical); % LDA threshold not optimized to minimize its own E[Risk]!
Pfp_Theoretical = length(find(decision==1 & Lvalidate==0))/length(find(Lvalidate==0));
Ptp_Theoretical = length(find(decision==1 & Lvalidate==1))/length(find(Lvalidate==1));
theoretical_minError = sum(decision~=Lvalidate)/length(Lvalidate),
% find emperical operating point
[minError, minError_Ind] = min(Perror),
tau_minError = thresholdList(minError_Ind)

% plot ROC curve
figuNum=1+figuNum;
figure(figuNum)
plot(Pfp,Ptp,'b-', Pfp_Theoretical, Ptp_Theoretical, 'ro', Pfp(minError_Ind), Ptp(minError_Ind), 'g+'), 
xlabel('P(False+)'),ylabel('P(True+)'), title('TheoreticalTheoretical ROC Curve'),
legend('ROC Curve', 'Theoretical', 'Emperical');

% Draw the decision boundary
horizontalGrid = linspace(floor(min(Xvalidate(1,:)))-2,ceil(max(Xvalidate(1,:)))+2);
verticalGrid = linspace(floor(min(Xvalidate(2,:)))-2,ceil(max(Xvalidate(2,:)))+2);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
g1GridValues = evalGaussian([h(:)';v(:)'],gmmParameters1.meanVectors(:,1), gmmParameters1.covMatrices(:,:,1));
g0GridValues = w(1)*evalGaussian([h(:)';v(:)'],gmmParameters0.meanVectors(:,1), gmmParameters0.covMatrices(:,:,1))...
              +w(2)*evalGaussian([h(:)';v(:)'],gmmParameters0.meanVectors(:,2), gmmParameters0.covMatrices(:,:,2));
discriminantScoreGridValues = log(g1GridValues)-log(g0GridValues)-tau_theoretical;
discriminantScoreGrid = reshape(discriminantScoreGridValues,length(verticalGrid),length(horizontalGrid));
figuNum=1+figuNum;
figure(3), % class 0 circle, class 1 +, correct green, incorrect red
ind00 = find(decision==0 & Lvalidate==0); % probability of true negative
ind10 = find(decision==1 & Lvalidate==0); % probability of false positive
ind01 = find(decision==0 & Lvalidate==1); % probability of false negative
ind11 = find(decision==1 & Lvalidate==1); % probability of true positive
plot(Xvalidate(1,ind00),Xvalidate(2,ind00),'og'); hold on,
plot(Xvalidate(1,ind10),Xvalidate(2,ind10),'or'); hold on,
plot(Xvalidate(1,ind01),Xvalidate(2,ind01),'+r'); hold on,
plot(Xvalidate(1,ind11),Xvalidate(2,ind11),'+g'); hold on,
axis equal,
figure(figuNum), contour(horizontalGrid,verticalGrid,discriminantScoreGrid, [0,0]); % plot equilevel contours of the discriminant function 
% including the contour at level 0 which is the decision boundary
legend('Correct Class 0', 'Wrong Class 0', 'Wrong Class 1', 'Correct Class 1', 'Classifier boundry'), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'), 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PART 2: Expectation Maximization Parameter Estimation Performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% for each training sample perform parameter estimation
% and then plot ROC curve and calculate min-P(error)
for  i=length(Ntrain):-1:1
    
    % training samples seperated by labels
    Xtrain_L0 = Samples.x{i}(:,Samples.label{i}==0); % label=0
    Xtrain_L1 = Samples.x{i}(:,Samples.label{i}==1); % label=1
    
    %EM parameter estimation
    M_0=2; M_1=1; % model size for L=0 & L=1
    options = statset('MaxIter',1000); % max iterations allowed
    gm_label0 = fitgmdist(Xtrain_L0',M_0,'Replicates',20,'Options',options); % run 20 random initiallizations
    gm_label1 = fitgmdist(Xtrain_L1',M_1,'Replicates',20,'Options',options); % run 20 random initiallizations
    
    % find discriminant scores using validation data conditionals
    g1_10kTraining = evalGaussian(Xvalidate,gm_label1.mu', gm_label1.Sigma');
    g0_10kTraining = gm_label0.ComponentProportion(1)*evalGaussian(Xvalidate,gm_label0.mu(:,1), gm_label0.Sigma(:,:,1))...
               + gm_label0.ComponentProportion(2)*evalGaussian(Xvalidate,gm_label0.mu(:,2), gm_label0.Sigma(:,:,2));
    discriminantScores = log(g1_10kTraining) - log(g0_10kTraining);
    
    % determine achivable theortical value at discriminant location
    y_theoretical = length(Xtrain_L0)/length(Xtrain_L1)
    tau_theoretical = log(y_theoretical)
    [Pfp,Ptp,Perror,thresholdList] = ROCcurve(discriminantScores,Lvalidate);
     % find theoretical operating point
    decision = (discriminantScores >= tau_theoretical); % LDA threshold not optimized to minimize its own E[Risk]!
    Pfp_Theoretical = length(find(decision==1 & Lvalidate==0))/length(find(Lvalidate==0));
    Ptp_Theoretical = length(find(decision==1 & Lvalidate==1))/length(find(Lvalidate==1));
    theoretical_minError = sum(decision~=Lvalidate)/length(Lvalidate),
    % find emperical operating point
    [minError, minError_Ind] = min(Perror),
    tau_minError = thresholdList(minError_Ind)
     
    figuNum = 1 + figuNum;
    figure(figuNum)
    plot(Pfp,Ptp,'b-', Pfp_Theoretical, Ptp_Theoretical, 'ro', Pfp(minError_Ind), Ptp(minError_Ind), 'g+'), 
    xlabel('P(False+)'),ylabel('P(True+)'), title(['ROC CURVE: ',num2str(Ntrain(i)),' point EM Parameter Esitimation validated using D^{20k}_{validate}']),
    legend('ROC Curve', 'Theoretical', 'Emperical');

end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PART 3: Linear and Quadratic Logistic Function Calculations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for  i=length(Ntrain):-1:1
    % data and label at current training sample set
    Xtrain = Samples.x{i};
    Xlabel = Samples.label{i};
    
    % linear logistic function clasification
    x_Linear = formLogisticX('L', Xtrain, Ntrain(i)); % generate the lenear logisitic data
    thetaInit_Linear = zeros(size(x_Linear,2), 1); % initial parameter values
    
    % use MATLAB 'fminsearch' to calculate minimum likelihood value and Optimize model
    options = optimset('MaxFunEvals',1e4*length(thetaInit_Linear)); % Matlab default is 200*length(thetaInit_Linear)
    label = double(Xlabel)';
    theta_Linear = fminsearch(@(thetaParam)(objectiveFunction(thetaParam, x_Linear, Xlabel, Ntrain(i)))...
                  ,thetaInit_Linear, options);
    % select points to draw line for boundary
    point_X1 = [min(x_Linear(:,2))-2, max(x_Linear(:,2))+2];
    point_X2 = (-1./theta_Linear(3)).*(theta_Linear(2).*point_X1+theta_Linear(1));
    boundaryLine = [point_X1;point_X2];
    % lineararize validation data with a leading constant 1 evaulated at zero
    % point of cost function
    test_set_Linear = [ones(Nvalidation, 1), Xvalidate'];
    decision_Linear = (test_set_Linear*theta_Linear >= 0)';
    Ptn = length(find(decision_Linear==0 & Lvalidate==0))/length(find(Lvalidate==0));
    Pfp = length(find(decision_Linear==1 & Lvalidate==0))/length(find(Lvalidate==0));
    Ptp = length(find(decision_Linear==1 & Lvalidate==1))/length(find(Lvalidate==1));
    Pfn = length(find(decision_Linear==0 & Lvalidate==1))/length(find(Lvalidate==1));
    Perror_Linear = sum(decision_Linear~=Lvalidate)/length(Lvalidate),
    
    figuNum = 1+figuNum;
    figure(figuNum), % class 0 circle, class 1 +, correct green, incorrect red
    ind00 = find(decision_Linear==0 & Lvalidate==0); % probability of true negative
    ind10 = find(decision_Linear==1 & Lvalidate==0); % probability of false positive
    ind01 = find(decision_Linear==0 & Lvalidate==1); % probability of false negative
    ind11 = find(decision_Linear==1 & Lvalidate==1); % probability of true positive
    plot(Xvalidate(1,ind00),Xvalidate(2,ind00),'og'); hold on,
    plot(Xvalidate(1,ind10),Xvalidate(2,ind10),'or'); hold on,
    plot(Xvalidate(1,ind01),Xvalidate(2,ind01),'+r'); hold on,
    plot(Xvalidate(1,ind11),Xvalidate(2,ind11),'+g'); hold on,
    plot(boundaryLine(1,:), boundaryLine(2,:))
    axis equal,
    legend('Correct Class 0', 'Wrong Class 0', 'Wrong Class 1', 'Correct Class 1', 'Classifier boundry'), 
    title(['Linear Logistic Classifier with ', num2str(Ntrain(i)),' training samples validated with D^{20k}_{validate}']),
    xlabel('x_1'), ylabel('x_2'),
    
    x_Quadratic = formLogisticX('Q', Xtrain, Ntrain(i));
    thetaInit_Quadratic = zeros(size(x_Quadratic,2), 1); % initial parameter values
    % use MATLAB 'fminsearch' to calculate minimum likelihood value and Optimize model
    options = optimset('MaxFunEvals',1e4*length(thetaInit_Quadratic)); % Matlab default is 200*length(thetaInit_Quadratic)
    theta_Quad = fminsearch(@(thetaParam)(objectiveFunction(thetaParam, x_Quadratic, Xlabel, Ntrain(i)))...
                  ,thetaInit_Quadratic);%,options);
    % Quadratic validation data with a leading constant 1 evaulated at zero
    % point of cost function
    test_set_Quad = [ones(Nvalidation, 1), Xvalidate(1,:)', Xvalidate(2,:)', (Xvalidate(1,:).^2)', (Xvalidate(1,:).*Xvalidate(2,:))', (Xvalidate(2,:).^2)'];
    decision_Quad = (test_set_Quad*theta_Quad >= 0)';
    Ptn = length(find(decision_Quad==0 & Lvalidate==0))/length(find(Lvalidate==0));
    Pfp = length(find(decision_Quad==1 & Lvalidate==0))/length(find(Lvalidate==0));
    Ptp = length(find(decision_Quad==1 & Lvalidate==1))/length(find(Lvalidate==1));
    Pfn = length(find(decision_Quad==0 & Lvalidate==1))/length(find(Lvalidate==1));
    Perror_Quad = sum(decision_Quad~=Lvalidate)/length(Lvalidate),
    
    figuNum = 1+figuNum;
    figure(figuNum), % class 0 circle, class 1 +, correct green, incorrect red
    ind00 = find(decision_Quad==0 & Lvalidate==0); % probability of true negative
    ind10 = find(decision_Quad==1 & Lvalidate==0); % probability of false positive
    ind01 = find(decision_Quad==0 & Lvalidate==1); % probability of false negative
    ind11 = find(decision_Quad==1 & Lvalidate==1); % probability of true positive
    plot(Xvalidate(1,ind00),Xvalidate(2,ind00),'og'); hold on,
    plot(Xvalidate(1,ind10),Xvalidate(2,ind10),'or'); hold on,
    plot(Xvalidate(1,ind01),Xvalidate(2,ind01),'+r'); hold on,
    plot(Xvalidate(1,ind11),Xvalidate(2,ind11),'+g'); hold on,
    axis equal,
    legend('Correct Class 0', 'Wrong Class 0', 'Wrong Class 1', 'Correct Class 1'), 
    title(['Quadratic Logistic Classifier with ', num2str(Ntrain(i)),' training samples validated with D^{20k}_{validate}']),
    xlabel('x_1'), ylabel('x_2'),
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%&&
%%% Some functions provided by Professor Deniz 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%&&

function [x,labels] = generateDataFromGMM(N,gmmParameters)
    % Generates N vector samples from the specified mixture of Gaussians
    % Returns samples and their component labels
    % Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters.priors; % priors should be a row vector
    meanVectors = gmmParameters.meanVectors;
    covMatrices = gmmParameters.covMatrices;
    n = size(gmmParameters.meanVectors,1); % Data dimensionality
    C = length(priors); % Number of components
    x = zeros(n,N); labels = zeros(1,N); 
    % Decide randomly which samples will come from each component
    u = rand(1,N); thresholds = [cumsum(priors),1];
    for l = 1:C
        indl = find(u <= thresholds(l)); Nl = length(indl);
        labels(1,indl) = l*ones(1,Nl);
        u(1,indl) = 1.1*ones(1,Nl); % these samples should not be used again
        x(:,indl) = mvnrnd(meanVectors(:,l),covMatrices(:,:,l),Nl)';
    end
end

%%% provided by professor Deniz
function g = evalGaussian(x,mu,Sigma)
    % Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
    [n,N] = size(x);
    C = ((2*pi)^n * det(Sigma))^(-1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
    g = C*exp(E);
end

%%% provided by professor Deniz
function [Pfp,Ptp,Perror,thresholdList] = ROCcurve(discriminantScores,labels)
    [sortedScores,ind] = sort(discriminantScores,'ascend');
    thresholdList = [min(sortedScores)-eps,(sortedScores(1:end-1)+sortedScores(2:end))/2, max(sortedScores)+eps];
    for i = 1:length(thresholdList)
        tau = thresholdList(i);
        decisions = (discriminantScores >= tau);
        Ptn(i) = length(find(decisions==0 & labels==0))/length(find(labels==0));
        Pfp(i) = length(find(decisions==1 & labels==0))/length(find(labels==0));
        Ptp(i) = length(find(decisions==1 & labels==1))/length(find(labels==1));
        Pfn(i) = length(find(decisions==0 & labels==1))/length(find(labels==1));
        Perror(i) = sum(decisions~=labels)/length(labels);
% alternate for Totatl error calulation        Perror(i) = Pfp(i)*(length(find(labels==0)))/length(labels)+Pfn(i)*(length(find(labels==1)))/length(labels);
    end
end

function [data, labels] = myGMM(PRIORS,N, GMMParams)
%   Inputs:
%       PRIORS: distribution of labels as priors
%       N: sample quantity
%       GMMParams: GMM structure with parameters
    
    % make sure distibution sums to one
    if(sum(PRIORS)~=1) 
        disp("PRIORS DO NOT SUM TO 1");
        return;
    end
    
    n = size(GMMParams(1).meanVectors,1); % Data dimensionality
    data = zeros(n,N); labels = zeros(1,N); 
    u = rand(1,N); thresholds = [cumsum(PRIORS),1];
    L = length(PRIORS); % Number of components
    for l = 1:L
        indl = find(u <= thresholds(l)); Nl = length(indl);
        labels(1,indl) = l*ones(1,Nl);
        u(1,indl) = 1.1*ones(1,Nl); % these samples should not be used again
        [data(:,indl), ~] = generateDataFromGMM(Nl, GMMParams(l));
    end
end

% 2-d samples
function plotSamples(figNum, graphCount, Samples)
    figure(figNum)
    subWidth = ceil(sqrt(graphCount)); 
    subHeight = ceil(graphCount/subWidth);
    for i=1:graphCount
        subplot(subWidth,subHeight,i), hold on
        x = Samples.x{:,i};
        label = Samples.label{i};
        plot(x(1,label==0), x(2,label==0), 'b*'),
        plot(x(1,label==1), x(2,label==1), 'r*'),  
        axis equal, hold off
        xlabel('x_1'); ylabel('x_2'); title("D_{train}^{"+size(x,2)+"}");
    end
    legend('L=0','L=1'),
    title("D_{validation}^{"+size(x,2)+"}")
end

function LogisticX = formLogisticX(type, x, N)
    % Forms a logistic polynomial model matrix with N values in x up to
    % polynomial power DEG   
    d = size(x,1); % x data dimension    
    if type == 'L'
        LogisticX = zeros(1+d,N); LogisticX(1,:) = ones(1,N);        
        LogisticX((1:d)+1,:) = x;
        LogisticX = LogisticX';
    end    
    if type == 'Q'
        LogisticX = zeros(1+d*d,N); LogisticX(1,:) = ones(1,N);  
        LogisticX((1:d)+1,:) = x;
        LogisticX(4,:) = x(1,:).*x(1,:); 
        LogisticX(5,:) = x(1,:).*x(2,:);
        LogisticX(6,:) = x(2,:).*x(2,:);
        LogisticX = LogisticX';
    end
end

% function from Spring 2020 Assignemnt #2
function cost = objectiveFunction(theta, x, label, N)
    % Cost function to be minimized to get best fitting parameters
    h = 1./(1+exp(-x*theta)); % Sigmoid function
    cost = (-1/N)*((sum(label*log(h)))+(sum((1-label)*log(1-h))));
end