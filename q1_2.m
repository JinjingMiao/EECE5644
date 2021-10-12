close all; clear ; clc;
% Modifying provided "generateData_Exam1Question1.m" script for data generation
m(:,1) = [3;0;0]; 
Sigma(:,:,1) =[2 0 0;0 1 0;0 0 2]; % mean and covariance of data pdf conditioned on label 1
m(:,2) = [0;3;0]; 
Sigma(:,:,2) = [1 0 0;0 2 0;0 0 2]; % mean and covariance of data pdf conditioned on label 2
m(:,3) = [-1;0;3]; 
Sigma(:,:,3) = [1 0 0;0 1 0;0 0 3];% mean and covariance of data pdf conditioned on label 3 part1
m(:,4) = [0;-3;1]; 
Sigma(:,:,4) = [1 0 0;0 1 0;0 0 3];% mean and covariance of data pdf conditioned on label 3 part 2
Lambda = [0 1 1; 1 0 1; 1 1 0];%loss matrix
Lambda10=[0 1 10;1 0 10;1 1 0];%loss matrix10
Lambda100=[0 1 100;1 0 100;1 1 0];%loss matrix100
classPriors = [0.3,0.3,0.2,0.2]; 
thr = [0,cumsum(classPriors)];
N = 10000; 
u = rand(1,N); 
L = zeros(1,N); 
x = zeros(3,N);
figure(1),
markerList = {'d','+','o','o'};
colorList = {'r','g','b','b'};
for l = 1:4
    indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
    plot3(x(1,indices),x(2,indices),x(3,indices),[markerList{l},colorList{l}]); hold on; 
end
axis equal;
grid on; box on;
xlabel('x_1');
ylabel('x_2');
zlabel('x_3');
title('Original data samples');
legend('True Class 1','True Class 2','True Class 3');

% Part A
% Evaluate the MPE discriminant function for each class and observation
[n,N] = size(x);
g =zeros(3,N); % will construct a discriminant scores matrix
mu_f = m;
mu_f(:,3) = (m(:,3)+m(:,4))./2;
Sigma_f = zeros(3,3,3);
Sigma_f(:,:,[1 2]) = Sigma(:,:,[1 2]);
Sigma_f(:,:,3) = (Sigma(:,:,3)+Sigma(:,:,4))./4;
for k = 1:3
    mu = mu_f(:,k);
    cov = Sigma_f(:,:,k);
    prior = classPriors(k);
    C = ((2*pi)^n * det(cov))^(-1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(cov\(x-repmat(mu,1,N))),1); 
    g(k,:) = log(C*exp(E)) + log(prior);
end
[~,k_hat] = max(g);
L(L==4) = 3;
% Present the results
Nc1 = sum(L==1); Nc2 = sum(L==2); Nc3 = sum(L==3);
fprintf('Number of samples from Class 1: %d, Class 2: %d, Class 3: %d\n', Nc1, Nc2, Nc3);
fprintf('Confusion Matrix (rows: Predicted class, columns: True class): \n'); 
confMat = confusionmat(k_hat,L); disp(confMat);
fprintf('Total number of misclassified samples: %d \n', N - sum(diag(confMat)));
Pe = (N - sum(diag(confMat)))/ N;
fprintf('Empirically Estimated Probability of Error: %.4f \n', Pe);
%calculate each situation  true1-predict1...
t1p1 = ((L==1) & (k_hat==1)); t1p2 = ((L==1) & (k_hat==2)); t1p3 = ((L==1) & (k_hat==3)); 
t2p1 = ((L==2) & (k_hat==1)); t2p2 = ((L==2) & (k_hat==2)); t2p3 = ((L==2) & (k_hat==3));  
t3p1 = ((L==3) & (k_hat==1)); t3p2 = ((L==3) & (k_hat==2)); t3p3 = ((L==3) & (k_hat==3));
figure(2),
axis equal;grid on; box on; 
%scatter3(x(1,t1p1),x(2,t1p1),x(3,t1p1));
scatter3(x(1,t1p1),x(2,t1p1),x(3,t1p1),[markerList{1},colorList{2}]); hold on;
scatter3(x(1,t2p2),x(2,t2p2),x(3,t2p2),[markerList{2},colorList{2}]); 
scatter3(x(1,t3p3),x(2,t3p3),x(3,t3p3),[markerList{3},colorList{2}]);
scatter3(x(1,t1p2),x(2,t1p2),x(3,t1p2),[markerList{1},colorList{1}]); 
scatter3(x(1,t1p3),x(2,t1p3),x(3,t1p3),[markerList{1},colorList{1}]);
scatter3(x(1,t2p1),x(2,t2p1),x(3,t2p1),[markerList{2},colorList{1}]);
scatter3(x(1,t2p3),x(2,t2p3),x(3,t2p3),[markerList{2},colorList{1}]); 
scatter3(x(1,t3p1),x(2,t3p1),x(3,t3p1),[markerList{3},colorList{1}]);
scatter3(x(1,t3p2),x(2,t3p2),x(3,t3p2),[markerList{3},colorList{1}]);
xlabel('x_1');
ylabel('x_2');
zlabel('x_3');
title('Classification Decisions: marker shape for predicted labels, color for true labels');
% Generate an informative artifical legend with dummy data
dummy(1) = plot(nan,nan,[markerList{1},colorList{2}]); 
dummy(2) = plot(nan,nan,[markerList{2},colorList{2}]);
dummy(3) = plot(nan,nan,[markerList{3},colorList{2}]);
dummy(4) = plot(nan,nan,[markerList{1},colorList{1}]);
dummy(5) = plot(nan,nan,[markerList{2},colorList{1}]);
dummy(6) = plot(nan,nan,[markerList{3},colorList{1}]);
legend(dummy, {'Class 1 with correctly classified','Class 2 with correctly classified',...
    'Class 3 with correctly classified','Class 1 with incorrectly classified',...
    'Class 2 with incorrectly classified', 'Class 3 with incorrectly classified'});



