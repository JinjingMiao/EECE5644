clear;clc;
x0=0;
y0=0; %determine the original point
r=1;
for K=1:4 %set one loop to create plot
    if K==1
        angle=2*pi;
    elseif K==2
        angle=[0,0.5]*2*pi;
    elseif K==3
        angle=[0,1/3,2/3]*2*pi;
    elseif K==4
        angle=[0,1/4,2/4,3/4]*2*pi;
    end
xi = r*cos(angle); 
yi = r*sin(angle);    %determine the reference point   
rtrue=r*sqrt(rand(1,1));
seta=2*pi*rand(1,1);
xtrue=x0+rtrue*cos(seta);
ytrue=y0+rtrue*sin(seta); %determine the true point

x=linspace(-2,2);
y=linspace(-2,2);
[X,Y] = meshgrid(x,y);

n=normrnd(0,0.3,1,K);%determine the n parameter
c=0;
sigmaX=0.25;
sigmaY=0.25;
sig=0.3;

for i=1:K
    ri=abs((xtrue-xi(i)).^2+((ytrue-yi(i)).^2)).^(0.5)+n(i);
    while (ri<=0)
          ri=abs((xture-xi(i)).^2+((ytrue-yi(i)).^2)).^(0.5)+normrnd(0,0.3,1,1);
    end     %if matlab reminds here is one error, just type some spcae before ri
     c=c-((ri-distance(xi(i),yi(i),X,Y)).^2)/(2*(sig^2));
end %calculate the map
Z=-0.5*((X.^2/sigmaX^2)+(Y.^2/sigmaY^2))+c;
figure;
contour(X,Y,Z,'ShowText','on');hold on;
scatter(xi,yi,25,'r','filled'),hold on;
scatter(xtrue,ytrue,'k','+');hold on;
xlabel('x');ylabel('y');
title(['Coutour of MAP when K=',num2str(K)]);
end
function dis = distance(a,b,c,d)
dis=(abs((a-c).^2+(b-d).^2).^(0.5));
end