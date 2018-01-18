%x=-5:10; y=-5:10;
x = -4:0.1:4;
y = 1/sqrt(2*pi)*exp(-x.^2/2);
plot(x,y); axis off; hold on;
plot([0 0],[min(y) max(y)],'k',[min(x) max(x)],[0 0],'k');
ax=[max(x),0.99*max(x),0.99*max(x);0,0.01*(max(y)-min(y)),-0.01*(max(y)-min(y))];
fill(ax(1,:),ax(2,:),'k');
ay=[0,0.15,-0.15;max(y),max(y)-0.4,max(y)-0.4];
fill(ay(1,:),ay(2,:),'k'); hold on
for i=1:length(x)-1
    if x(i)~=0
        plot([x(i),x(i)],[0,0.1],'k'); hold on
        %a=text(x(i),-0.4,num2str(x(i)));
        set(a,'HorizontalAlignment','center')
    end
    if y(i)~=0
        plot([0,0.1],[y(i),y(i)],'k'); hold on
        %b=text(-0.4,y(i),num2str(y(i)));
        set(b,'HorizontalAlignment','center')
    end
end
c=text(-0.4,-0.4,num2str(0));
set(c,'HorizontalAlignment','center')