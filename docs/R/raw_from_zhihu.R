rm(list=ls(all=TRUE))
library("ggplot2")
library("gridExtra")
library("reshape2")
d = seq(0,10*pi,0.1)
datax<- sin(d)
dataxo<-sin(d)
datay<- cos(d) 
datayo<- cos(d)
dataz<- 0.5*sin(d)
datazo<- 0.5*sin(d)

datax[100:102]<- 3
datax[120]<- 2.7
datax[150]<- 6
datay[120]<- 7
datay[220]<- 4
dataz[50]<- 6
dataz[140]<- 3
dataz[200]<- 5

length <-length(datax)
x.matrix <- cbind(1:length, datax,dataxo)
colnames(x.matrix) <- c("number", "datax","dataxo")
plotdatax <- as.data.frame(x.matrix)


y.matrix <- cbind(1:length, datay, datayo)
colnames(y.matrix) <- c("number", "datay", "datayo")
plotdatay <- as.data.frame(y.matrix)

z.matrix <- cbind(1:length, dataz, datazo)
colnames(z.matrix) <- c("number", "dataz", "datazo")
plotdataz <- as.data.frame(z.matrix)

px <- ggplot()+
  geom_point(data = plotdatax, aes(x= number , y= datax), color = "red", size=2) +
  geom_line(data = plotdatax, aes(x=number,y=dataxo),color = "black", size = 1)

py <- ggplot()+
  geom_point(data = plotdatay, aes(x= number, y = datay), color = "orange",size=2) +
  geom_line(data = plotdatay, aes(x=number,y=datayo),color = "black",size=1)

pz <- ggplot()+
  geom_point(data = plotdataz, aes(x= number, y = dataz), color = "green",size=2) +
  geom_line(data = plotdataz, aes(x=number,y=datazo),color = "black",size=1)

grid.arrange(px,py,pz, ncol=1, nrow=3)