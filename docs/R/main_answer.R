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
  geom_point(data = plotdatax, aes(x= number , y= datax, color = "red"), size=2 ) +
  geom_line(data = plotdatax, aes(x=number,y = dataxo, color = "black"), size = 1 ) +
  scale_color_manual(name = "group",
                     values = c('red' = 'red', "black" = 'black'), 
                     breaks = c("red", "black"),
                     labels = c('数据', '原始数据')) + 
  scale_shape_manual(name = "group",
                     values = c("A" = 16, "B" = 18), 
                     labels = c('数据', '原始数据')) + 
  scale_linetype_manual(name = "group",
                        values = c("A" = 0, "B" = 1), 
                        labels = c('数据', '原始数据'))
px
# pure line or point

px <- ggplot()+
  geom_point(data = plotdatax, aes(x= number , y= datax, color = "red", shape = "A", linetype = "A"), size=2 ) +
  geom_line(data = plotdatax, aes(x=number,y = dataxo, color = "black", shape = "B", linetype = "B"), size = 1 ) +
  scale_color_manual(name = "group",
                     values = c('red' = 'red', "black" = 'black'), 
                     breaks = c("red", "black"),
                     labels = c('数据', '原始数据')) + 
  scale_shape_manual(name = "group",
                     values = c("A" = 16, "B" = NA), 
                     labels = c('数据', '原始数据')) + 
  scale_linetype_manual(name = "group",
                          values = c("A" = 0, "B" = 1), 
                          labels = c('数据', '原始数据'))

px + theme(legend.title=element_blank(),
           legend.position = c(0.9, 0.9))

## more clear
length <-length(datax)
x.matrix1 <- cbind(1:length, datax, "c1")
x.matrix2 <- cbind(1:length, dataxo, "c2")
x.matrix <- rbind(x.matrix1, x.matrix2)
colnames(x.matrix) <- c("number", "datax","dataxo")
plotdatax <- as.data.frame(x.matrix)


plotdatax$group = ifelse(datax == dataxo, "c2", "c1")

px <- ggplot(plotdatax, aes(color = group, shape = group, linetype = group))+
  geom_point(aes(x= number , y= datax), size=2 ) +
  geom_line(aes(x=number,y = dataxo), size = 1 ) +
  scale_color_manual(name = "group",
                     values = c("c1" = 'red', "c2" = 'black'), 
                     labels = c('数据', '原始数据')) + 
  scale_shape_manual(name = "group",
                     values = c("c1" = 16, "c2" = NA), 
                     labels = c('数据', '原始数据')) + 
  scale_linetype_manual(name = "group",
                        values = c("c1" = 0, "c2" = 1), 
                        labels = c('数据', '原始数据'))

px + theme(legend.title=element_blank(),
           legend.position = c(0.9, 0.9))

## reconstruct dataset

length <-length(datax)
x.matrix1 <- cbind(1:length, datax)
x.matrix2 <- cbind(1:length, dataxo)
x.matrix <- rbind(x.matrix1, x.matrix2)
colnames(x.matrix) <- c("number", "datax")
plotdatax <- as.data.frame(x.matrix)

plotdatax$group <- rep(c("c1", "c2"), each = length)

px <- ggplot(plotdatax, aes(x = number, y = datax, color = group, shape = group, linetype = group))+
  geom_point(size=2) +
  geom_line(size = 1) +
  scale_color_manual(name = "group",
                     values = c("c1" = 'red', "c2" = 'black'), 
                     labels = c('数据', '原始数据')) + 
  scale_shape_manual(name = "group",
                     values = c("c1" = 16, "c2" = NA), 
                     labels = c('数据', '原始数据')) + 
  scale_linetype_manual(name = "group",
                        values = c("c1" = 0, "c2" = 1), 
                        labels = c('数据', '原始数据'))

px + theme(legend.title=element_blank(),
           legend.position = c(0.9, 0.9))


px <- ggplot()+
  geom_point(data = plotdatax, aes(x= number , y= datax, color = "red"), size=2 ) +
  geom_line(data = plotdatax, aes(x=number,y = dataxo, color = "black"), size = 1 ) +
  scale_color_manual(name = "group",
                     values = c('red' = 'red', "black" = 'black'), 
                     breaks = c("red", "black"),
                     labels = c('数据', '原始数据')) + 
  scale_shape_manual(name = "group",
                     values = c("A" = 16, "B" = NA), 
                     labels = c('数据', '原始数据')) + 
  scale_linetype_manual(name = "group",
                        values = c(0, 1), 
                        breaks = c(0, 1),
                        labels = c('数据', '原始数据'))

px #+ theme(legend.title=element_blank())

py <- ggplot()+
  geom_point(data = plotdatay, aes(x= number, y = datay), color = "orange",size=2) +
  geom_line(data = plotdatay, aes(x=number,y=datayo),color = "black",size=1)

pz <- ggplot()+
  geom_point(data = plotdataz, aes(x= number, y = dataz), color = "green",size=2) +
  geom_line(data = plotdataz, aes(x=number,y=datazo),color = "black",size=1)

grid.arrange(px,py,pz, ncol=1, nrow=3)