## ################################################################################################
## refer to 
##  https://stackoverflow.com/questions/18394391/r-custom-legend-for-multiple-layer-ggplot
## ################################################################################################

require(ggplot2)
x=seq(1,10,length=100)
data=data.frame(x,dnorm(x,mean=6.5,sd=1))
names(data)=c('x','new.data')
x.ribbon=seq(1,10,length=20)
ribbon=data.frame(x.ribbon,
                  dnorm(x.ribbon,mean=5,sd=1)+.01,
                  dnorm(x.ribbon,mean=5,sd=1)-.01,
                  dnorm(x.ribbon,mean=5,sd=1))
names(ribbon)=c('x.ribbon','max','min','avg')
#ggplot()+geom_ribbon(data=ribbon,aes(ymin=min,ymax=max,x=x.ribbon),fill='lightgreen')+
#  geom_line(data=ribbon,aes(x=x.ribbon,y=avg),color='black')+
#  geom_line(data=data,aes(x=x,y=new.data),color='red')+
#  xlab('x')+ylab('density')

ggplot()+#geom_ribbon(data=ribbon,aes(ymin=min,ymax=max,x=x.ribbon,fill='lightgreen'))+
  geom_line(data=ribbon,aes(x=x.ribbon,y=avg,color='black'))+
  geom_line(data=data,aes(x=x,y=new.data,color='red'))+
  xlab('x')+ylab('density') + 
  #scale_fill_identity(name = 'the fill', guide = 'legend',labels = c('m1')) +
  scale_colour_manual(name = 'the colour', 
                      values =c('black'='black','red'='red'), labels = c('c2','c1'))