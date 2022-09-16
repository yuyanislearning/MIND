library(ggplot2)
library(tidyverse)
library(RColorBrewer)
my_col = brewer.pal(3, 'Set1')
dat = data.frame(seq=c(0.3846153846,	0.4536923077,	0.5406153846,			0.6879230769),#0.6164615385,
                 graph=c(0.3279230769,	0.4155384615,	0.4836923077,		0.5194615385),#0.5313076923,
                 both=c(0.3601538462,	0.4850769231,	0.5600769231,	0.7019230769),#0.6136153846,
                 proportion=c(0.1,0.3,0.5,1))#,0.7

dat = dat %>% gather('type','AUPRC', -proportion)
# png('temp.png')#,width = 720, height = 720)
pdf('ablation.pdf')
ggplot(dat, aes(x=proportion, y=AUPRC, col=type, group=type))+
  geom_line()+geom_point()+xlab('proportion of data used')+
  scale_color_manual(labels=c('sequence + structure', 'structure only','sequence only'), values =my_col)

dev.off()
