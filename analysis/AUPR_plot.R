library(readr)
library(RColorBrewer)
library(scales)
library(fmsb)

# Read in data
why_lstm = read_csv("res/why_lstm.csv")
# Clean data

data_sample = why_lstm[,2:14]
rownames(data_sample) = why_lstm$experiment_name
#c(0.596,0.426,0.645,0.499,0.551,0.832,0.458,0.555,0.803,0.835,0.285,0.927,0.943)
data_sample = rbind( c(0.072,0.088,0.103,0.027,0.034,0.074,0.128,0.046,0.079,0.132,0.04,0.046,0.123,0.016),data_sample)#PTM
# data_sample = rbind( c(0.039,0.079,0.068,0.035,0.056,0.07,0.028,0.115,0.203,0.073,0.22,0.107,0.058),data_sample)#OPTM
data_sample = data_sample[1,]
data_sample = rbind(rep(1,13) , rep(0,13) , data_sample)
rownames(data_sample)[3] = 'Baseline'

data_sample = rbind(data_sample, c(0.764,0.418,0.202,0.256,0.439,0.738,0.353,0.426,0.564,0.502,0.218,0.881,0.427))
data_sample = rbind(data_sample, c(0.9,0.469,0.65,0.613,0.644,0.919,0.478,0.586,0.863,0.856,0.416,0.929,0.97))
# data_sample = rbind(data_sample, c(0.736,0.652,0.535,0.213,0.32,0.377,0.691,0.743,0.623,0.745,0.633,0.744,0.59))
# rownames(data_sample)[6] = '15 fold'
rownames(data_sample)[5] = 'Multilabel'
rownames(data_sample)[4] = 'Single label'

colnames(data_sample) = c('Hydro_K', 'Hydro_P', 'Me_K', 'Me_R', 'N6-Ac_K', 'Palm_C', 'Phos_ST', 'Phos_Y',
'Pyro_Q', 'SUMO_K','Ub_K','Glyco_N', 'Glyco_ST')#PTM
# colnames(data_sample) = c('Hydro_R', 'Hydro_N', 'Hydro_D', 'Cys4HNE_C', 'CysSO2H_C', 'CysSO3H_C', 'Lys2AAA_K', 'MetO2_M',
# 'MetO_M', 'Hydro_F','Hydro_W','Hydro_Y', 'Hydro_V')#OPTM
# Figure setup
coul = brewer.pal(nrow(data_sample)-2, "Set1")
colors_border = coul
colors_in = alpha(coul, 0.3)
# plot with default options:
pdf('figures/Fig2B.pdf', width = 10, height = 7)
radarchart(data_sample, axistype = 1 , pty = 32,
           # custom polygon
           pcol = colors_border , pfcol = colors_in , plwd = 3 , plty=1,#plty = c(2,rep(1,4)),
           # custom the grid
           cglcol = "grey", cglty = 1, axislabcol = "grey", 
           caxislabels = seq(0,1,0.25), cglwd = 0.8,
           # custom labels
           vlcex=0.8 
)
# Add a legend
legend(x = 1.4, y = 1, legend = rownames(data_sample)[-c(1,2)], bty = "n", lty=1, lwd=2,
       pch = 19 , col = colors_border , text.col = "black", cex = 1, pt.cex = 0.5)
dev.off()
