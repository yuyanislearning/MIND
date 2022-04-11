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
data_sample = rbind( c(0.088,0.103,0.027,0.034,0.074,0.128,0.046,0.079,0.132,0.04,0.046,0.123,0.016),data_sample)
data_sample = rbind(rep(1,13) , rep(0,13) , data_sample)
rownames(data_sample)[3] = 'Baseline'
data_sample = rbind(data_sample, c(0.778,0.491,0.635,0.553,0.612,0.782,0.477,0.629,0.83,0.866,0.389,0.92,0.918))
data_sample = rbind(data_sample, c(0.112,0.394,0.608,0.34,0.456,0.803,0.419,0.608,0.773,0.808,0.218,0.917,0.05))
rownames(data_sample)[5] = 'Single label'
rownames(data_sample)[4] = 'Multilabel'
# Figure setup
coul = brewer.pal(nrow(data_sample)-2, "Set1")
colors_border = coul
colors_in = alpha(coul, 0.3)
# plot with default options:
pdf('figures/AUPR_all.pdf', width = 10, height = 7)
radarchart(data_sample, axistype = 1 , 
           # custom polygon
           pcol = colors_border , pfcol = colors_in , plwd = 3 , plty = c(2,rep(1,4)),
           # custom the grid
           cglcol = "grey", cglty = 1, axislabcol = "grey", 
           caxislabels = seq(0,1,0.25), cglwd = 0.8,
           # custom labels
           vlcex=0.8 
)
# Add a legend
legend(x = 1.4, y = 1, legend = rownames(data_sample)[-c(1,2)], bty = "n", lty=c(2,rep(1,4)), lwd=2,
       pch = 19 , col = colors_border , text.col = "black", cex = 1, pt.cex = 0.5)
dev.off()
