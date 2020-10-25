rm(list=ls())
library("FactoMineR")
library("factoextra")
data <- read.table("output.txt",header = F,skip = 1, row.names= 1)
library(hash)
h <- hash('UUU'='Phe','UUC'='Phe','UUA'='Leu','UUG'='Leu',
          'UCU'='Ser','UCC'='Ser','UCA'='Ser','UCG'='Ser',
          'UAU'='Tyr','UAC'='Tyr','UAA'='Stop','UAG'='Stop',
          'UGU'='Cys','UGC'='Cys','UGA'='Stop','UGG'='Trp',
          'CUU'='Leu','CUC'='Leu','CUA'='Leu','CUG'='Leu',
          'CCU'='Pro','CCC'='Pro','CCA'='Pro','CCG'='Pro',
          'CAU'='His','CAC'='His','CAA'='Gln','CAG'='Gln',
          'CGU'='Arg','CGC'='Arg','CGA'='Arg','CGG'='Arg',
          'AUU'='Ile','AUC'='Ile','AUA'='Ile','AUG'='Met',
          'ACU'='Thr','ACC'='Thr','ACA'='Thr','ACG'='Thr',
          'AAU'='Asn','AAC'='Asn','AAA'='Lys','AAG'='Lys',
          'AGU'='Ser','AGC'='Ser','AGA'='Arg','AGG'='Arg',
          'GUU'='Val','GUC'='Val','GUA'='Val','GUG'='Val',
          'GCU'='Ala','GCC'='Ala','GCA'='Ala','GCG'='Ala',
          'GAU'='Asp','GAC'='Asp','GAA'='Glu','GAG'='Glu',
          'GGU'='Gly','GGC'='Gly','GGA'='Gly','GGG'='Gly')
AA <- vector()
i <- 1
for (codon in row.names(data)){
  AA[i] <- h[[codon]][1]
  i <- i+1
}
data$AA <- AA

mat <- data[,-ncol(data)]
hc=hclust(dist(mat))
plot(hc)
clus = as.factor(cutree(hc, 3))
table(clus)
data$cluster <- clus
hydro <- c('Ala','Val','Leu','Ile','Pro','Phe','Trp','Met')
neural <- c('Gly','Ser','Thr','Cys','Tyr','Asn','Gln')
alkaline <- c('Lys','Arg','His')
data$polarity <- ifelse(data$AA=="Stop","none",ifelse(data$AA %in% hydro, "Hydrophobic", ifelse(data$AA %in% 
                        neural, "Neural", ifelse(data$AA %in% alkaline, "Alkaline", "Acidic"))))
# 先对氨基酸之间聚类再分配标签做PCA hclust和pheatmap
data.pca <- PCA(mat, graph = FALSE)
fviz_pca_ind(data.pca,#repel =T,
             geom.ind = "point", # show points only (nbut not "text")
             col.ind = clus, # color by groups
             addEllipses = T, # Concentration ellipses
             legend.title = "classes"
) 
ggsave('PCA.png',width = 8, height = 6)

table(data$cluster,data$AA)
data$polarity <- ifelse(data$AA=="Stop","none",ifelse(data$AA %in% hydro, "Hydrophobic","Hydrophilic"))
col <- subset(data, select = c('cluster','polarity'))
library(pheatmap)
pheatmap(t(mat),annotation_col = col,cluster_rows = FALSE,
         color = colorRampPalette(c("navy", "white", "firebrick3"))(50),
         show_rownames = FALSE)

library(Rtsne)
tsne_out <- Rtsne(mat,dims = 2, perplexity=20,
                  pca = T,
                  check_duplicates = F, ) 
tsne_plot <- data.frame(x = tsne_out$Y[,1], y = tsne_out$Y[,2], cluster = paste("Cluster",clus))
tsne_plot$codon <- row.names(mat)
ggplot(tsne_plot) + geom_label(aes(x=x,y=y,fill=cluster,label=codon),colour = "white", fontface = "bold",size = 2.5) +
    theme_bw()
ggsave('Rtsne.png',width = 8, height = 6)
