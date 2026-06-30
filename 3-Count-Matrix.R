library(Rsamtools)
library(dplyr)
library(GenomicRanges)
library(edgeR)
library(ggplot2)
args <- commandArgs(trailingOnly = TRUE)

test.samples <- args[1]
ref.samples <- args[2]
out.dir <- args[3]
dir.create(out.dir)
my.test.samples <- list.files(test.samples)
my.ref.samples <- list.files(ref.samples)

for (test_sample in my.test.samples){
  results <- data.frame(
    test_sample = character(),
    ref_sample  = character(),
    R2          = numeric(),
    stringsAsFactors = FALSE
  )
  for (ref_sample in my.ref.samples){
    test.counts <- paste0(test.samples,'/',test_sample)
    ref.counts  <- paste0(ref.samples,'/',ref_sample)
    
    test.count <- read.csv(test.counts, sep = "\t", header=TRUE, row.names=4)
    ref.count  <- read.csv(ref.counts, sep = "\t", header=TRUE, row.names=4)
    
    test.count$exonLength <- test.count$end - test.count$start
    ref.count$exonLength <- ref.count$end - ref.count$start
    
    test.exonLength <- test.count$exonLength
    ref.exonLength  <- ref.count$exonLength
    
    test.count$exonLength <- NULL
    test.count$start <- NULL
    test.count$end   <- NULL
    test.count$chrom <- NULL
    
    ref.count$exonLength <- NULL
    ref.count$start <- NULL
    ref.count$end   <- NULL
    ref.count$chrom <- NULL
    
    # rpkm normalization
    test.dge <- DGEList(counts = test.count, genes = data.frame(Length = test.exonLength))
    ref.dge  <- DGEList(counts = ref.count,  genes = data.frame(Length = ref.exonLength))
    
    test.rpkm <- rpkm(test.dge)
    ref.rpkm  <- rpkm(ref.dge)
    
    counts <- data.frame(test = test.rpkm, ref = ref.rpkm)
    
    # Linear model to calculate R²
    lm_fit <- lm(count.1 ~ count, data = counts)
    R2 <- summary(lm_fit)$r.squared
    
    # Save result to results data.frame
    results <- rbind(results, data.frame(test_sample, ref_sample, R2))
    # pdf(paste0("/cbio/projects/003/saifeldeen/samples/tools_benchmarking/tools/nmdscan/correlations/female/", test_sample,"_",ref_sample,".pdf"), width = 6, height = 6)
    # # Plot with R² annotation
    # ggplot(counts, aes(x = count, y = count.1)) +
    #   geom_point(alpha = 0.6) +
    #   geom_smooth(method = "lm", se = FALSE, color = "red") +
    #   labs(
    #     x = test_sample,
    #     y = ref_sample,
    #     title = paste("Exon-level correlation (R² =", round(R2, 3), ")")
    #   ) +
    #   theme_minimal()
    # dev.off()
  }
  # Write results to CSV
  write.csv(results,
            paste0(out.dir, '/', test_sample,".csv"),
            row.names = FALSE)
}

############### Generate COmbined Dataset
  # 
for (test_sample in my.test.samples){
  Final_results <- data.frame(
    Gene=character(),
    Counts=numeric(),
    Ref1=numeric(),
    Ref2=numeric(),
    Ref3=numeric(),
    Ref4=numeric(),
    Ref5=numeric(),
    Ref6=numeric(),
    Ref7=numeric(),
    Ref8=numeric(),
    Ref9=numeric(),
    Ref10=numeric(),
    meanRef=numeric())
  test.corr <- read.csv(paste0(out.dir, '/', test_sample,".csv"), sep=',', header = TRUE)
  test.corr <- test.corr[ order (test.corr$R2, decreasing = TRUE),]
  test.counts <- paste0(test.samples,'/',test_sample)
  test.count <- read.csv(test.counts, sep = "\t", header=TRUE, row.names=4)
  Data <- data.frame(
    Gene=row.names(test.count),
    chrom=test.count$chrom,
    start=test.count$start,
    end=test.count$end,
    counts=test.count$count
  )
  
  topRef <- test.corr$ref_sample[1:10]
  counter <- 1
  for (ref_sample in topRef){
    print(ref_sample)
    ref.counts <- paste0(ref.samples,'/',ref_sample)
    ref.count  <- read.csv(ref.counts, sep = "\t", header=TRUE, row.names=4)
    Data[,paste0("Ref",counter)] <- ref.count$count
    counter <- counter + 1
  }
  
  Data <- Data[rowSums(Data[, 6:15] < 10) <= 3, ]
  
  Data$exonLength <- Data$end - Data$start
  
  exonLengths <- Data$exonLength
  
  raw.counts <- Data[, c("Gene", "counts", paste0("Ref", 1:10))]
  
  row.names(raw.counts) <- raw.counts$Gene
  raw.counts$Gene <- NULL
  
  # rpkm normalization
  dge <- DGEList(counts = raw.counts, genes = data.frame(Length = exonLengths))
  rpkm <- rpkm(dge)
  rpkm <- as.data.frame(rpkm)
  rpkm$meanRef <- rowMeans(rpkm[,2:11])
  rpkm <- cbind(Gene = rownames(rpkm), rpkm)
  rownames(rpkm) <- NULL
  Final_results <- rbind(Final_results, rpkm)
  # Write results to CSV
  write.csv(Final_results,
            paste0(out.dir,'/', gsub(pattern = ".txt", replacement = "", x = test_sample),'_count_matrix.csv'),
            row.names = F)
  
}
