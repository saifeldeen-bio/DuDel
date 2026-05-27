# !/bin/bash
GFFFile=$1
GenesList=$2
out=$3

printf 'chrom\tstart\tend\tgene\n' > $out".bed"
for Gene in $(cat $GenesList | uniq); do
echo "Processing: $Gene"
#Extract exon coordinates for the gene from the GFF file
grep -w "$Gene" "$GFFFile" | awk '$3 == "exon" && $0 ~ /gene='$Gene'(;|$)/' | \
     awk '{print $1"\t"$4-1"\t"$5"\t"$9"\t.\t"$7}' | grep "NM" | grep "Ensembl" | \
     awk '{match($1, /[0-9]+/); chr=substr($1, RSTART, RLENGTH) + 0; print "chr" chr "\t" $2 "\t" $3 "\t" "'$Gene'_"NR}' | awk '!seen[$2]++' | \
 awk '
 BEGIN { prev = 0 }
 {
     split($4, a, "_")
     exon_num = a[2] + 0
     if (exon_num - prev <= 1) {
         print
         prev = exon_num
     }
 }' | \
 awk '{
     if ($1 == "chr23") $1 = "chrX";
     print $0
 }' OFS='\t' >> $out".bed"
 done
