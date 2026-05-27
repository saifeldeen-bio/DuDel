# !/bin/bash
refBams=$1 # text File containing the path to the reference panel (bam files)
testBams=$2 # text File containing the path to the test samples (bam files)
bedFile=$3
mkdir -p ./ReferencesCounts ./TestCounts
for refbam in $(cat $refBams);do 
sampleID=$(basename $refbam '.bam')
echo "chrom start   end gene    count" > ./ReferencesCounts/$sampleID.txt
bedtools coverage -a $bedFile -b $refbam | awk '{print $1"\t"$2"\t"$3"\t"$4"\t"$5}' >> ./ReferencesCounts/$sampleID.txt
done
for testbam in $(cat $testBams);do 
sampleID=$(basename $testbam '.bam')
echo "chrom start   end gene    count" > ./TestCounts/$sampleID.txt
bedtools coverage -a $bedFile -b $testbam | awk '{print $1"\t"$2"\t"$3"\t"$4"\t"$5}' >> ./TestCounts/$sampleID.txt
done
