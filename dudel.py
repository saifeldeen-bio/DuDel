#import modules
import os
import sys
import argparse
import pandas as pd
import numpy as np
import colorama
import joblib
from imblearn.ensemble import BalancedRandomForestClassifier
import regex as rg
from pathlib import Path
from datetime import date
import warnings
from tqdm import tqdm
import time
######################################################## Define the Arguments #####################################################
class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        print(f'Error: {message}')
        self.print_help()
        sys.exit(2)
parser = argparse.ArgumentParser(
    description="DuDel v1.0 is a Random Forest classifier for detecting exon-level CNVs from WES data in neuromuscular disorders",
    usage='dudel -e [exon level read count] -b [exon level bed file] -m [path to RF model] -o [out directory] \n ' \
    'optional -p [Phenotypes] -v [generate vcf file] -c [ClinGen Haploinsufficiency and Triplosensitivity annotations] -G [gene predictive scores] -P [protein predictive scores] -s [summary report] -r [list of recurrent genes: txt]',
    epilog="DuDel v1.0 – Powered by Clinical Omics & Informatics Unit (COIN), Neuroscience Institute, University of Cape Town, South Africa."
)
parser.add_argument('-e', '--exonCounts', type=str, required=True,
                    help='Path to the exon-level normalized read count file from your WES sample')

parser.add_argument('-b', '--bedFile', type=str, required=True,
                    help='Path to the BED file containing exon coordinates for CNV detection')

parser.add_argument('-g', '--gender', type=str, required=True,
                    help='Sample\'s gender: [male, female]')

parser.add_argument('-m', '--model', type=str, required=True,
                    help='Path to the pre-trained Random Forest model for dudel v1.0')

parser.add_argument('-o', '--outDir', type=str, required=False,
                    help='Path to the output directory')

parser.add_argument('-r', '--recurrents', type=str, required=False,
                    help='a list of recurrent genes with common CNVs, stored in a text file and sepatrated by new lines')

parser.add_argument('-p', '--phenoData', action='store_true', default=False, required=False, 
                    help='Add the phenotype annotations for each gene [defult: False]')

parser.add_argument('-v', '--VCF', action='store_true', default=False, required=False, 
                    help='Generate vcf file [defult: False]')

parser.add_argument('-c', '--ClinGen', action='store_true', default=False, required=False, 
                    help='Add ClinGen haploinsufficiency and triplosensitivity annotations [defult: False]')

parser.add_argument('-G', '--GPS', action='store_true', default=False, required=False, 
                    help='Add gene predictive scores: LOEUF, pRec, pLI, sHet, pHaplo, and pTriplo scores [defult: False]')

parser.add_argument('-P', '--PPS', action='store_true', default=False, required=False, 
                    help='Add protein predictive scores: pDN, pGOF, and pLOF scores [defult: False]')

parser.add_argument('-s', '--summaryReport', action='store_true', default=False, required=False, 
                    help='Generate summary report for CNVs')

args = parser.parse_args()
######################################################## Create the Output Directory #####################################################
print("""
########################################
  
██████╗ ██╗   ██╗██████╗ ███████╗██╗     
██╔══██╗██║   ██║██╔══██╗██╔════╝██║     
██║  ██║██║   ██║██║  ██║█████╗  ██║     
██║  ██║██║   ██║██║  ██║██╔══╝  ██║     
██████╔╝╚██████╔╝██████╔╝███████╗███████╗
╚═════╝  ╚═════╝ ╚═════╝ ╚══════╝╚══════╝
Exon-Specific CNV caller from WES Data

#########################################                                           
""")
warnings.filterwarnings("ignore")

try:
    os.mkdir(args.outDir)
except:
    print(f"OutDir exists: {args.outDir}")
######################################################## load Datasets #############################################################
totalSteps = 3
if args.phenoData:
    totalSteps += 1
if args.ClinGen:
    totalSteps += 1
if args.GPS:
    totalSteps += 1
if args.PPS:
    totalSteps += 1
if args.VCF:
    totalSteps += 1
if args.summaryReport:
    totalSteps += 1
with tqdm(total=totalSteps, desc="Running DuDel", ncols=100) as pbar:
    tqdm.write("Loading data...")
    time.sleep(0.01)
    sample = pd.read_csv(args.exonCounts, header=0, index_col=0, sep=",")
    sample['ref_std'] = sample[['Ref1', 'Ref2', 'Ref3', 'Ref4', 'Ref5', 
                             'Ref6', 'Ref7', 'Ref8', 'Ref9', 'Ref10']].std(axis=1)
    sample['log2_ratio'] = np.log2((sample['counts'] + 0.1) / (sample['meanRef'] + 0.1))
    sample['z_score'] = (sample['counts'] - sample['meanRef']) / (sample['ref_std'] + 0.1)
    bedFile = pd.read_csv(args.bedFile, header=0, sep="\t")
    RandomForestClassifierModel = joblib.load(args.model)
    Genes = sample.index
    samplePath = Path(args.exonCounts)
    sampleID = samplePath.stem.replace("_count_matrix", "")
    pbar.update(1)
    ######################################################## CNV calling ######################################################################
    tqdm.write("CNV calling...")
    time.sleep(0.01)
    bedFile = bedFile.rename(columns={"gene": "Gene"})
    sample = sample[RandomForestClassifierModel.feature_names_in_]
    cnvs = RandomForestClassifierModel.predict(sample)
    df = pd.DataFrame(
        index=Genes,
        data=cnvs,
        columns=["cnv"])
    allCnvs = df.loc[(df['cnv'] == 0) | (df['cnv'] == 2)]
    allCnvs = allCnvs.reset_index() 
    allCnvs = pd.merge(allCnvs, bedFile, on="Gene", how="left")
    sample = sample.reset_index()
    allCnvs = pd.merge(allCnvs, sample[["Gene","counts","meanRef"]], on="Gene", how="left")
    allCnvs["Ratio"] = allCnvs["counts"] / allCnvs["meanRef"]
    allCnvs['log2_ratio'] = np.log2((allCnvs['counts'] + 0.1) / (allCnvs['meanRef'] + 0.1))
    allCnvs["Exon"] = "exon-" + allCnvs["Gene"].str.split("_").str[1]
    allCnvs["id"] = (allCnvs["chrom"].astype(str) + ":" +
        allCnvs["start"].astype(str) + "-" +
        allCnvs["end"].astype(str))
    allCnvs["Gene"] = allCnvs["Gene"].str.split("_").str[0]
    allCnvs["CNV"] = allCnvs["cnv"].map({0: "DEL", 2: "DUP"})
    allCnvs = allCnvs.drop(columns="cnv")
    deletions = allCnvs.loc[(allCnvs['CNV'] == "DEL")]
    Duplications = allCnvs.loc[(allCnvs['CNV'] == "DUP")]
    if args.gender == "male":
        deletions["Genotype"] = np.where(deletions["log2_ratio"] <= -1.1, np.where(deletions["chrom"].isin(["chrX", "chrY"]), "1/.", "1/1"), "0/1")
        deletions["CopyNumber"] = np.where(deletions["log2_ratio"] <= -1.1, "0", "1")
        Duplications["Genotype"] = np.where((Duplications["log2_ratio"] >= 0.2) & (Duplications["log2_ratio"] < 0.7), "0/1", "1/1")
        Duplications["CopyNumber"] = np.where((Duplications["log2_ratio"] >= 0.2) & (Duplications["log2_ratio"] < 0.7), "3", "4")
    else:
        deletions["Genotype"] = np.where(deletions["log2_ratio"] <= -1.1, "1/1", "0/1")
        deletions["CopyNumber"] = np.where(deletions["log2_ratio"] <= -1.1, "0", "1")
        Duplications["Genotype"] = np.where((Duplications["log2_ratio"] >= 0.2) & (Duplications["log2_ratio"] < 0.7), "0/1", "1/1")
        Duplications["CopyNumber"] = np.where((Duplications["log2_ratio"] >= 0.2) & (Duplications["log2_ratio"] < 0.7), "3", "4")
    allCnvs = pd.concat([deletions, Duplications])
    def orderChr(x):
        x = x.replace("chr", "")
        if x == "X": return 23
        if x == "Y": return 24
        return int(x)
    allCnvs=allCnvs.sort_values(by="chrom", key=lambda col: col.map(orderChr))
    pbar.update(1)

    if args.phenoData:
        tqdm.write("Add phenotyps...")
        time.sleep(0.01)
        pheno = pd.read_csv("./datasets/Phenotypes.tsv", header=0, sep="\t")
        def getPhenotype(gene):
            match = pheno[pheno['Gene'].str.contains(rf'\b{gene}\b', case=False, na=False)]
            if not match.empty:
                return "; ".join(match['Phenotype'].unique())
            else:
                return None
        allCnvs["Phenotype"] = allCnvs["Gene"].apply(getPhenotype)
        pbar.update(1)

    if args.ClinGen:
        tqdm.write("Add Haploinsufficiency & Triplosensitivity Scores...")
        time.sleep(0.01)
        ClinGenData = pd.read_csv("./datasets/ClinGen_gene_curation_list_GRCh38.tsv", header=0, sep="\t")
        def getClinGen(gene):
            match = ClinGenData[ClinGenData['Gene'].str.contains(rf'\b{gene}\b', case=False, na=False)]
            if not match.empty:
                return match[[
                    "Haploinsufficiency_Score",
                    "Triplosensitivity_Score",
                    "Haploinsufficiency_Description",
                    "Triplosensitivity_Description",
                    "Haploinsufficiency_DiseaseID",
                    "Triplosensitivity_DiseaseID"
                ]].iloc[0]
            else:
                return pd.Series(["NaN"]*6,
                    index=[
                        "Haploinsufficiency_Score",
                        "Triplosensitivity_Score",
                        "Haploinsufficiency_Description",
                        "Triplosensitivity_Description",
                        "Haploinsufficiency_DiseaseID",
                        "Triplosensitivity_DiseaseID"
                    ]
                )
        allCnvs = allCnvs.join(allCnvs["Gene"].apply(getClinGen))
        pbar.update(1)

    if args.GPS:
        tqdm.write("Add gene predictive scores...")
        time.sleep(0.01)
        GPSData = pd.read_csv("./datasets/genePredictiveScores.tsv", header=0, sep="\t")
        def getGPS(gene):
            match = GPSData[GPSData['Gene'].str.contains(rf'\b{gene}\b', case=False, na=False)]
            if not match.empty:
                return match[[
                    "lof.oe_ci.upper",
                    "lof.oe_ci.lower",
                    "lof.pRec",
                    "lof.pLI",
                    "s_het_drift",
                    "pHaplo",
                    "pTriplo"
                ]].iloc[0]
            else:
                return pd.Series(["NaN"]*7,
                    index=[
                    "lof.oe_ci.upper",
                    "lof.oe_ci.lower",
                    "lof.pRec",
                    "lof.pLI",
                    "s_het_drift",
                    "pHaplo",
                    "pTriplo"
                    ]
                )
        allCnvs = allCnvs.join(allCnvs["Gene"].apply(getGPS))
        pbar.update(1)

    if args.PPS:
        tqdm.write("Add protein predictive Scores...")
        time.sleep(0.01)
        PPSData = pd.read_csv("./datasets/proteinPredictiveScores.tsv", header=0, sep="\t")
        def getPPS(gene):
            match = PPSData[PPSData['Gene'].str.contains(rf'\b{gene}\b', case=False, na=False)]
            if not match.empty:
                return match[[
                    "pDN",
                    "pGOF",
                    "pLOF"
                ]].iloc[0]
            else:
                return pd.Series(["NaN"]*3,
                    index=[
                    "pDN",
                    "pGOF",
                    "pLOF"
                    ]
                )
        allCnvs = allCnvs.join(allCnvs["Gene"].apply(getPPS))
        pbar.update(1)

    if args.VCF:
        tqdm.write("Generate VCF...")
        time.sleep(0.01)
        with open(fr"{args.outDir}/{sampleID}.dudel.cnv.vcf", "w") as vcf:
            vcf.write("##fileformat=VCFv4.2\n")
            vcf.write("##source=dudel\n")
            vcf.write('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">\n')
            vcf.write('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">\n')
            vcf.write('##INFO=<ID=GENE,Number=.,Type=String,Description="Gene name(s)">\n')
            vcf.write('##INFO=<ID=EXON,Number=.,Type=String,Description="Exon number">\n')
            vcf.write('##FILTER=<ID=PASS,Description="All filters passed">\n')
            vcf.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
            vcf.write(f'#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sampleID}\n')
            for line in allCnvs.itertuples():
                vcf.write(f'{line.chrom}\t{line.start}\t{line.id}\tN\t<{line.CNV.upper()[0:3]}>\t.\tPASS\tEND={line.end};SVTYPE={line.CNV.upper()[0:3]};GENE={line.Gene};EXON={line.Exon}\tGT\t{line.Genotype}\n')   
        vcf.close()
        pbar.update(1)

    if args.summaryReport:
        tqdm.write("Generate summary report...")
        time.sleep(0.01)
        sortedGenes = pd.DataFrame()
        sortedGenes["Gene"] = allCnvs["Gene"].unique()
        sortedGenes["nExons"] = 0
        for gene in list(allCnvs["Gene"]):
                    GeneData = allCnvs[allCnvs['Gene'].str.contains(rf'\b{gene}\b', case=False, na=False)]
                    nExons = np.array(GeneData["Exon"].sort_values())
                    sortedGenes.loc[sortedGenes['Gene'] == gene, 'nExons'] = len(nExons)
        sortedGenes = sortedGenes.sort_values(by='nExons', ascending=False)        
        today = date.today()

    def getGeneData(DATASET, GENE):
        GeneData = DATASET[DATASET['Gene'].str.contains(rf'\b{GENE}\b', case=False, na=False)]
        pheno = "N/A"
        HapScore = HapDescription = HapDiseaseID = "N/A"
        TripScore = TripDescription = TripDiseaseID = "N/A"
        loeuf = pRec = pLI = sHet = pHaplo = pTriplo = "N/A"
        pDN = pGOF = pLOF = "N/A"
        Chrom = str(GeneData["chrom"].unique())
        Starts = np.array(GeneData["start"].sort_values())
        Ends = np.array(GeneData["end"].sort_values())
        Exons = np.array(GeneData["Exon"].sort_values())
        GeneData['exon_num'] = GeneData['Exon'].str.extract('(\d+)').astype(int)
        GeneData = GeneData.sort_values('exon_num')
        exon_cnv_dict = dict(zip(GeneData["Exon"], GeneData['CNV'].str.cat(GeneData['CopyNumber'], sep='(CN=') + ')'))
        if args.phenoData:
            pheno = str(GeneData["Phenotype"].unique())
        if args.ClinGen:
            try:
                HapScore = float(GeneData["Haploinsufficiency_Score"].unique())
            except:
                 HapScore = "NaN"
            HapDescription = str(GeneData["Haploinsufficiency_Description"].unique())
            HapDiseaseID = str(GeneData["Haploinsufficiency_DiseaseID"].unique())
            try:
                TripScore = float(GeneData["Triplosensitivity_Score"].unique())
            except:
                TripScore = "NaN"
            TripDescription = str(GeneData["Triplosensitivity_Description"].unique())
            TripDiseaseID = str(GeneData["Triplosensitivity_DiseaseID"].unique())
        if args.GPS:
            try:
                loeuf = float(GeneData["lof.oe_ci.upper"].unique())
            except:
                loeuf = "NaN"
            try:
                pRec = float(GeneData["lof.pRec"].unique())
            except:
                pRec = "NaN"
            try:
                pLI = float(GeneData["lof.pLI"].unique())
            except:
                pLI = "NaN"
            try:
                sHet = float(GeneData["s_het_drift"].unique())
            except:
                sHet = "NaN"
            try:
                pHaplo = float(GeneData["pHaplo"].unique())
            except:
                pHaplo = "Nan"
            try:
                pTriplo = float(GeneData["pTriplo"].unique())
            except:
                pTriplo = "NaN"
        if args.PPS:
            try:
                pDN = float(GeneData["pDN"].unique())
            except:
                pDN = "NaN"
            try:
                pGOF = float(GeneData["pGOF"].unique())
            except:
                pGOF = "NaN"
            try:
                pLOF = float(GeneData["pLOF"].unique())
            except:
                pLOF = "NaN"
        return (Chrom, Starts, Ends, Exons, exon_cnv_dict, pheno, 
                HapScore, HapDescription, HapDiseaseID, TripScore, 
                TripDescription, TripDiseaseID, loeuf, pRec, pLI, sHet, pHaplo, 
                pTriplo, pDN, pGOF, pLOF)

    with open(fr"{args.outDir}/{sampleID}.SummaryReport.txt", "w") as sr:
                    sr.write("""
        ##########################################################
        #                                                        #
        #                       DuDel V1.0                       #
        #         Exon-Specific CNV caller from WES Data         #
        #                                                        #
        #################### Summary Report ######################""")
                    sr.write(f"\n\nSample ID:{sampleID}\nsex:{args.gender}\nDate:{today}\nTotal CNVs:{len(allCnvs)} CNVs\nNumber of Deletions:{len(allCnvs[allCnvs['CNV'] == 'Deletion'])} DELs\nNumber of Duplications:{len(allCnvs[allCnvs['CNV'] == 'Duplication'])} DUPs\nChromosomes:{str(allCnvs['chrom'].unique())}\n\n")
                    sr.write('################ Abbreviations ################\n\n')
                    sr.write('LOEUF   ->  Loss-of-function observed/expected\n')
                    sr.write('PLI     ->  Probability of loss-of-function intolerance\n')
                    sr.write('pREC    ->  probability of being recessive\n')
                    sr.write('sHET    ->  Heterozygous loss-of-function intolerance\n')
                    sr.write('pHaplo  ->  Probability of haploinsufficiency\n')
                    sr.write('pTriplo ->  Probability of triplosensitivity\n')
                    sr.write('pDN     ->  Probability of dominant-negative mechanism\n')
                    sr.write('pGOF    ->  Probability of gain-of-function mechanism\n')
                    sr.write('pLOF    ->  Probability of loss-of-function mechanism\n')
                    sr.write('CN      ->  Copy Numbers Copy Numbers (0 = homo/hemizygous deletion, 1=single-copy loss, 3=single-copy gain, and 4=amplification)\n\n')
                    sr.write('#################### CNVs #####################\n\n')
                    if args.recurrents:
                        sr.write('#------------------- Recurrent Genes -------------------#\n\n')
                        with open(args.recurrents, "r") as re:
                            for gene in re:
                                Gene = gene.strip()
                                (Chrom, Starts, Ends, Exons, exon_cnv_dict, pheno, 
                                HapScore, HapDescription, HapDiseaseID, TripScore, 
                                TripDescription, TripDiseaseID, loeuf, pRec, pLI, sHet, pHaplo, 
                                pTriplo, pDN, pGOF, pLOF) = getGeneData(allCnvs, Gene)
                                sr.write(f'Gene:{Gene}     Chr:{Chrom}     Exons:{exon_cnv_dict}\n\nPhenotype:{pheno if pheno else "NAN"}   \nHaploinsufficiency Score:{HapScore if HapScore else "NAN"}\nHaploinsufficiency Description:{HapDescription if HapDescription else "NAN"}\n')
                                sr.write(f'Haploinsufficiency DiseaseID:{HapDiseaseID if HapDiseaseID else "NAN"}\nTriplosensitivity Score:{TripScore if TripScore else "NaN"}\nTriplosensitivity Description:{TripDescription if TripDescription else "NaN"}\nTriplosensitivity Disease ID:{TripDiseaseID if TripDiseaseID else "NaN"}\n\n')
                                sr.write(f'Gene Predictive Scores\nLOEUF:{loeuf if loeuf else "NAN"}     PLI:{pLI if pLI else "NaN"}     PREC:{pRec if pRec else "NaN"}     sHET:{sHet if sHet else "NaN"}     pHaplo:{pHaplo if pHaplo else "NaN"}     pTriplo:{pTriplo if pTriplo else "NaN"}\n\nProtein Predictive Scores\npDN:{pDN if pDN else "NaN"}     pGOF:{pGOF if pGOF else "NaN"}     pLOF:{pLOF if pLOF else "NaN"}\n\n-----------------------------------------------\n\n')
                                sortedGenes.drop(sortedGenes[sortedGenes['Gene'] == Gene].index, inplace=True)
                            sr.write('#-------------------  Other Genes -------------------#\n\n')
                            for Gene in list(sortedGenes["Gene"]):
                                (Chrom, Starts, Ends, Exons, exon_cnv_dict, pheno, 
                                HapScore, HapDescription, HapDiseaseID, TripScore, 
                                TripDescription, TripDiseaseID, loeuf, pRec, pLI, sHet, pHaplo, 
                                pTriplo, pDN, pGOF, pLOF) = getGeneData(allCnvs, Gene)
                                sr.write(f'Gene:{Gene}     Chr:{Chrom}     Exons:{exon_cnv_dict}\n\nPhenotype:{pheno if pheno else "NAN"}   \nHaploinsufficiency Score:{HapScore if HapScore else "NAN"}\nHaploinsufficiency Description:{HapDescription if HapDescription else "NAN"}\n')
                                sr.write(f'Haploinsufficiency DiseaseID:{HapDiseaseID if HapDiseaseID else "NAN"}\nTriplosensitivity Score:{TripScore if TripScore else "NaN"}\nTriplosensitivity Description:{TripDescription if TripDescription else "NaN"}\nTriplosensitivity Disease ID:{TripDiseaseID if TripDiseaseID else "NaN"}\n\n')
                                sr.write(f'Gene Predictive Scores\nLOEUF:{loeuf if loeuf else "NAN"}     PLI:{pLI if pLI else "NaN"}     PREC:{pRec if pRec else "NaN"}     sHET:{sHet if sHet else "NaN"}     pHaplo:{pHaplo if pHaplo else "NaN"}     pTriplo:{pTriplo if pTriplo else "NaN"}\n\nProtein Predictive Scores\npDN:{pDN if pDN else "NaN"}     pGOF:{pGOF if pGOF else "NaN"}     pLOF:{pLOF if pLOF else "NaN"}\n\n-----------------------------------------------\n\n')    
                            sr.write("dudel v1.0 – Powered by Clinical Omics & Informatics Unit | COIN\nNeuroscience Institute\nUniversity of Cape Town, South Africa")
                    else:
                        for Gene in list(sortedGenes["Gene"]):
                            (Chrom, Starts, Ends, Exons, exon_cnv_dict, pheno, 
                                HapScore, HapDescription, HapDiseaseID, TripScore, 
                                TripDescription, TripDiseaseID, loeuf, pRec, pLI, sHet, pHaplo, 
                                pTriplo, pDN, pGOF, pLOF) = getGeneData(allCnvs, Gene)
                            sr.write(f'Gene:{Gene}     Chr:{Chrom}     Exons:{exon_cnv_dict}\n\nPhenotype:{pheno if pheno else "NAN"}   \nHaploinsufficiency Score:{HapScore if HapScore else "NAN"}\nHaploinsufficiency Description:{HapDescription if HapDescription else "NAN"}\n')
                            sr.write(f'Haploinsufficiency DiseaseID:{HapDiseaseID if HapDiseaseID else "NAN"}\nTriplosensitivity Score:{TripScore if TripScore else "NaN"}\nTriplosensitivity Description:{TripDescription if TripDescription else "NaN"}\nTriplosensitivity Disease ID:{TripDiseaseID if TripDiseaseID else "NaN"}\n\n')
                            sr.write(f'Gene Predictive Scores\nLOEUF:{loeuf if loeuf else "NAN"}     PLI:{pLI if pLI else "NaN"}     PREC:{pRec if pRec else "NaN"}     sHET:{sHet if sHet else "NaN"}     pHaplo:{pHaplo if pHaplo else "NaN"}     pTriplo:{pTriplo if pTriplo else "NaN"}\n\nProtein Predictive Scores\npDN:{pDN if pDN else "NaN"}     pGOF:{pGOF if pGOF else "NaN"}     pLOF:{pLOF if pLOF else "NaN"}\n\n-----------------------------------------------\n\n')
                        sr.write("dudel v1.0 – Powered by Clinical Omics & Informatics Unit | COIN\nNeuroscience Institute\nUniversity of Cape Town, South Africa")
                        sr.close()
    pbar.update(1)        
    tqdm.write("Saving the results...")
    time.sleep(0.01)
    allCnvs.to_csv(fr"{args.outDir}/{sampleID}.dudel.csv")
    pbar.update(1)
print("\nDuDel v1.0 – Powered by Clinical Omics & Informatics Unit | COIN\nNeuroscience Institute\nUniversity of Cape Town, South Africa")
