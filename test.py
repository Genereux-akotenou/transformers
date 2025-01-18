from concurrent.futures import ProcessPoolExecutor
import os
import pandas as pd
from tqdm import tqdm
import gzip
from Bio import SeqIO

storage_folder = "./Data/RawCompletGenonme/"
etl_folder_raw = "./Data/ORF/"
etl_folder = "./Data/TransformedORFSeq2Seq2-lite/"

# Function to process a single row
def process_row(row, storage_folder, etl_folder_raw):
    folder_name = f"{row['#assembly_accession']}_{row['asm_name']}"
    folder_path = os.path.join(storage_folder, folder_name)
    batch_result = []

    if not os.path.exists(folder_path):
        return None

    # Process GFF file
    gff_path = os.path.join(folder_path, "genomic.gff.gz")
    if not os.path.exists(gff_path):
        return None

    columns = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
    with gzip.open(gff_path, 'rt') as f:
        lines = [line.strip() for line in f if not line.startswith('#')]
    gff_data = [line.split('\t') for line in lines]
    gff_df = pd.DataFrame(gff_data, columns=columns)
    gff_df["start"] = pd.to_numeric(gff_df["start"])
    gff_df["end"] = pd.to_numeric(gff_df["end"])
    cds_df = gff_df[gff_df["type"] == "CDS"]

    # Process FNA file
    fna_path = os.path.join(folder_path, "genomic.fna.gz")
    if not os.path.exists(fna_path):
        return None

    etl_folder_raw_custom = etl_folder_raw+"/"+folder_name
    subprocess.run(['orfipy', fna_path, '--dna', 'genome.fa', '--outdir', etl_folder_raw_custom], capture_output=True, text=True)
    orf_file_path = os.path.join(etl_folder_raw_custom, "genome.fa")

    # Parse ORF file
    for record in SeqIO.parse(orf_file_path, "fasta"):
        header = record.description
        sequence = str(record.seq)
        stop_codon = # extrcat from headernit loke 'stop:TAA'
        orf_start, orf_end = map(int, header.split("[")[1].split("]")[0].split("-"))
        orf_strand = # extrcat the string it like (+) ot (-) in the header
        label = 0

        # i wanna ckech if start and end are between a range of a gene. and they are in the same strand and mod(start_seq, 3) == mode(start_gene, 3)
        # then label is 1 else by defult it 0
        
        batch_result.append({"sequence": sequence, "label": label})

    # Clean up
    #for file in os.listdir(etl_folder_raw_custom):
    #    os.remove(os.path.join(etl_folder_raw_custom, file))

    return folder_name, batch_result

# Parallel processing
annotated_summary__ = annotated_summary_.sample(frac=0.8, random_state=42)
batch = 1
batch_took = 0
batch_size = 1
batch_results = []
with ProcessPoolExecutor(max_workers=40) as executor:
    futures = []
    for i, row in tqdm(annotated_summary__.iterrows(), total=len(annotated_summary__)):
        futures.append(executor.submit(process_row, row, storage_folder, etl_folder_raw))

    # Collect results
    for future in tqdm(futures, desc="Collecting Results"):
        result = future.result()
        if result:
            folder_name, batch_result = result
            batch_results.extend(batch_result)
            batch_took += 1
            if batch_took >= batch_size:
                batch_folder = os.path.join(etl_folder, f"batch_{batch}")
                os.makedirs(batch_folder, exist_ok=True)
                batch_data = pd.DataFrame(batch_results)
                # change here ----
                #label_0 = batch_data[batch_data['label'] == 0]
                #label_1 = batch_data[batch_data['label'] == 1]
                #if len(label_1) > len(label_0):
                #    label_1 = label_1.sample(len(label_0), random_state=42)
                #else:
                #    label_0 = label_0.sample(len(label_1), random_state=42)
                #balanced_data = pd.concat([label_0, label_1]).sample(frac=1, random_state=42)
                #train_size = int(0.8 * len(balanced_data))
                #train_data = balanced_data.iloc[:train_size]
                #dev_data   = balanced_data.iloc[train_size:]
                #train_data.to_csv(os.path.join(batch_folder, "train.csv"), index=False)
                #dev_data.to_csv(os.path.join(batch_folder, "dev.csv"), index=False)
                # ----------------
                train_size = int(0.8 * len(batch_data))
                train_data = batch_data.iloc[:train_size]
                dev_data = batch_data.iloc[train_size:]
                train_data.to_csv(os.path.join(batch_folder, "train.csv"), index=False)
                dev_data.to_csv(os.path.join(batch_folder, "dev.csv"), index=False)

                # Update
                batch += 1
                batch_took = 0
                batch_results = []
                #break # for test purpose

                # Cleanup
                del batch_data, train_data, dev_data
                gc.collect()

    # Save any remaining data as the last batch
    if batch_results:
        batch_folder = os.path.join(etl_folder, f"batch_{batch}")
        os.makedirs(batch_folder, exist_ok=True)
        batch_data = pd.DataFrame(batch_results)
        train_size = int(0.8 * len(batch_data))
        train_data = batch_data.iloc[:train_size]
        dev_data = batch_data.iloc[train_size:]
        train_data.to_csv(os.path.join(batch_folder, "train.csv"), index=False)
        dev_data.to_csv(os.path.join(batch_folder, "dev.csv"), index=False)
        
        #    folder_name, batch_result = result
        ##    batch_folder = os.path.join(etl_folder, folder_name)
        #    os.makedirs(batch_folder, exist_ok=True)
        #    batch_data = pd.DataFrame(batch_result)
        #    train_size = int(0.8 * len(batch_data))
        #    train_data = batch_data.iloc[:train_size]
        #    dev_data = batch_data.iloc[train_size:]
        #    train_data.to_csv(os.path.join(batch_folder, "train.csv"), index=False)
        #    dev_data.to_csv(os.path.join(batch_folder, "dev.csv"), index=False)