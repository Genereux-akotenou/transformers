from concurrent.futures import ProcessPoolExecutor
import os
import pandas as pd
from tqdm import tqdm
import gzip
from Bio import SeqIO
import subprocess
from intervaltree import Interval, IntervalTree

storage_folder = "./Data/RawCompletGenonme/"
etl_folder_raw = "./Data/ORF/"
etl_folder = "./Data/TransformedORFSeq2Seq2-lite/"

# Helper function to create interval trees for gene data
def create_interval_trees(cds_df):
    trees = {}
    for _, row in cds_df.iterrows():
        seqid = row["seqid"]
        if seqid not in trees:
            trees[seqid] = IntervalTree()
        trees[seqid].add(Interval(row["start"], row["end"] + 1, {"strand": row["strand"], "start_mod": row["start"] % 3}))
    return trees

# Function to process a batch of ORF records
def process_orf_batch(batch, interval_trees):
    batch_result = []
    for record in batch:
        header = record.description
        sequence = str(record.seq)
        stop_codon = header.split("stop:")[1].split()[0] 
        orf_start, orf_end = map(int, header.split("[")[1].split("]")[0].split("-"))
        orf_strand = header.split("(")[1].split(")")[0]  

        label = 0
        seqid = header.split("_ORF")[0]

        if seqid in interval_trees:
            overlaps = interval_trees[seqid][orf_start:orf_end + 1]
            for overlap in overlaps:
                gene_data = overlap.data
                if (
                    gene_data["strand"] == orf_strand and
                    orf_start % 3 == gene_data["start_mod"]
                ):
                    label = 1
                    break

        batch_result.append({"sequence": sequence+stop_codon, "label": label})
    return batch_result

# Function to process a single row
def process_row(row, storage_folder, etl_folder_raw):
    folder_name = f"{row['#assembly_accession']}_{row['asm_name']}"
    folder_path = os.path.join(storage_folder, folder_name)

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

    # Create interval trees for gene data
    interval_trees = create_interval_trees(cds_df)

    # Process FNA file
    fna_path = os.path.join(folder_path, "genomic.fna.gz")
    if not os.path.exists(fna_path):
        return None

    etl_folder_raw_custom = os.path.join(etl_folder_raw, folder_name)
    os.makedirs(etl_folder_raw_custom, exist_ok=True)
    subprocess.run(['orfipy', fna_path, '--dna', 'genome.fa', '--outdir', etl_folder_raw_custom], capture_output=True, text=True)
    orf_file_path = os.path.join(etl_folder_raw_custom, "genome.fa")

    # Parse ORF file in parallel
    orf_records = list(SeqIO.parse(orf_file_path, "fasta"))
    batch_size = 100  # Adjust batch size based on available resources
    batches = [orf_records[i:i + batch_size] for i in range(0, len(orf_records), batch_size)]

    batch_results = []
    with ProcessPoolExecutor(max_workers=20) as executor:
        for result in executor.map(process_orf_batch, batches, [interval_trees] * len(batches)):
            batch_results.extend(result)

    # Clean up
    for file in os.listdir(etl_folder_raw_custom):
        os.remove(os.path.join(etl_folder_raw_custom, file))

    return folder_name, batch_results


# Parallel processing 
# ----------------------------------------------------------------------------------------
annotated_summary__ = annotated_summary_.sample(frac=0.8, random_state=42)
batch = 1
batch_took = 0
batch_size = 1
batch_results = []
with ProcessPoolExecutor(max_workers=2) as executor:
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