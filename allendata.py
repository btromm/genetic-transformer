from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
from abc_atlas_access.abc_atlas_cache.anndata_utils import get_gene_data
import anndata
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import warnings
import requests
from matplotlib import pyplot as plt
import numpy as np

"""
This script contains functions to load data from the Allen Brain Cell Types Atlas. Takes ~5 minutes to run on already downloaded dataset.
"""

def load_preprocessed(datapath=Path('./data/abc_atlas')):
    # Check if processed data already exists
    if datapath.exists() and (datapath / 'preprocessed/gene.feather').exists() and (datapath / 'preprocessed/receptor_groups.feather').exists() and (datapath / 'preprocessed/data.feather').exists():
            print('Loading preprocessed data from disk...')
            gene = pd.read_feather(datapath / 'preprocessed/gene.feather')
            matrix = pd.read_feather(datapath / 'preprocessed/data.feather')
            receptor_groups = pd.read_feather(datapath / 'preprocessed/receptor_groups.feather')
            group_expression = pd.read_feather(datapath / 'preprocessed/group_expression.feather')
            group_colors = pd.read_feather(datapath / 'preprocessed/group_colors.feather')
            return matrix, gene, receptor_groups, group_expression, group_colors

def load_data(datapath=Path('./data/abc_atlas'), goi=None):    
    if goi is None:
        warnings.warn("goi (genes of interest) parameter is not specified. This may lead to unexpected behavior.", UserWarning)
    
    abc_cache = AbcProjectCache.from_cache_dir(datapath)

    ## Get list of genes of interest (from full 10X data)
    print('Loading gene metadata...')
    gene = abc_cache.get_metadata_dataframe(directory='WMB-10X', file_name='gene').set_index('gene_identifier')
    if goi is not None:
        gene = gene[gene['gene_symbol'].str.contains(goi, case=False, na=False)]
    gene = gene[gene['name'].str.contains('receptor|transporter|neuropeptide', case=False)]
    genes_to_remove = ['Sucnr1', 'Pgrmc1', 'Pgrmc2', 'Ghrhr', 'Gab1', 'Gab2', 'Gab3']  # Add as many as you like
    gene = gene[~gene['gene_symbol'].isin(genes_to_remove)]
    genelist = gene['gene_symbol'].tolist()

    print('Creating list of receptor groups...')
    receptor_groups = {
        "dopamine": [x for x in genelist if "Drd" in x],
        "serotonin": [x for x in genelist if "Htr" in x],
        "acetylcholine": [x for x in genelist if "Chrn" in x or "Chrm" in x],
        "norepinephrine": [x for x in genelist if "Adra" in x or "Adrb" in x],
        "glutamate": [x for x in genelist if "Grm" in x or "Gri" in x],
        "GABA": [x for x in genelist if "Gab" in x],
        "histamine": [x for x in genelist if "Hrh" in x],
        "opioid": [x for x in genelist if "Opr" in x],
        "cannabinoid": [x for x in genelist if "Cnr" in x],
        "neuropeptide": [x for x in genelist if "Np" in x],
        "transporter": [x for x in genelist if "Slc" in x or "Vmat" in x],
        "sigma": [x for x in genelist if "Sigma" in x],
        "glycine": [x for x in genelist if "Glr" in x]
    }
    
    receptor_groups = pd.DataFrame.from_dict(receptor_groups, orient="index").T.melt(var_name='group', value_name='gene_symbol').dropna(subset=['gene_symbol']).reset_index(drop=True)

    # Create directory for preprocessed data if it doesn't exist
    (datapath / 'preprocessed').mkdir(parents=True, exist_ok=True)

    # Load 10X data
    print("Loading 10X data...")
    cell_metadata_10x = abc_cache.get_metadata_dataframe(directory='WMB-10X', file_name='cell_metadata_with_cluster_annotation').set_index('cell_label')
    if datapath.exists():
        wmb_files = list(datapath.glob('**/*WMB*.h5ad')) # Check if directory exists and contains WMB files
        if len(wmb_files) > 0:
            downloaded = 1
        else:
            downloaded = 0
    else:
        raise Exception(f"Data path '{datapath}' does not exist.")

    if downloaded == 0:
        print("   Downloading 10X gene data... stand by, this will take a while. ~150GB")
        genes = get_gene_data(
            abc_atlas_cache=abc_cache,
            all_cells=cell_metadata_10x,
            all_genes=gene,
            selected_genes=genelist,
            data_type='log2')
    elif downloaded == 1:
        print("   Using previously downloaded 10X gene expression data...")
        gene_files = list(datapath.glob('**/*WMB*.h5ad'))
        genes = None
        if gene_files:
            for f in tqdm(gene_files):
                file = anndata.read_h5ad(f,backed='r') # Read the file
                pred = [x in genelist for x in file.var.gene_symbol] # Find genes of interest
                subset = file[:, file.var[pred].index].to_df()
                subset.columns=file.var[pred].gene_symbol
                if genes is None:
                    genes = subset
                else:
                    genes = pd.concat([genes, subset], axis=0)

    # Remove duplicate genes
    genes = genes[~genes.index.duplicated(keep='first')]
    matrix_10x = cell_metadata_10x.join(genes)

    ## Load (imputed) spatial transcriptomics data
    # If you haven't downloaded before, this will be ~50GB
    print('   Preprocessing spatial transcriptomics data...')

    cell_spatial = abc_cache.get_metadata_dataframe(
        directory = 'MERFISH-C57BL6J-638850',
        file_name = 'cell_metadata_with_cluster_annotation',
        dtype={"cell_label": str}
    )
    cell_spatial.rename(columns={'x': 'x_section',
                     'y': 'y_section',
                     'z': 'z_section'},
            inplace=True)
    cell_spatial.set_index('cell_label', inplace=True)

    print('      Loading spatial coordinates...')
    reconstructed_coords = abc_cache.get_metadata_dataframe(
    directory='MERFISH-C57BL6J-638850-CCF',
    file_name='reconstructed_coordinates',
    dtype={"cell_label": str}
)
    reconstructed_coords.rename(columns={'x': 'x_reconstructed',
                                     'y': 'y_reconstructed',
                                     'z': 'z_reconstructed'},
                            inplace=True)
    reconstructed_coords.set_index('cell_label', inplace=True)
    cell_spatial = cell_spatial.join(reconstructed_coords, how='inner')

    print('      Loading parcellation data...')
    parcellation_annotation = abc_cache.get_metadata_dataframe(directory='Allen-CCF-2020', file_name='parcellation_to_parcellation_term_membership_acronym')
    parcellation_annotation.set_index('parcellation_index', inplace=True)
    parcellation_annotation.columns = ['parcellation_%s'% x for x in  parcellation_annotation.columns]

    parcellation_color = abc_cache.get_metadata_dataframe(directory='Allen-CCF-2020', file_name='parcellation_to_parcellation_term_membership_color')
    parcellation_color.set_index('parcellation_index', inplace=True)
    parcellation_color.columns = ['parcellation_%s'% x for x in  parcellation_color.columns]
    cell_spatial.join(parcellation_annotation, how='inner')
    cell_spatial.join(parcellation_color, how='inner')
    
    print('      Loading spatial expression data...')
    imputed_h5ad_path = abc_cache.get_data_path('MERFISH-C57BL6J-638850-imputed', 'C57BL6J-638850-imputed/log2')
    adata = anndata.read_h5ad(imputed_h5ad_path, backed='r') # lazy load
    pred = [x in genelist for x in adata.var.gene_symbol] # Find the genes of interest
    filtered = adata.var[pred]
    spatial_subset = adata[:, filtered.index].to_df() # Actually pull the data
    spatial_subset.rename(columns=filtered.to_dict()['gene_symbol'], inplace=True)

    adata.file.close()
    del adata

    print('      Joining spatial expression with metadata...')
    matrix_spatial_impute = cell_spatial.join(spatial_subset, on='cell_label')

    print('   Checking which genes are imputed...')
    # Label which genes are imputed and which are only in the 10X data
    genespatialimpute = abc_cache.get_metadata_dataframe(directory='MERFISH-C57BL6J-638850-imputed', file_name='gene').set_index('gene_identifier')
    genespatial = abc_cache.get_metadata_dataframe(directory='MERFISH-C57BL6J-638850', file_name='gene').set_index('gene_identifier')

    gene['imputed'] = ~genespatialimpute.gene_symbol.isin(genespatial.gene_symbol)
    gene['only10x'] = ~gene.gene_symbol.isin(genespatialimpute.gene_symbol)

    print('   Checking G-protein coupling...')
    gene = gcoupling(gene)

    print('   Combining 10X and spatial data...')
    matrix_10x = matrix_10x.drop(columns=['x', 'y'])
    matrix_10x['dataset'] = '10X'
    matrix_spatial_impute['dataset'] = 'MERFISH'
    matrix = pd.concat([matrix_10x, matrix_spatial_impute], axis=0)

    print('   Assigning cell type')
    matrix = celltype(matrix)

    print('   Removing useless metadata\n  ...note: this includes most of the Allen Institute taxonomy except for Neurotransmitter') # this might be subjective !
    matrix = matrix.drop(columns=['average_correlation_score', 'barcoded_cell_sample_label',
                                    'brain_section_label', 'cell_barcode', 'class',
                                    'class_color', 'cluster', 'cluster_alias',
                                    'cluster_color', 'donor_genotype', 'donor_label',
                                    'donor_sex','entity', 'feature_matrix_label',
                                    'library_label', 'library_method', 'subclass',
                                    'subclass_color','supertype', 'supertype_color'])
    
    print('   Creating group expression matrix')
    group_expression = pd.DataFrame()
    unique_groups = receptor_groups['group'].unique()
    for group in unique_groups:
        group_genes = receptor_groups[receptor_groups['group'] == group]['gene_symbol'].tolist()
        group_genes = [g for g in group_genes if g in matrix.columns]
        if group_genes:
            group_expression[group] = matrix[group_genes].mean(axis=1)

    print('Saving neuromod colors')
    group_colors = {
        'dopamine': "#dc8a78",
        'serotonin': "#dd7878",
        'acetylcholine': "#ea76cb",
        'norepinephrine': "#8839ef",
        'glutamate': "#d20f39",
        'GABA': "#e64553",
        'histamine': "#fe640b",
        'opioid': "#df8e1d",
        'cannabinoid': "#40a02b",
        'transporter': "#179299",
        'glycine': "#04a5e5",
        'sigma': "#1e66f5"
    }
    print('Saving data to .feather')
    matrix.to_feather(datapath / 'preprocessed/data.feather')
    gene.to_feather(datapath / 'preprocessed/gene.feather')
    receptor_groups.to_feather(datapath / 'preprocessed/receptor_groups.feather')
    group_expression.to_feather(datapath / 'preprocessed/group_expression.feather')
    group_colors_df = pd.DataFrame.from_dict(group_colors, orient='index', columns=['color'])
    group_colors_df.to_feather(datapath / 'preprocessed/group_colors.feather')

    print("\n...Done! :-)")
    return matrix, gene, receptor_groups, group_expression, group_colors

def ensembl_to_uniprot(ensembl):
    """
    Given a list of Ensembl IDs, return a list of Uniprot IDs.
    """
    uniprot = []
    for id in tqdm(ensembl):
        url = f'https://rest.uniprot.org/uniprotkb/search?query={id}'
        response = requests.get(url)
        if response.ok:
            data = response.json()
            if data.get('results'):
                uniprot.append(data['results'][0]['uniProtkbId'])
            else:
                uniprot.append(None)
        else:
            # Raise an exception with meaningful info rather than silently returning None
            raise ConnectionError(f"Failed to retrieve data from Uniprot API: {response.status_code} - {response.text}")
            
    uniprot = [x.split('_')[0] if x is not None else None for x in uniprot]
    return uniprot

def gcoupling(gene):
    """
    Given a list of Uniprot IDs, return a dictionary of G-protein coupling information for each gene.
    """
    # Make a complete copy of the DataFrame
    gene = gene.copy(deep=True)
    
    # Get uniprot IDs - will raise exception if API fails
    uniprot = ensembl_to_uniprot(gene.index.tolist())
    gene.insert(1, "Uniprot", uniprot)
    gene["Selectivity"] = None
    
    # Load database
    db = pd.read_csv('./data/GProteinDB.csv')

    # Manual changes for specific receptors
    gene.loc[gene['gene_symbol'] == 'Adra1d', 'Uniprot'] = 'ADA1D'
    gene.loc[gene['gene_symbol'] == 'Adra1b', 'Uniprot'] = 'ADA1B'
    
    # Process Uniprot-based selectivity
    for rec in gene['Uniprot'].values:
        if rec is None:
            continue
        if rec in db['Receptor'].values:
            selectivity = db.loc[db['Receptor'] == rec, 'Selectivity'].iloc[0]
            gene.loc[gene['Uniprot'] == rec, 'Selectivity'] = selectivity
    
    # Process transporters - vectorized approach
    slc_mask = gene['gene_symbol'].str.contains("Slc", na=False)
    gene.loc[slc_mask, 'Selectivity'] = "Transporter"
    
    return gene

def celltype(matrix):
    # Create boolean masks for each cell type
    matrix['Astrocytes'] = matrix['supertype'].str.contains('Astro', na=False)
    matrix['Pyramidal'] = matrix['supertype'].str.contains('L2|L3|L2/L3|L4|L5|L6|L5/6|L6b', na=False)
    matrix['SST'] = matrix['supertype'].str.contains('Sst', na=False)
    matrix['PV'] = matrix['supertype'].str.contains('Pvalb', na=False)
    matrix['VIP'] = matrix['supertype'].str.contains('Vip', na=False)
    
    # Count cells of each type
    total_cells = len(matrix)
    print(f"\nTotal number of cells in dataset: {total_cells:,}")
    print("\nCell type counts and percentages:")
    for cell_type in ['Astrocytes', 'Pyramidal', 'PV', 'SST', 'VIP']:
        count = matrix[cell_type].sum()
        percentage = (count / total_cells) * 100
        print(f"{cell_type}: {count:,} cells ({percentage:.1f}% of total)")

    # Create a new column for consolidated cell type
    matrix['cell_type'] = 'Other'
    for cell_type in ['Astrocytes', 'Pyramidal', 'PV', 'SST', 'VIP']:
        matrix.loc[matrix[cell_type], 'cell_type'] = cell_type.upper() if cell_type == 'PV' else cell_type
    matrix = matrix.drop(columns=['Astrocytes', 'Pyramidal', 'PV', 'SST', 'VIP'])

    return matrix

def to_matlab(matrix, gene, receptor_groups, datapath=Path('./data/abc_atlas')):
    """
    Save data to .mat file for MATLAB compatibility.
    Preserves column and row labels for all data structures.
    """
    from scipy.io import savemat
    
    # Create dictionary with both data and column names
    gene = gene.fillna('NaN')
    matrix = matrix.fillna('NaN')
    receptor_groups = receptor_groups.fillna('NaN')

    data_dict = {
        'matrix_data': matrix.values,
        'matrix_columns': np.array(matrix.columns, dtype=object),
        'matrix_index': np.array(matrix.index, dtype=object),
        
        'gene_data': gene.values,
        'gene_columns': np.array(gene.columns, dtype=object),
        'gene_index': np.array(gene.index, dtype=object),
        
        'receptor_groups_data': receptor_groups.values,
        'receptor_groups_columns': np.array(receptor_groups.columns, dtype=object)
    }
    
    # Save to .mat file
    savemat(datapath / 'data.mat', data_dict)
    print("Data saved to MATLAB format with column labels preserved")

# TODO Update this so it works for your new data structure
def plot_section(xx=None, yy=None, cc=None, val=None, pcmap=None, 
                 overlay=None, extent=None, bcmap=plt.cm.Greys_r, alpha=1.0,
                 fig_width = 6, fig_height = 6):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_width, fig_height)

    if xx is not None and yy is not None and pcmap is not None:
        plt.scatter(xx, yy, s=0.5, c=val, marker='.', cmap=pcmap)
    elif xx is not None and yy is not None and cc is not None:
        plt.scatter(xx, yy, s=0.5, color=cc, marker='.', zorder=1)   
        
    if overlay is not None and extent is not None and bcmap is not None:
        plt.imshow(overlay, cmap=bcmap, extent=extent, alpha=alpha, zorder=2)
        
    ax.set_ylim(11, 0)
    ax.set_xlim(0, 11)
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig, ax