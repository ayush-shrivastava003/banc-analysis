import pandas as pd
import numpy as np
from scipy import sparse
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm

# unlike connections_princeton.csv, all rows are unique edges but have not been filtered
edge_list = pd.read_feather('data/banc_888_edgelist_simple_v3.feather').query('count >= 3')
attrs = pd.read_feather('data/banc_888_meta.feather')

proofread_ids = attrs[(attrs['proofread'].str.upper() == 'TRUE') | (attrs['roughly_proofread'].str.upper() == 'TRUE')]['banc_888_id']
attrs = attrs[attrs['banc_888_id'].isin(proofread_ids)]
edge_list = edge_list[(edge_list['pre'].isin(proofread_ids)) & (edge_list['post'].isin(proofread_ids))]

edge_list['pre'] = edge_list['pre'].astype(np.int64)
edge_list['post'] = edge_list['post'].astype(np.int64)
all_neurons = pd.concat([edge_list['pre'], edge_list['post']]).unique()
id_to_idx = {root_id: i for i, root_id in enumerate(all_neurons)}
N = len(all_neurons)
print(f"{N} neurons in BANC")

pre_ids = edge_list['pre'].map(id_to_idx).values.astype(np.int32)
post_ids = edge_list['post'].map(id_to_idx).values.astype(np.int32)

G = sparse.csr_matrix(((np.ones(shape=len(pre_ids), dtype=bool)), (pre_ids, post_ids)), shape=(N, N))
G_rev = G.transpose().tocsr()

def bfs(graph, source):
    dist_arr = np.full(graph.shape[0], -1, np.int32)
    current = np.zeros(graph.shape[0], bool)

    dist_arr[source] = 0
    current[source] = True
    depth = 0

    while current.any():
        depth += 1
        next = np.asarray(current @ graph).ravel() > 0 
        next = next & (dist_arr == -1)

        if not next.any():
            break

        dist_arr[next] = depth
        current = next

    return dist_arr

def shortest_paths(source, target):
    source_indices = np.array([id_to_idx[s] for s in source if s in id_to_idx])
    target_indices = np.array([id_to_idx[t] for t in target if t in id_to_idx])

    is_looping_source = len(source_indices) <= len(target_indices) # find the smaller neuron population to reduce number of BFS calls

    small_indices = source_indices if is_looping_source else target_indices
    large_indices = target_indices if is_looping_source else source_indices
    union_mask = np.zeros(len(pre_ids), bool)

    for s in small_indices:
        if is_looping_source:
            distance = bfs(G, np.array([[s]]))
            valid = (distance[pre_ids] >= 0) & (distance[post_ids] >= 0) & (distance[post_ids] == distance[pre_ids] + 1)
        else:
            distance = bfs(G_rev, np.array([[s]]))
            valid = (distance[pre_ids] >= 0) & (distance[post_ids] >= 0) & (distance[pre_ids] == distance[post_ids] + 1)

        dag_pre_ids = pre_ids[valid]
        dag_post_ids = post_ids[valid]

        G_dag = sparse.csr_matrix((np.ones(len(dag_pre_ids), bool), (dag_pre_ids, dag_post_ids)), shape=(N, N))

        if is_looping_source:
            reach = bfs(G_dag.transpose().tocsr(), large_indices) >= 0
            keep = reach[dag_post_ids]
        else:
            reach = bfs(G_dag, large_indices) >= 0
            keep = reach[dag_pre_ids]

        edge_ok = valid.copy()
        edge_ok[valid] = keep

        union_mask |= edge_ok

    return edge_list[union_mask].copy()

# Define the neuron populations
oviDNs = pd.read_csv('data/ovidns.csv')['root_id'].astype(int).tolist()
CAs = pd.read_csv('data/ca.csv')['root_id'].astype(int).tolist()
ORNs = pd.read_csv('data/yeast_orns.csv')['root_id'].astype(int).tolist()

# Define the combinations of source and target groups
combinations = [
    (ORNs, oviDNs, 'ORN_to_oviDN'),
    (oviDNs, CAs, 'oviDN_to_CA'),
    (CAs, oviDNs, 'CA_to_oviDN'),
    (ORNs, CAs, 'ORN_to_CA'),
]
# Initialize an empty list to store all shortest path edges
all_shortest_path_edges = []

# Iterate over each combination and get the shortest path edges
for source, target, label in combinations:
    edges = shortest_paths(source, target)
    edges['direction'] = label  # Add a column indicating the direction
    all_shortest_path_edges.append(edges)

# Concatenate all shortest path edges into a single DataFrame
master_edge_list = pd.concat(all_shortest_path_edges, ignore_index=True)

# Save the master edge list to a CSV file (optional)
master_edge_list.to_csv('out/master_edge_list_with_direction.csv', index=False)

# ---- Enrichment analysis ----
attrs['neurotransmitter_predicted'] = attrs['neurotransmitter_predicted'].fillna('Unknown')
dir_to_name = {
    'ORN_to_oviDN': 'Yeast ORN-OviDN',
    'ORN_to_CA': 'Yeast ORN-CA',
    'oviDN_to_CA': 'OviDN-CA',
    'CA_to_oviDN': 'CA-OviDN'
}

def bh_fdr(p): # Benjamini-Hochberg false discovery rate correction
    p = np.asarray(p)
    n = p.size
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)

rows = []
M = len(attrs['banc_888_id'].unique())
neurotransmitters = ['serotonin', 'dopamine', 'acetylcholine', 'gaba', 'glutamate', 'octopamine', 'tyramine', 'histamine', 'Unknown']
attrs['banc_888_id'] = attrs['banc_888_id'].astype(np.int64)
print(M)
for direction, name in dir_to_name.items():
    e = master_edge_list[master_edge_list['direction'] == direction]
    print(f"{e.shape[0]} edges in {name} circuit")
    ids = set(e["pre"]) | set(e["post"])
    print(len(ids), "neurons in", name, "circuit")
    sub = attrs[attrs["banc_888_id"].isin(ids)]
    print(f"{sub.shape[0]} neurons in {name} circuit")
    N = len(sub)
    for nt in neurotransmitters:
        K = int((attrs["neurotransmitter_predicted"] == nt).sum())
        k = int((sub["neurotransmitter_predicted"] == nt).sum())
        exp = N * K / M
        p = fisher_exact([[k, N - k], [K - k, M - N - (K - k)]],
                            alternative="two-sided")[1]
        rows.append(dict(circuit=name, key=direction, NT=nt, N_circuit=N, k=k,
                            K_brain=K, expected=exp, fold=(k / N) / (K / M),
                            log2fc=np.log2(max(k, 0.5) / exp), p=p))
df = pd.DataFrame(rows)
df["q"] = bh_fdr(df["p"].values)
df.to_csv('out/enrichment_analysis.csv', index=False)

plt.rcParams['font.family'] = 'Arial'
nt_labels = {'serotonin': 'Serotonin', 'dopamine': 'Dopamine', 'acetylcholine': 'Acetylcholine', 'gaba': 'GABA', 'glutamate': 'Glutamate',
             'octopamine': 'Octopamine', 'tyramine': 'Tyramine', 'histamine': 'Histamine', 'Unknown': 'Unknown'}
fig, ax = plt.subplots(figsize=(7.0, 3.0))  # sized so cells are roughly square and text fills them
circuits = list(dir_to_name.values())
log2fcs = df.pivot(index='circuit', columns='NT', values='log2fc').loc[circuits, neurotransmitters]
k_values = df.pivot(index='circuit', columns='NT', values='k').loc[circuits, neurotransmitters]
q_values = df.pivot(index='circuit', columns='NT', values='q').loc[circuits, neurotransmitters]
M = log2fcs.where(k_values > 0).clip(-3, 3).values
im = ax.imshow(M, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0, vmin=-3, vmax=3), aspect='auto')

def stars(q):
    return "***" if q < 1e-3 else "**" if q < 1e-2 else "*" if q < 0.05 else ""

for i in range(len(circuits)):
    for j in range(len(neurotransmitters)):
        k = k_values.values[i, j]
        s = stars(q_values.values[i, j])
        if k == 0:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='0.82', edgecolor='white', lw=.8, zorder=2))
            ax.text(j, i - 0.08, '0 obs', ha='center', va='center', fontsize=8, zorder=3)
            ax.text(j, i + 0.26, s, ha='center', va='center', fontsize=7, zorder=3)
        else:
            v = log2fcs.values[i, j]
            text_color = 'white' if abs(np.clip(v, -3, 3)) > 1.9 else 'black'
            ax.text(j, i - 0.08, f"{v:+.2f}", ha='center', va='center', fontsize=8, color=text_color)
            ax.text(j, i + 0.26, s, ha='center', va='center', fontsize=7, color=text_color)

ax.set_xticks(range(len(neurotransmitters)), [nt_labels[nt] for nt in neurotransmitters], fontsize=8, rotation=35, ha='right', rotation_mode='anchor')
ax.set_yticks(range(len(circuits)), circuits, fontsize=8)
ax.set_xticks(np.arange(-0.5, len(neurotransmitters), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(circuits), 1), minor=True)
ax.grid(which='minor', color='white', lw=0.8)
ax.tick_params(which='both', length=0)
for s in ax.spines.values():
    s.set_visible(False)
color_bar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03, shrink=0.9, ticks=[-3, 0, 3])
color_bar.set_label('log$_2$ fold change', fontsize=8, labelpad=2)
color_bar.ax.set_yticklabels(['\u2264 \u22123', '0', '\u2265 +3'], fontsize=8)
color_bar.ax.tick_params(length=2)
color_bar.outline.set_visible(False)
fig.tight_layout()
fig.savefig('out/enrichment_analysis.png', dpi=800, facecolor='white')
