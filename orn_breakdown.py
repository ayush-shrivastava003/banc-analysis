import pandas as pd
import plotly.graph_objects as go

yeast_orns = pd.read_csv('data/yeast_orns.csv')['root_id'].astype(int)
ovidns = pd.read_csv('data/ovidns.csv')['root_id'].astype(int)
ca = pd.read_csv('data/ca.csv')['root_id'].astype(int)
edge_list = pd.read_csv('out/master_edge_list_with_direction.csv')
edge_list['pre'] = edge_list['pre'].astype(int)
edge_list['post'] = edge_list['post'].astype(int)
attrs = pd.read_feather('data/banc_888_meta.feather')

proofread_ids = attrs[(attrs['proofread'].str.upper() == 'TRUE') | (attrs['roughly_proofread'].str.upper() == 'TRUE')]['banc_888_id'].astype(int)
yeast_orns = set(yeast_orns) & set(proofread_ids.astype(int))
print(attrs[attrs['banc_888_id'].astype(int).isin(yeast_orns)].groupby('cell_sub_class').count())

orn_ovidn_edges = edge_list[edge_list['direction'] == 'ORN_to_oviDN']
orn_ca_edges = edge_list[edge_list['direction'] == 'ORN_to_CA']

orns_ovidns_only = set(orn_ovidn_edges[orn_ovidn_edges['pre'].isin(yeast_orns)]['pre'].unique())
orns_ca_only = set(orn_ca_edges[orn_ca_edges['pre'].isin(yeast_orns)]['pre'].unique())
orns_both = orns_ovidns_only & orns_ca_only
orns_neither = yeast_orns - orns_ovidns_only - orns_ca_only - orns_both
orns_ovidns_only = orns_ovidns_only - orns_both
orns_ca_only = orns_ca_only - orns_both

# Prepare data for the pie chart
labels = ['ORN to OviDN only', 'ORN to CA only', 'Both', 'Neither']
values = [len(orns_ovidns_only), len(orns_ca_only), len(orns_both), len(orns_neither)]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0)])

fig.update_layout(
    title="Overlap of Yeast-Responding ORN Circuits",
    annotations=[dict(text="", x=0.5, y=0.5, font_size=20, showarrow=False)]
)

fig.write_image('out/orn_overlap.png')
# fig.show()