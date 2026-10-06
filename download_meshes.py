"""
Stage A — runs in your NORMAL environment (not Blender's Python).

Fetches mesh geometry (not skeletons) for the BANC circuit neurons and
writes each one to disk as a .obj file. Blender never talks to CAVE/
CloudVolume directly, so this stage's only job is to produce plain mesh
files that Stage B (running inside Blender) can import with zero
network dependencies.

Source: static precomputed:// mesh mirror (same bucket you're already
using for the CNS outline in 3d_skeleton.py), not the graphene/pcg_skel
skeleton path. This is a plain CloudVolume precomputed fetch -- no
chunked-graph traversal, so it's much faster for ~600 neurons than
asking pcg_skel to skeletonize each one, and it gives you real surface
geometry instead of a wire skeleton.

If you specifically need mesh state that reflects proofreading *after*
the mirror was last built, swap MESH_SOURCE for a CAVEclient graphene
mesh fetch instead -- flagged below.
"""

import navis
from cloudvolume import CloudVolume
import pandas as pd
from pathlib import Path

# --- Config ---
MESH_SOURCE = "precomputed://gs://lee-lab_brain-and-nerve-cord-fly-connectome/neuron_meshes"
OUTPUT_DIR = "" # replace with the relevant output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Same navis/cloudvolume monkeypatch as the CloudVolume tutorial -- teaches
# cloudvolume to return navis.MeshNeuron objects directly instead of raw
# trimesh-like objects.
navis.patch_cloudvolume()

def id_to_group(circuit: str):
    yeast_orns = pd.read_csv("data/yeast_orns.csv")["root_id"].tolist()
    ovidns = pd.read_csv('data/ovidns.csv')['root_id'].tolist()
    ca = pd.read_csv('data/ca.csv')["root_id"].tolist()

    edge_list = pd.read_csv('out/master_edge_list_with_direction.csv')
    edge_list = edge_list[edge_list['direction'] == circuit]
    all_neurons = pd.concat([edge_list['pre_root_id'], edge_list['post_root_id']]).unique()
    attrs = pd.read_feather('data/banc_888_meta.feather')[['banc_888_id', 'neurotransmitter_predicted']].fillna('Unknown')

    return {root_id: "ORN" if root_id in yeast_orns else "OviDN" if root_id in ovidns else "CA" if root_id in ca else attrs[attrs['banc_888_id'] == root_id]['neurotransmitter_predicted'].iloc[0] for root_id in all_neurons}

def main():
    vol = CloudVolume(MESH_SOURCE, use_https=True, progress=False)

    # Fetch one ID at a time rather than batched. This costs some overhead
    # vs. a single vol.mesh.get(batch) call, but per-batch fetches fail
    # ENTIRELY if even one ID in the batch has no mesh in this snapshot --
    # you lose the whole batch, not just the missing ID, which is why the
    # last run silently dropped more neurons than were actually missing.
    # Per-ID fetch means a single bad ID costs you exactly that one neuron.
    for c in ['ORN_to_CA', 'ORN_to_oviDN', 'CA_to_oviDN', 'oviDN_to_CA']:
        if Path(OUTPUT_DIR / c / 'manifest.csv').exists():
            print(f"Meshes for {c} already downloaded.")
            continue
        OUTPUT_DIR.joinpath(c).mkdir(exist_ok=True)
        missing = []
        manifest = pd.DataFrame(columns=["root_id", "group", "mesh_id"])
        id2g = id_to_group(c)
        all_ids = list(id2g.keys())
        for i, root_id in enumerate(all_ids):
            try:
                m = vol.mesh.get(root_id, as_navis=True)
                # vol.mesh.get returns a NeuronList even for a single ID
                m = navis.simplify_mesh(m, F=0.2, backed='blender')
                m = m[0]
            except Exception as e:
                missing.append((root_id, str(e)))
                continue

            out_path = OUTPUT_DIR / c / f"{m.id}.obj"
            navis.write_mesh(m, str(out_path))
            manifest.loc[len(manifest)] = {
                "root_id": root_id,
                "group": id2g[root_id],
                "mesh_id": m.id,
            }

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(all_ids)} ({len(missing)} missing so far)")

        print(f"\nWrote {len(manifest)} mesh files to {OUTPUT_DIR}/")
        if missing:
            print(f"Missing/failed: {len(missing)} of {len(all_ids)} ({len(missing)/len(all_ids)*100:.1f}%)")
            missing_df = pd.DataFrame(missing, columns=["root_id", "error"])
            missing_df.to_csv(OUTPUT_DIR / c / "missing_meshes.csv", index=False)
            print(f"Full list + error messages written to {OUTPUT_DIR / c / 'missing_meshes.csv'}")
            print("Worth checking whether these are real gaps in the mesh mirror (e.g. very")
            print("recently proofread IDs not yet baked into the snapshot) vs. something else,")
            print("since silently rendering a figure with ~5% of yeast ORNs missing is worth")
            print("noting in the methods even if it doesn't change the qualitative picture.")

        # Also fetch the CNS outline mesh as background context, same as
        # 3d_skeleton.py's outline_mesh fetch -- reuse that ID once you've
        # confirmed it via the probe print in the original script.
        OUTLINE_SOURCE = "precomputed://gs://lee-lab_brain-and-nerve-cord-fly-connectome/region_outlines"
        cv_outline = CloudVolume(OUTLINE_SOURCE, use_https=True, progress=False)
        outline_id = 1  # confirm via cv_outline.mesh.meta.info, per 3d_skeleton.py's probe
        outline_mesh = cv_outline.mesh.get(outline_id, as_navis=True)
        navis.write_mesh(outline_mesh, str(OUTPUT_DIR / c / "cns_outline.obj"))
        print("Wrote cns_outline.obj")

        # Write a manifest mapping root_id -> group, so Stage B (in Blender)
        # can assign colors without needing pandas/CAVE at all. yeast_orns.csv
        # has no 'group' column of its own, so tag those rows explicitly
        # rather than relying on given_neurons.csv (which has ALL ~3,006 ORNs,
        # not just the yeast-responsive subset -- filtering by root_id keeps
        # only the ones actually fetched, which is exactly the yeast subset
        # here since yeast_orns.csv IDs are a strict subset of given_neurons).
        # written_ids = set(str(w) for w in written)
        # manifest_rows = []
        # for root_id in yeast_orns:
        #     if str(root_id) in written_ids:
        #         manifest_rows.append({"root_id": root_id, "group": "ORN"})
        # other_primary = given_neurons[
        #     given_neurons["group"].isin(["OviDN", "CA"])
        #     & given_neurons["root_id"].astype(str).isin(written_ids)
        # ]
        # manifest = pd.concat([pd.DataFrame(manifest_rows), other_primary[["root_id", "group"]]], ignore_index=True)
        manifest.to_csv(OUTPUT_DIR / c / "manifest.csv", index=False)
        print(f"Wrote manifest.csv ({len(manifest)} neurons, matching what was actually fetched)")


if __name__ == "__main__":
    main()
