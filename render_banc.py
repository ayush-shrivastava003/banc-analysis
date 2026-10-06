"""
Stage B -- runs INSIDE Blender's bundled Python, headless, on Amarel.

Three modes, all through the same binary and the same scene setup:

    # the original fixed anatomical views
    blender --background --python amarel_rb.py -- \\
        --mesh-dir /scratch/$USER/out/ORN_to_CA \\
        --out-dir /scratch/$USER/renders/ORN_to_CA \\
        --engine CYCLES --device GPU

    # a turntable orbit about the anterior-posterior axis, as PNG frames
    blender --background --python amarel_rb.py -- \\
        --mesh-dir /scratch/$USER/out/ORN_to_CA \\
        --out-dir /scratch/$USER/renders/ORN_to_CA \\
        --mode orbit --frames 360 --resolution 1080 --samples 64 --resume

    # mux those frames into an mp4 (CPU only, no meshes loaded)
    blender --background --python amarel_rb.py -- \\
        --mesh-dir /scratch/$USER/out/ORN_to_CA \\
        --out-dir /scratch/$USER/renders/ORN_to_CA \\
        --mode encode --frames 360 --fps 30

The orbit stays in this file rather than a sibling script because it
shares everything that actually costs time or is easy to get wrong --
mesh import and materials, lighting, Cycles device setup, the
bounding-box-derived anatomical axis inference. A second file would have
to import from this one or duplicate it, and this repo already shows what
duplication does: render_banc.py (the copy rbjob actually runs) and
amarel_rb.py have already drifted apart.

Everything after the bare `--` is this script's own argv; Blender consumes
everything before it. This is a real gotcha the first time you hit it --
`blender --background --python foo.py --mesh-dir X` will NOT work, because
Blender itself tries to parse `--mesh-dir` and fails silently/weirdly.

This script does NOT use navis.interfaces.blender (b3d.Handler) at all --
that module is built for interactive/live sessions (h.select(), h.colorize()
assume you're clicking around in an open Blender window). For a one-shot
headless render, going through bpy directly is actually less code and more
predictable, since b3d's convenience layer buys you nothing when there's no
human in the loop to interact with the result.
"""

import bpy
import sys
import csv
import math
import argparse
from pathlib import Path
from mathutils import Vector, Matrix


def parse_args():
    # Strip everything up to and including Blender's own '--' separator.
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    # Not required for --mode encode, which only reads PNGs back off disk.
    # Checked after parsing so the error names the mode.
    p.add_argument("--mesh-dir", type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    # Both Eevee spellings accepted: Blender 4.2-4.5 called it
    # BLENDER_EEVEE_NEXT, Blender 5.0 renamed it back to BLENDER_EEVEE once
    # it became the only Eevee. resolve_engine() maps whichever you pass
    # onto whatever this Blender actually offers.
    p.add_argument("--engine", default="CYCLES",
                   choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"])
    p.add_argument("--device", default="GPU", choices=["GPU", "CPU"],
                   help="Cycles backend only. Has no effect on Eevee, which "
                        "always rasterizes on the GPU.")
    p.add_argument(
        "--compute-type",
        default=None,
        choices=["CUDA", "OPTIX", "HIP", "ONEAPI", "METAL", "NONE"],
        help="Cycles compute backend. Auto-detected from platform if not given: "
             "METAL on macOS, CUDA on Amarel's NVIDIA gpu partition. Override "
             "explicitly if auto-detection guesses wrong.",
    )
    p.add_argument("--resolution", type=int, default=2400, help="render width, px")
    p.add_argument("--resolution-y", type=int, default=None,
                   help="render height, px. Defaults to --resolution (square frame).")
    p.add_argument("--samples", type=int, default=128)

    p.add_argument("--mode", default="stills", choices=["stills", "orbit", "encode"],
                   help="stills: the fixed anatomical views (the original behaviour). "
                        "orbit: render a turntable frame sequence. "
                        "encode: mux an already-rendered frame sequence into a video "
                        "(no meshes are imported, no GPU needed -- run it on a cheap "
                        "CPU node after the orbit array job finishes).")
    p.add_argument("--frames", type=int, default=360,
                   help="Frames in one full 360-degree orbit.")
    p.add_argument("--fps", type=int, default=30, help="Playback rate of the encoded video.")
    p.add_argument("--orbit-margin", type=float, default=1.35,
                   help="Framing slack for the orbit. One value for the whole orbit, "
                        "unlike the per-view margins used by the stills.")
    p.add_argument("--orbit-reverse", action="store_true",
                   help="Orbit the other way round. Which lateral side you pass first "
                        "is not knowable from bounding-box extents alone (see "
                        "infer_axis_roles), so this is the knob to flip if the first "
                        "half of the video shows the wrong side.")
    p.add_argument("--fixed-lights", action="store_true",
                   help="Keep the lamps fixed in world space instead of co-rotating "
                        "them with the camera. Gives a 'lit room' look where the "
                        "specimen turns through light and shade -- including a mostly "
                        "backlit stretch around the ventral view. Off by default.")
    p.add_argument("--shard-index", type=int, default=0,
                   help="This task's index within a SLURM array job (0-based).")
    p.add_argument("--shard-count", type=int, default=1,
                   help="Number of array tasks splitting the orbit. Frames are dealt "
                        "out round-robin rather than in contiguous blocks, so a task "
                        "that dies leaves an evenly-sampled orbit you can still "
                        "preview, not a missing quarter.")
    p.add_argument("--resume", action="store_true",
                   help="Skip frames whose PNG already exists. rbjob runs --requeue, "
                        "so a preempted render should be restartable for free.")
    p.add_argument("--frame-dir", type=Path, default=None,
                   help="Where orbit frames live. Defaults to <out-dir>/frames.")
    p.add_argument("--video-out", type=Path, default=None,
                   help="Output video path for --mode encode. Defaults to "
                        "<out-dir>/orbit.mp4.")

    args = p.parse_args(argv)
    if args.resolution_y is None:
        args.resolution_y = args.resolution
    if args.frame_dir is None:
        args.frame_dir = args.out_dir / "frames"
    if args.video_out is None:
        args.video_out = args.out_dir / "orbit.mp4"
    if args.mode != "encode" and args.mesh_dir is None:
        p.error(f"--mesh-dir is required for --mode {args.mode}")
    if not 0 <= args.shard_index < args.shard_count:
        p.error(f"--shard-index {args.shard_index} out of range for "
                f"--shard-count {args.shard_count}")
    return args


GROUP_COLORS = {
    "ORN": (0.85, 0.15, 0.15, 1.0),
    "OviDN": (0.15, 0.75, 0.15, 1.0),
    "CA": (0.15, 0.25, 0.85, 1.0),
    "dopamine": (0.4793, 0.1912, 0.1413, 1.0),      # Dopamine (#B87969)
    "serotonin": (0.2623, 0.1221, 0.3005, 1.0),     # Serotonin (#8C6295)
    "acetylcholine": (0.3005, 0.3663, 0.6172, 1.0),     # Acetylcholine (#95A3CE)
    "gaba": (0.6654, 0.3916, 0.0648, 1.0),    # GABA (#D5A848)
    "glutamate": (0.2384, 0.3916, 0.0999, 1.0),    # Glutamate (#86A859)
    "histamine": (0.5711, 0.2016, 0.2542, 1.0),    # Histamine (#C77C8A)
    "tyrosine": (0.159, 0.3864, 0.3663, 1.0),      # Tyrosine (#6FA7A3)
    "octopamine": (0.1683, 0.107, 0.314, 1.0),       # Octopamine (#725C98)
    "Unknown": (0.5906, 0.5906, 0.5906, 1.0) # Unknown (#CACACA)
}

# Camera views expressed as ANATOMICAL labels, not raw world-axis vectors.
# The earlier version hardcoded world (0,0,1) = dorsal, which assumed the
# CNS's dorsal-ventral axis lines up with Blender's world Z -- it doesn't
# (confirmed empirically: bounding box extents were X=88.7, Y=31.6,
# Z=109.6, and a CNS's longest axis is anterior-posterior, shortest is
# dorsal-ventral -- so world Z is actually A-P here, world Y is D-V, world
# X is left-right). Rather than hardcode that one-off mapping (fragile if
# a different circuit's outline mesh or a coordinate-convention change
# shifts things), axis roles are inferred from bounding-box shape at
# render time: longest extent = anterior-posterior, shortest = dorsal-
# ventral, middle = medial-lateral. This holds for any CNS-shaped volume
# regardless of which world axis it happens to occupy.
#
# Each entry: (name, axis_role, sign, up_role). sign=+1/-1 picks which end
# of that axis the camera sits on; up_role names which anatomical axis
# should map to the camera's "up" so the roll comes out right (this is
# the fix for the earlier upside-down dorsal render, which used a fixed
# world-Y up-hint that fought the real anatomy).
CAMERA_VIEWS = [
    ("dorsal", "dorsal_ventral", -1, "anterior_posterior", "medial_lateral", "anterior_posterior"),
    ("anterior", "anterior_posterior", -1, "dorsal_ventral", "medial_lateral", "dorsal_ventral"),
    ("lateral_L", "medial_lateral", +1, "dorsal_ventral", "anterior_posterior", "dorsal_ventral"),
    ("lateral_R", "medial_lateral", -1, "dorsal_ventral", "anterior_posterior", "dorsal_ventral"),
    ("oblique", None, None, "dorsal_ventral", None, None),  # handled specially, see render_all_angles
]

MESH_SCALE = 1 / 10000  # matches the scaling convention in 3d_skeleton.py / b3d.Handler(scaling=...)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block_type in (bpy.data.meshes, bpy.data.materials):
        for block in list(block_type):
            if block.users == 0:
                block_type.remove(block)


def load_manifest(mesh_dir):
    """Returns mesh_id (str, no extension) -> group.

    Mesh IDs are NOT root IDs (navis.write_mesh names them by some
    internal/UUID-like scheme, not the root_id you passed in), so the
    manifest must carry both the mesh_id and the root_id, and this
    lookup has to key on mesh_id since that's the only thing
    import_meshes can see when it globs the directory.
    """
    mapping = {}
    manifest_path = mesh_dir / "manifest.csv"
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        if "mesh_id" not in reader.fieldnames:
            raise ValueError(
                f"manifest.csv has columns {reader.fieldnames}, expected a "
                "'mesh_id' column -- update Stage A to write the mesh "
                "ID (path.stem) alongside root_id and group."
            )
        for row in reader:
            mapping[row["mesh_id"]] = row["group"]
    return mapping


def make_material(name, rgba, emission_strength=0.15):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = 0.4
    # A little emission keeps thin ORN processes visible against dark
    # background without needing aggressive scene lighting everywhere.
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = rgba
    emit.inputs["Strength"].default_value = emission_strength
    mix = nodes.new("ShaderNodeAddShader")

    links.new(bsdf.outputs["BSDF"], mix.inputs[0])
    links.new(emit.outputs["Emission"], mix.inputs[1])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def import_meshes(mesh_dir, id_to_group, materials):
    obj_files = sorted(p for p in mesh_dir.glob("*.obj") if p.stem != "cns_outline")
    print(f"Importing {len(obj_files)} meshes...")

    imported = []
    for i, path in enumerate(obj_files):
        stem = path.stem  # matches manifest's "mesh_id" column, NOT root_id
        # Blender 5.x/4.x use wm.obj_import; older versions used
        # import_scene.obj -- if this throws AttributeError on your
        # cluster's Blender version, swap to the legacy operator.
        bpy.ops.wm.obj_import(filepath=str(path))
        obj = bpy.context.selected_objects[-1]
        obj.name = stem
        obj.scale = (MESH_SCALE, MESH_SCALE, MESH_SCALE)

        group = id_to_group.get(stem, "Other")
        mat = materials.get(group, materials.get("Other"))
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        imported.append(obj)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(obj_files)}")
            # Only meaningful in an interactive GUI session (headless
            # --background has no window to redraw and view_layer.update()
            # is a no-op cost there) -- keeps Blender's UI responsive/
            # visibly-alive during a long import run in the console instead
            # of appearing frozen for the full ~35s+ duration. Harmless to
            # leave in for headless runs, just does nothing useful there.
            if bpy.context.window_manager.windows:
                bpy.context.view_layer.update()
                for window in bpy.context.window_manager.windows:
                    for area in window.screen.areas:
                        area.tag_redraw()

    # Optionally import the CNS outline as translucent context.
    outline_path = mesh_dir / "cns_outline.obj"
    if outline_path.exists():
        bpy.ops.wm.obj_import(filepath=str(outline_path))
        outline_obj = bpy.context.selected_objects[-1]
        outline_obj.name = "cns_outline"
        outline_obj.scale = (MESH_SCALE, MESH_SCALE, MESH_SCALE)
        outline_mat = bpy.data.materials.new("outline_mat")
        outline_mat.use_nodes = True
        outline_mat.blend_method = "BLEND"
        bsdf = outline_mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.7, 0.7, 0.7, 1.0)
        bsdf.inputs["Alpha"].default_value = 0.06
        outline_obj.data.materials.append(outline_mat)
        imported.append(outline_obj)
        print("Imported cns_outline.obj")

    return imported


def setup_lighting():
    # Three-point-ish setup: key, fill, rim. Sun lamps since the scene
    # scale after MESH_SCALE is small and point-light falloff would need
    # per-scene tuning; sun lamps give consistent directional light
    # regardless of scene scale.
    def add_sun(name, rotation_deg, energy):
        bpy.ops.object.light_add(type="SUN", location=(0, 0, 0))
        light = bpy.context.object
        light.name = name
        light.data.energy = energy
        light.rotation_euler = tuple(math.radians(d) for d in rotation_deg)
        return light

    lights = [
        add_sun("key_light", (45, 0, 45), energy=3.0),
        add_sun("fill_light", (60, 0, -120), energy=1.0),
        add_sun("rim_light", (-30, 0, 160), energy=1.5),
    ]

    world = bpy.data.worlds["World"] if "World" in bpy.data.worlds else bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.01, 0.01, 0.015, 1.0)  # near-black backdrop
        bg.inputs[1].default_value = 1.0

    # Returned so the orbit can spin the lamps in lockstep with the camera.
    return lights


def detect_compute_type():
    """METAL on Apple Silicon/macOS, CUDA elsewhere (Amarel's gpu partition
    is NVIDIA). This is a reasonable default, not a guarantee -- an Intel
    Mac with an AMD GPU would also want METAL (per Blender's own default-
    device logic), and a non-NVIDIA cluster node would need something else
    entirely, so --compute-type exists to override this outright."""
    import platform
    system = platform.system()
    if system == "Darwin":
        return "METAL"
    return "CUDA"


def resolve_engine(engine):
    """Map a requested engine name onto one this Blender actually has.

    Blender 5.0 renamed BLENDER_EEVEE_NEXT back to BLENDER_EEVEE; 4.2-4.5
    used the _NEXT spelling. Assigning the wrong one raises a bare
    TypeError, which is a confusing way to find out your cluster's Blender
    is a different major version than your laptop's.

    Availability is probed by ATTEMPTING the assignment, not by reading
    bl_rna.properties["engine"].enum_items. `engine` is a dynamic enum:
    Cycles and Workbench register themselves as add-ons at runtime and
    show up only through RNA's dynamic item callback, which the Python
    enum_items accessor does not invoke. Reading it returns just
    ['BLENDER_EEVEE'] -- so an availability check based on it rejects
    CYCLES, which is very much installed.
    """
    scene = bpy.context.scene
    candidates = [engine]
    if engine.startswith("BLENDER_EEVEE"):
        for alias in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
            if alias not in candidates:
                candidates.append(alias)

    valid = "unknown"
    for cand in candidates:
        try:
            scene.render.engine = cand
        except TypeError as exc:
            valid = str(exc).split("in ")[-1]
            continue
        if cand != engine:
            print(f"Engine {engine} not in this Blender "
                  f"({bpy.app.version_string}); using {cand}.")
        return cand
    raise RuntimeError(f"Engine {engine} unavailable; this Blender offers {valid}")


def setup_render(engine, device, resolution_x, resolution_y, samples, compute_type=None):
    scene = bpy.context.scene
    engine = resolve_engine(engine)
    scene.render.engine = engine
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False

    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True

        if device == "GPU":
            compute_type = compute_type or detect_compute_type()
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = compute_type
            prefs.get_devices()
            enabled_any = False
            for d in prefs.devices:
                d.use = (d.type == compute_type)
                enabled_any = enabled_any or d.use
            if not enabled_any:
                print(f"WARNING: no Cycles device of type {compute_type} found -- "
                      f"available devices: {[(d.name, d.type) for d in prefs.devices]}. "
                      "Falling back to CPU for this render.")
                device = "CPU"
        scene.cycles.device = device
        print(f"Cycles device: {device}" + (f" ({compute_type})" if device == "GPU" else ""))
    else:
        # Eevee rasterizes through OpenGL and needs a live GPU context even
        # under --background. On a headless Linux node it goes through EGL,
        # and where that cannot be set up -- an Amarel login node, or any
        # non-gpu partition -- it fails at the first render() call with
        #   EGL Error (0x3009): EGL_BAD_MATCH
        # rather than with anything that mentions Eevee or the GPU.
        #
        # Note that --device does NOT apply to Eevee: it only picks the
        # Cycles backend. There is no CPU Eevee, so --engine BLENDER_EEVEE
        # --device CPU is not a CPU render, it is a GPU render with a
        # misleading flag. For a genuinely CPU-only box, Cycles with
        # --device CPU is the only option.
        if device == "CPU":
            print("NOTE: --device CPU has no effect on Eevee -- Eevee always "
                  "rasterizes on the GPU and needs a working GPU/EGL context. "
                  "If this render dies with EGL_BAD_MATCH, either run it on a "
                  "gpu-partition node or switch to --engine CYCLES --device CPU.")
        scene.eevee.taa_render_samples = samples


def compute_scene_bounds(objects):
    """World-space bounding box center + per-axis extents across the
    given mesh objects. Called with ONLY the CNS outline object (not the
    neuron meshes) -- the outline is a strict spatial superset of every
    neuron in any circuit subset, so it's a stable, circuit-independent
    reference frame. Framing off the neuron subset instead would make
    different circuit renders inconsistently scaled/centered against
    one another.
    """
    min_co = Vector((float("inf"),) * 3)
    max_co = Vector((float("-inf"),) * 3)

    for obj in objects:
        if obj.type != "MESH":
            continue
        # bound_box gives 8 corners in LOCAL space; matrix_world converts
        # to world space, which accounts for the per-object .scale we set
        # in import_meshes (MESH_SCALE) automatically.
        for corner in obj.bound_box:
            world_co = obj.matrix_world @ Vector(corner)
            min_co.x = min(min_co.x, world_co.x)
            min_co.y = min(min_co.y, world_co.y)
            min_co.z = min(min_co.z, world_co.z)
            max_co.x = max(max_co.x, world_co.x)
            max_co.y = max(max_co.y, world_co.y)
            max_co.z = max(max_co.z, world_co.z)

    center = (min_co + max_co) / 2
    extents = max_co - min_co  # full extent per axis, not half
    radius = extents.length / 2
    print(f"Scene bounds: center={tuple(round(c, 3) for c in center)}, "
          f"extents={tuple(round(e, 3) for e in extents)}, radius={radius:.3f}")
    return center, extents, radius


def infer_axis_roles(extents):
    """Map world axes (0=X, 1=Y, 2=Z) to anatomical roles by extent size:
    longest = anterior_posterior, shortest = dorsal_ventral, middle =
    medial_lateral. This holds for any CNS-shaped bounding volume
    regardless of which world axis it happens to land on for a given
    mesh export/coordinate convention -- confirmed against the actual
    printed extents (X=88.68, Y=31.55, Z=109.56): Z is longest (A-P),
    Y is shortest (D-V), X is middle (M-L), matching known fly CNS
    proportions (elongated A-P, thin D-V).
    Returns {role_name: (axis_index, extent_value)}.
    """
    axis_extents = sorted(enumerate(extents), key=lambda t: t[1], reverse=True)
    roles = {
        "anterior_posterior": axis_extents[0],
        "medial_lateral": axis_extents[1],
        "dorsal_ventral": axis_extents[2],
    }
    print("Inferred axis roles (role -> (axis_index, extent)):", roles)
    return roles


def unit_vector_for_axis(axis_index, sign):
    v = [0.0, 0.0, 0.0]
    v[axis_index] = float(sign)
    return Vector(v)


def frame_camera_on_target(cam_obj, target, location, up_hint="Y"):
    direction = target - location
    cam_obj.location = location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", up_hint).to_euler()

def aim_camera(cam_obj, target, location, up_vec):
    """Point a camera at `target` with `up_vec` exactly vertical in frame.

    Built from an explicit orthonormal basis rather than
    Vector.to_track_quat(), which the orbit cannot use. to_track_quat takes
    the up hint as a NAMED axis, and the orbit's up axis is anterior-
    posterior, which for this data is world Z -- so the call came out as
    to_track_quat("-Z", "Z"), naming the same axis for both track and up.
    That is degenerate: there is no roll information in it, so Blender falls
    back to a shortest-arc rotation whose roll drifts with the view
    direction. Measured over the orbit, the roll came out as exactly minus
    the orbit angle, which is why the specimen appeared to tumble about
    several axes at once instead of spinning about one.

    A Blender camera looks down local -Z with local +Y up, so local +Z is
    'back', pointing away from the target.
    """
    back = (location - target).normalized()
    right = up_vec.cross(back)
    if right.length < 1e-9:
        raise ValueError(
            "Camera up vector is parallel to the view direction -- roll is "
            "undefined. For the A-P orbit this cannot happen (the spin axis "
            "is always perpendicular to the view direction), so hitting this "
            "means the axis roles were inferred wrong."
        )
    right.normalize()
    true_up = back.cross(right)
    cam_obj.location = location
    cam_obj.rotation_euler = Matrix((
        (right.x, true_up.x, back.x),
        (right.y, true_up.y, back.y),
        (right.z, true_up.z, back.z),
    )).to_euler()


def distance_for_fov(cam_data, horizontal_extent, vertical_extent, margin=1.15):
    """Required camera distance so both extents fit in frame, with margin."""
    fov_h = cam_data.angle_x   # radians, live from lens+sensor_width
    fov_v = cam_data.angle_y   # radians, live from lens+sensor_height
    d_for_width = horizontal_extent / (2 * math.tan(fov_h / 2))
    d_for_height = vertical_extent / (2 * math.tan(fov_v / 2))
    return max(d_for_width, d_for_height) * margin


def render_all_angles(out_dir, resolution):
    out_dir.mkdir(parents=True, exist_ok=True)

    outline_obj = bpy.data.objects.get("cns_outline")
    if outline_obj is None:
        raise RuntimeError(
            "No 'cns_outline' object found -- framing needs the outline mesh "
            "as the stable reference volume (see compute_scene_bounds docstring). "
            "Check that cns_outline.obj exists in mesh-dir and import_meshes loaded it."
        )
    center, extents, radius = compute_scene_bounds([outline_obj])
    if radius == 0:
        raise RuntimeError("Computed scene radius is 0 -- bounding box degenerate, "
                            "check that the outline mesh actually has real geometry.")
    axis_roles = infer_axis_roles(extents)

    # # Distance scaled to bounding radius, with margin so the fullest
    # # extent (anterior-posterior, always the longest) still fits in frame.
    # # 1.3x the A-P half-extent is tighter than the earlier flat 2.2x*radius
    # # guess, and won't clip since it's derived from the real long axis
    # # rather than a generic sphere-fit distance.
    # ap_axis, ap_extent = axis_roles["anterior_posterior"]
    # distance = max(radius * 2.2, ap_extent * 0.75)

    bpy.ops.object.camera_add(location=(0, 0, 0))
    cam_obj = bpy.context.object
    cam_obj.name = "render_cam"
    cam_obj.data.lens = 50
    cam_obj.data.clip_end = radius * 5  # avoid far-plane clipping at these scales
    bpy.context.scene.camera = cam_obj

    for name, axis_role, sign, up_role, h_dim, v_dim in CAMERA_VIEWS:
        if name == "oblique": continue
        if axis_role is None:
            # oblique: average of anterior + one lateral direction, so it
            # doesn't depend on a single anatomical axis.
            ap_idx, _ = axis_roles["anterior_posterior"]
            ml_idx, _ = axis_roles["medial_lateral"]
            dv_idx, _ = axis_roles["dorsal_ventral"]
            dir_vec = (
                unit_vector_for_axis(ap_idx, -1) * 0.7
                + unit_vector_for_axis(ml_idx, +1) * 0.7
                + unit_vector_for_axis(dv_idx, +1) * 0.4
            ).normalized()
        else:
            axis_idx, _ = axis_roles[axis_role]
            dir_vec = unit_vector_for_axis(axis_idx, sign)

        up_axis_idx, _ = axis_roles[up_role]
        # Blender's to_track_quat up-hint wants a named axis ("X"/"Y"/"Z"),
        # not a vector -- map the anatomical up-role back to whichever
        # world axis it corresponds to.
        up_hint = "XYZ"[up_axis_idx]

        margin = 0.8 if name == "dorsal" else 1.5 if name == "anterior" else 1.2
        distance = distance_for_fov(cam_obj.data, axis_roles[h_dim][1], axis_roles[v_dim][1], margin=margin)
        location = center + dir_vec * distance
        frame_camera_on_target(cam_obj, center, location, up_hint=up_hint)

        out_path = out_dir / f"banc_{name}.png"
        bpy.context.scene.render.filepath = str(out_path)
        print(f"Rendering '{name}' from {tuple(round(c, 2) for c in location)} "
              f"(up hint: world {up_hint}) -> {out_path}")
        bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------------------
# Orbit / turntable
# ---------------------------------------------------------------------------

def effective_fov(cam_data, res_x, res_y):
    """Horizontal/vertical FOV in radians for the CURRENT render resolution.

    Deliberately not cam_data.angle_x / .angle_y, which is what
    distance_for_fov uses. Those RNA values come from sensor_width (36mm)
    and sensor_height (24mm) and ignore the render aspect entirely, so with
    sensor_fit='AUTO' -- which maps sensor_width onto the LONGER render
    dimension -- .angle_y is simply wrong for any square or portrait frame.
    At the default square 2400x2400 it reports 27.0deg vertical when the
    true vertical FOV is 39.6deg, i.e. it asks for a camera ~50% further
    back than needed.

    The stills path is left on the old behaviour on purpose: its per-view
    margins (0.8 / 1.5 / 1.2) were hand-tuned against that error, and
    correcting the FOV underneath them would silently re-frame every
    figure already rendered. The orbit is new, so it gets the right math
    and one honest margin.
    """
    sensor = cam_data.sensor_width
    lens = cam_data.lens
    if res_x >= res_y:
        width_mm, height_mm = sensor, sensor * res_y / res_x
    else:
        width_mm, height_mm = sensor * res_x / res_y, sensor
    return (2 * math.atan(width_mm / (2 * lens)),
            2 * math.atan(height_mm / (2 * lens)))


def orbit_frame_indices(n_frames, shard_index, shard_count):
    """Frames this task owns, dealt round-robin across the array."""
    return list(range(shard_index, n_frames, shard_count))


def render_orbit(args):
    """Camera turntable about the anterior-posterior axis.

    Frame 0 sits at the dorsal view, matching CAMERA_VIEWS' 'dorsal' entry
    exactly, then swings through one lateral, the ventral view, the other
    lateral and back. Because the rotation axis IS the A-P axis and the
    camera's up-hint is also the A-P axis, the up vector stays exactly
    perpendicular to the view direction for the whole sweep -- so
    to_track_quat never has to pick between two near-degenerate rolls and
    the anterior end stays pinned to the top of frame with no mid-orbit
    flip. Orbiting about either of the other two axes would not be free of
    that problem and would need a real roll-tracking rig.
    """
    frame_dir = args.frame_dir
    frame_dir.mkdir(parents=True, exist_ok=True)

    outline_obj = bpy.data.objects.get("cns_outline")
    if outline_obj is None:
        raise RuntimeError(
            "No 'cns_outline' object found -- framing needs the outline mesh "
            "as the stable reference volume (see compute_scene_bounds docstring). "
            "Check that cns_outline.obj exists in mesh-dir and import_meshes loaded it."
        )
    center, extents, radius = compute_scene_bounds([outline_obj])
    if radius == 0:
        raise RuntimeError("Computed scene radius is 0 -- bounding box degenerate, "
                           "check that the outline mesh actually has real geometry.")
    axis_roles = infer_axis_roles(extents)
    ap_idx, ap_extent = axis_roles["anterior_posterior"]
    ml_idx, ml_extent = axis_roles["medial_lateral"]
    dv_idx, dv_extent = axis_roles["dorsal_ventral"]

    bpy.ops.object.camera_add(location=(0, 0, 0))
    cam_obj = bpy.context.object
    cam_obj.name = "orbit_cam"
    cam_obj.data.lens = 50
    cam_obj.data.clip_end = radius * 5
    bpy.context.scene.camera = cam_obj

    # ONE distance for the whole orbit. Re-deriving it per frame the way the
    # stills do would dolly the camera in and out as the silhouette narrows
    # and widens, which reads as a breathing/zooming artefact rather than a
    # clean turntable.
    #
    # Vertical extent is constant: A-P is the up axis, so the A-P extent is
    # what has to fit top-to-bottom at every angle. Horizontal extent
    # sweeps between the M-L and D-V extents; for an axis-aligned box seen
    # at angle t the projected width is |ml*cos t| + |dv*sin t|, which peaks
    # at hypot(ml, dv) on the diagonal. Framing for that peak guarantees no
    # frame in the orbit clips, at the cost of the dorsal view sitting a
    # little looser than the dedicated dorsal still.
    fov_h, fov_v = effective_fov(cam_obj.data,
                                 bpy.context.scene.render.resolution_x,
                                 bpy.context.scene.render.resolution_y)
    max_horizontal = math.hypot(ml_extent, dv_extent)
    distance = max(max_horizontal / (2 * math.tan(fov_h / 2)),
                   ap_extent / (2 * math.tan(fov_v / 2))) * args.orbit_margin

    axis_vec = unit_vector_for_axis(ap_idx, -1)          # the spin axis
    start_dir = unit_vector_for_axis(dv_idx, -1)         # dorsal, per CAMERA_VIEWS
    # The A-P axis is held exactly vertical in frame for every frame. Its
    # +sign is not arbitrary: at theta=0 this reproduces the existing
    # 'dorsal' still's camera matrix to 6e-08, so frame 0 of the orbit is
    # the dorsal figure and the rest of the orbit inherits its orientation.
    up_vec = unit_vector_for_axis(ap_idx, +1)
    direction_sign = -1.0 if args.orbit_reverse else 1.0

    # Lamps are world-fixed sun lamps, so a camera-only orbit would drag the
    # specimen through the lighting and leave the ventral half of the video
    # backlit and muddy. Spinning the lamps by the same rotation keeps the
    # key/fill/rim relationship to the camera constant, which is what you
    # want for a figure turntable. --fixed-lights opts out.
    sun_lamps = [o for o in bpy.data.objects if o.type == "LIGHT"]
    base_light_rotations = {o.name: o.rotation_euler.to_matrix().to_4x4() for o in sun_lamps}

    indices = orbit_frame_indices(args.frames, args.shard_index, args.shard_count)
    print(f"Orbit: {args.frames} frames total, this task renders {len(indices)} "
          f"(shard {args.shard_index}/{args.shard_count}), distance={distance:.3f}, "
          f"spin axis=world {'XYZ'[ap_idx]}, frame-up=world {'XYZ'[ap_idx]}, "
          f"lights {'fixed' if args.fixed_lights else 'co-rotating'}")

    rendered = skipped = 0
    for i in indices:
        out_path = frame_dir / f"frame_{i:04d}.png"
        if args.resume and out_path.exists():
            skipped += 1
            continue

        # i / frames, never i / (frames - 1): frame 0 and frame `frames`
        # would be the same camera position, so rendering the endpoint too
        # gives a duplicated frame and a visible hitch on loop.
        theta = direction_sign * 2 * math.pi * i / args.frames
        rot = Matrix.Rotation(theta, 4, axis_vec)
        dir_vec = (rot @ start_dir).normalized()
        location = center + dir_vec * distance
        aim_camera(cam_obj, center, location, up_vec)

        if not args.fixed_lights:
            for lamp in sun_lamps:
                lamp.rotation_euler = (rot @ base_light_rotations[lamp.name]).to_euler()

        bpy.context.scene.render.filepath = str(out_path)
        print(f"  frame {i:04d}/{args.frames} theta={math.degrees(theta) % 360:6.2f}deg "
              f"-> {out_path.name}")
        bpy.ops.render.render(write_still=True)
        rendered += 1

    print(f"Orbit shard done: {rendered} rendered, {skipped} skipped (already present).")
    return frame_dir


def encode_video(args):
    """Mux frame_*.png into an H.264 mp4 using Blender's own bundled FFmpeg.

    Going through the sequencer rather than shelling out to ffmpeg means
    this needs nothing on Amarel beyond the same Blender binary the render
    already uses -- no `module load ffmpeg`, no conda env. If you would
    rather use a real ffmpeg, the equivalent is:

        ffmpeg -framerate 30 -i frames/frame_%04d.png \
               -c:v libx264 -pix_fmt yuv420p -crf 18 orbit.mp4

    Rendering frames first and encoding as a separate step is the point:
    rbjob runs under --requeue, and a job preempted 200 frames into a
    direct-to-video render loses all 200. PNGs on /scratch survive, and
    --resume picks up where it stopped.
    """
    frame_dir = args.frame_dir
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"No frame_*.png found in {frame_dir} -- run --mode orbit first.")

    expected = {f"frame_{i:04d}.png" for i in range(args.frames)}
    missing = sorted(expected - {f.name for f in frames})
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {args.frames} frames missing from {frame_dir}, "
            f"first few: {missing[:5]}. Re-run the orbit (with --resume) for the "
            "shards that failed before encoding, otherwise the video will jump."
        )

    scene = bpy.context.scene
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution_y
    scene.render.resolution_percentage = 100
    scene.render.fps = args.fps
    scene.frame_start = 1
    scene.frame_end = len(frames)
    scene.render.use_sequencer = True

    # Blender 5.0 split ImageFormatSettings into a media_type ('IMAGE' /
    # 'MULTI_LAYER_IMAGE' / 'VIDEO') plus a file_format that is filtered to
    # match it. With the default media_type='IMAGE', 'FFMPEG' is simply not
    # in the file_format enum and assigning it raises
    #   TypeError: enum "FFMPEG" not found in ('AVIF', 'JPEG', ... 'TIFF')
    # which reads exactly like a Blender built without FFmpeg support. It
    # is not -- the codecs are there, the enum is just scoped. media_type
    # has to be set FIRST. On 4.x there is no media_type and file_format
    # took FFMPEG directly, hence the hasattr.
    img = scene.render.image_settings
    if hasattr(img, "media_type"):
        img.media_type = "VIDEO"
    valid = {i.identifier for i in img.bl_rna.properties["file_format"].enum_items}
    if "FFMPEG" not in valid:
        raise RuntimeError(
            "This Blender really was built without FFmpeg support -- no FFMPEG "
            f"in file_format even after media_type=VIDEO (offered: {sorted(valid)}). "
            "Encode the frames with a standalone ffmpeg instead; see this "
            "function's docstring for the equivalent command line."
        )
    img.file_format = "FFMPEG"
    ff = scene.render.ffmpeg
    ff.format = "MPEG4"
    ff.codec = "H264"
    ff.constant_rate_factor = "HIGH"
    ff.ffmpeg_preset = "GOOD"
    ff.gopsize = args.fps
    ff.audio_codec = "NONE"

    # use_file_extension=False + an explicit .mp4 filepath, otherwise Blender
    # appends the frame range and you get 'orbit.mp40001-0360.mp4'.
    args.video_out.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.video_out)
    scene.render.use_file_extension = False

    seq = scene.sequence_editor_create()
    # Blender 4.4 renamed SequenceEditor.sequences to .strips; 5.x is
    # strips-only. Support both so this runs on the cluster's Blender and
    # on an older local one. hasattr, not `getattr(...) or ...`: an empty
    # bpy_prop_collection is falsy, so the `or` form would silently fall
    # through to the 5.x-nonexistent .sequences on every first run.
    strips = seq.strips if hasattr(seq, "strips") else seq.sequences
    strip = strips.new_image(name="orbit", filepath=str(frames[0]),
                             channel=1, frame_start=1)
    for f in frames[1:]:
        strip.elements.append(f.name)

    print(f"Encoding {len(frames)} frames at {args.fps} fps -> {args.video_out}")
    bpy.ops.render.render(animation=True)

    if not args.video_out.exists():
        raise RuntimeError(f"Encode reported success but {args.video_out} is not there.")
    size_mb = args.video_out.stat().st_size / 1e6
    print(f"Wrote {args.video_out} ({size_mb:.1f} MB, "
          f"{len(frames) / args.fps:.1f}s at {args.fps} fps)")


def main():
    args = parse_args()
    clear_scene()

    if args.mode == "encode":
        # Nothing to import, no GPU to configure -- this mode only reads
        # PNGs off disk, so it can run on a cheap CPU-only node.
        encode_video(args)
        print("Done.")
        return

    id_to_group = load_manifest(args.mesh_dir)
    materials = {g: make_material(f"mat_{g}", c) for g, c in GROUP_COLORS.items()}
    materials["Other"] = make_material("mat_Other", (0.6, 0.6, 0.6, 1.0))

    import_meshes(args.mesh_dir, id_to_group, materials)
    setup_lighting()
    # Must precede any framing: effective_fov reads the scene resolution to
    # work out the real aspect, so the camera distance depends on this
    # having run first.
    setup_render(args.engine, args.device, args.resolution, args.resolution_y,
                 args.samples, args.compute_type)

    if args.mode == "orbit":
        render_orbit(args)
        print("Done. Orbit frames written to", args.frame_dir)
    else:
        render_all_angles(args.out_dir, args.resolution)
        print("Done. Frames written to", args.out_dir)


if __name__ == "__main__":
    main()