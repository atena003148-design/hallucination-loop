import bpy
import sys
import os
import math
import mathutils
import random

# ============================================================
# Blender Headless Render: Valeur (Natural Tonal Gradation)
# ============================================================
# Soft, enveloping light that reveals volume through smooth
# tonal transitions rather than harsh contrast.

argv = sys.argv
if "--" not in argv:
    print("Error: input file path and output path must be provided after '--'")
    sys.exit(1)

args = argv[argv.index("--") + 1:]
if len(args) < 3:
    print("Usage: blender -b -P blender_render.py -- <input_file> <output_front> <output_back>")
    sys.exit(1)

input_file = os.path.abspath(args[0])
output_front = os.path.abspath(args[1])
output_back = os.path.abspath(args[2])

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import model
# Detect actual file format by reading header bytes
actual_format = "obj" # Default fallback
try:
    with open(input_file, "rb") as f:
        header = f.read(5)
        if header.startswith(b'glTF'):
            actual_format = "glb"
        elif header.startswith(b'{'):
            actual_format = "gltf"
        elif b'FBX' in header:
            actual_format = "fbx"
        # Otherwise, treat as obj
except Exception as e:
    print(f"Failed to read file header: {e}")

try:
    if actual_format in ('glb', 'gltf'):
        bpy.ops.import_scene.gltf(filepath=input_file)
    elif actual_format == 'fbx':
        bpy.ops.import_scene.fbx(filepath=input_file)
    else:
        try:
            bpy.ops.wm.obj_import(filepath=input_file)
        except AttributeError:
            bpy.ops.import_scene.obj(filepath=input_file)
except Exception as e:
    print(f"Failed to import file: {e}")
    sys.exit(1)

meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if not meshes:
    print("No meshes found in the imported file.")
    sys.exit(1)

# Join all meshes into one, then center and normalize
bpy.ops.object.select_all(action='DESELECT')
for mesh in meshes:
    mesh.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()

active_mesh = bpy.context.view_layer.objects.active
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
active_mesh.location = (0, 0, 0)

meshes = [active_mesh]  # Update meshes list to only contain the joined mesh

# ---- Natural Clay Material (warm grey, slight subsurface) ----
clay_mat = bpy.data.materials.new(name="NaturalClay")
clay_mat.use_nodes = True
nodes = clay_mat.node_tree.nodes
bsdf = nodes.get("Principled BSDF")
if bsdf:
    # Warm mid-grey, like unfired clay
    bsdf.inputs["Base Color"].default_value = (0.55, 0.52, 0.48, 1)
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.85  # Slightly less than full matte
    for spec_name in ["Specular IOR Level", "Specular"]:
        if spec_name in bsdf.inputs:
            bsdf.inputs[spec_name].default_value = 0.1  # Faint specular for volume cues
            break

for mesh in meshes:
    mesh.data.materials.clear()
    mesh.data.materials.append(clay_mat)

# ---- Soft Valeur Lighting (natural, enveloping) ----

# Large overhead softbox (main ambient fill from above)
soft_main = bpy.data.lights.new(name="SoftMain", type='AREA')
soft_main.energy = 300
soft_main.size = 8  # Very large = very soft shadows
soft_main.color = (1, 0.98, 0.95)  # Warm daylight
obj_main = bpy.data.objects.new(name="SoftMain", object_data=soft_main)
bpy.context.scene.collection.objects.link(obj_main)
obj_main.location = (0, 0, 4)
obj_main.rotation_euler = (0, 0, 0)  # Points straight down

# Side fill (gentle, to lift shadows without killing them)
side_fill = bpy.data.lights.new(name="SideFill", type='AREA')
side_fill.energy = 120
side_fill.size = 6
side_fill.color = (0.92, 0.95, 1.0)  # Cool complement
obj_side = bpy.data.objects.new(name="SideFill", object_data=side_fill)
bpy.context.scene.collection.objects.link(obj_side)
obj_side.location = (-3, -1, 1.5)
obj_side.rotation_euler = (math.radians(30), 0, math.radians(-60))

# Subtle backlight for edge separation (not harsh rim)
back_sep = bpy.data.lights.new(name="BackSeparation", type='AREA')
back_sep.energy = 150
back_sep.size = 5
back_sep.color = (1, 1, 1)
obj_back = bpy.data.objects.new(name="BackSeparation", object_data=back_sep)
bpy.context.scene.collection.objects.link(obj_back)
obj_back.location = (1, 3, 2)
obj_back.rotation_euler = (math.radians(-45), 0, math.radians(160))

# ---- Camera & Rendering Config ----
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'

# ---- Neutral warm background (studio feel, not black void) ----
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes.get('Background')
if bg_node:
    bg_node.inputs[0].default_value = (0.18, 0.17, 0.16, 1)  # Dark warm grey
    bg_node.inputs[1].default_value = 0.3  # Low ambient contribution

cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

def get_randomized_location(base_location, mesh_obj):
    vec = mathutils.Vector(base_location)
    dist = vec.length
    if dist == 0: return base_location
    norm_vec = vec / dist
    
    dims = mesh_obj.dimensions
    max_dim = max(dims.x, dims.y, dims.z)
    if max_dim < 0.01: max_dim = 1.0
        
    # オブジェクトが収まる安全な最小距離 (FOV≈40度) - 少し引き気味に調整
    safe_min = max_dim * 2.2
    # 最大距離も拡張して全体的に引く
    safe_max = max(5.5, max_dim * 3.5)
    if safe_min >= safe_max:
        safe_max = safe_min + 1.0
        
    random_dist = random.uniform(safe_min, safe_max)
    return norm_vec * random_dist

# ---- Front view (0°) ----
base_front_loc = (0, -4.5, 0.3)
cam_obj.location = get_randomized_location(base_front_loc, active_mesh)
cam_obj.rotation_euler = (math.radians(88), 0, 0)
scene.render.filepath = output_front
print(f"Rendering Front to: {output_front}")
bpy.ops.render.render(write_still=True)

# ---- Back view (Randomized) ----
back_camera_patterns = [
    {"location": (0, 4.5, 0.3), "rotation": (math.radians(88), 0, math.radians(180)), "name": "Direct Back"},
    {"location": (3.18, 3.18, 0.3), "rotation": (math.radians(88), 0, math.radians(135)), "name": "Back Right"},
    {"location": (-3.18, 3.18, 0.3), "rotation": (math.radians(88), 0, math.radians(225)), "name": "Back Left"},
    {"location": (0, 3.8, 2.4), "rotation": (math.radians(60), 0, math.radians(180)), "name": "High Back"}
]

chosen_pattern = random.choice(back_camera_patterns)
cam_obj.location = get_randomized_location(chosen_pattern["location"], active_mesh)
cam_obj.rotation_euler = chosen_pattern["rotation"]
scene.render.filepath = output_back
print(f"Rendering Back ({chosen_pattern['name']}) to: {output_back}")
bpy.ops.render.render(write_still=True)

print("Render complete.")
