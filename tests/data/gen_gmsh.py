"""Generate the Gmsh fixtures in this directory with real Gmsh (not run by the tests).

Run outside the project env, with a Python that has the `gmsh` module (4.15.2 was used):
    python3 tests/data/gen_gmsh.py
Model: two unit boxes side by side (x in [0, 2]), fragmented, transfinite 2 x 2 x 2 hex8 each.
Volume 1 is x < 1, volume 2 is x > 1.
"""

from pathlib import Path

import gmsh

OUT = Path(__file__).parent


def build(name, groups, version=4.1, saveall=False, binary=False, order=1, partition=0, inp=False):
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add(name)
    a = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    b = gmsh.model.occ.addBox(1, 0, 0, 1, 1, 1)
    gmsh.model.occ.fragment([(3, a)], [(3, b)])
    gmsh.model.occ.synchronize()
    for _, c in gmsh.model.getEntities(1):
        gmsh.model.mesh.setTransfiniteCurve(c, 3)
    for _, s in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(s)
        gmsh.model.mesh.setRecombine(2, s)
    for _, v in gmsh.model.getEntities(3):
        gmsh.model.mesh.setTransfiniteVolume(v)

    def box(x0, x1, dim, y1=1.1):
        return [e for _, e in gmsh.model.getEntitiesInBoundingBox(x0, -0.1, -0.1, x1, y1, 1.1, dim)]

    x0, x2, edge = box(-0.1, 0.1, 2), box(1.9, 2.1, 2), box(-0.1, 0.1, 1, y1=0.1)
    for dim, ents, tag, label in groups(x0, x2, edge):
        gmsh.model.addPhysicalGroup(dim, ents, tag=tag, name=label)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)  # hex20, not hex27, at order 2
    gmsh.model.mesh.generate(3)
    if order > 1:
        gmsh.model.mesh.setOrder(order)
    if partition:
        gmsh.model.mesh.partition(partition)
    gmsh.option.setNumber("Mesh.MshFileVersion", version)
    gmsh.option.setNumber("Mesh.SaveAll", int(saveall))
    gmsh.option.setNumber("Mesh.Binary", int(binary))
    gmsh.write(str(OUT / f"{name}.msh"))
    if inp:
        gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)
        path = OUT / f"{name}.inp"
        gmsh.write(str(path))
        path.write_text(path.read_text().replace(str(OUT) + "/", ""))  # heading holds the path
    gmsh.finalize()


def messy(x0, x2, edge):
    return [
        (3, [1], 7, "left"),
        (3, [2], 8, ""),
        (2, x0, 3, "clamp"),
        (2, x0 + x2, 4, ""),  # x0 in a second, unnamed group
        (1, edge, 3, "edge"),  # same tag as clamp, other dimension
    ]


def solid(x0, x2, edge):
    return [(3, [1, 2], 5, "solid"), (2, x0, 3, "clamp"), (1, edge, 4, "edge")]


build("gmsh_messy", messy)
build("gmsh_pair", solid, order=2, inp=True)
build("gmsh_nogroups", lambda *_: [])
build("gmsh_vol_two_groups", lambda *_: [(3, [1, 2], 7, ""), (3, [1], 8, "")])
build("gmsh_untagged_volume", lambda x0, *_: [(3, [1], 7, ""), (2, x0, 3, "clamp")])
build("gmsh_saveall", lambda x0, *_: [(3, [1], 7, ""), (2, x0, 3, "clamp")], saveall=True)
build("gmsh_surfaces_only", lambda x0, *_: [(2, x0, 3, "clamp")])
build(
    "gmsh_dup_name",
    lambda x0, x2, edge: [(3, [1, 2], 7, ""), (2, x0, 3, "clamp"), (1, edge, 3, "clamp")],
)
build(
    "gmsh_fallback_clash",
    lambda x0, x2, edge: [(3, [1, 2], 7, ""), (2, x0, 5, "physical_2_4"), (2, x2, 4, "")],
)
build("gmsh_v22", messy, version=2.2)
build("gmsh_v40", messy, version=4.0)
build("gmsh_binary", messy, binary=True)
build("gmsh_partitioned", messy, partition=2)
