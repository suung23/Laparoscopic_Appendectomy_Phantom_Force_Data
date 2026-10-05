"""Paired envelope intervals with local one-to-one sheet correspondence.

Preserves multiple ray intervals. Local sampling does not prove global absence
of intersections, and ambiguous sheet merges remain open boundaries.
"""
import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from mesentery_surface import sample_envelope


def split_vertex_fans(points,faces):
    """Separate disconnected face fans that otherwise touch only at one vertex."""
    faces=faces.copy();parents=list(range(len(points)))
    for vertex in range(len(points)):
        incident=np.flatnonzero(np.any(faces==vertex,axis=1))
        if len(incident)<2:continue
        neighbours={}
        for f in incident:
            for other in faces[f]:
                if other!=vertex:neighbours.setdefault(int(other),[]).append(int(f))
        adjacency={int(f):set() for f in incident}
        for connected in neighbours.values():
            for f in connected:adjacency[f].update(connected)
        remaining=set(adjacency);components=[]
        while remaining:
            pending=[min(remaining)];component=set()
            while pending:
                f=pending.pop()
                if f in component:continue
                component.add(f);pending.extend(adjacency[f]-component)
            remaining-=component;components.append(sorted(component))
        for component in components[1:]:
            new=len(parents);parents.append(vertex)
            for f in component:faces[f,faces[f]==vertex]=new
    parents=np.asarray(parents,dtype=int)
    return points[parents],faces,parents


def extract_multilayer(mesh,spacing=.00075):
    center,axes,x,y,xx,yy,origins,groups=sample_envelope(mesh,spacing)
    points=[];rays=[];chords=[];odd=[];multi=[]
    for ray,depth in groups.items():
        if len(depth)%2:odd.append(ray);continue
        if len(depth)>2:multi.append(ray)
        for lo,hi in zip(depth[::2],depth[1::2]):
            if hi-lo<=1e-8:continue
            points.append(np.array([xx.ravel()[ray],yy.ravel()[ray],(lo+hi)/2])@axes+center)
            rays.append(ray);chords.append(hi-lo)
    points=np.asarray(points);rays=np.asarray(rays);chords=np.asarray(chords)
    if not len(points):raise ValueError('No paired intervals')
    inside=mesh.contains(points)
    grid={}
    for i in np.flatnonzero(inside):grid.setdefault(int(rays[i]),[]).append(int(i))
    triangles=[];edges=set()
    for row in range(len(y)-1):
        for col in range(len(x)-1):
            a=row*len(x)+col;b=a+1;c=a+len(x);d=c+1
            for tri in ((a,b,d),(a,d,c)):
                if not all(v in grid for v in tri):continue
                triangles.append(tri)
                edges.update(tuple(sorted(pair)) for pair in ((tri[0],tri[1]),(tri[0],tri[2]),(tri[1],tri[2])))
    # Batch interior tests; a candidate edge may not pass through empty space.
    pairs=[];edge_slices={}
    for edge in sorted(edges):
        start=len(pairs)
        pairs.extend((a,b) for a in grid[edge[0]] for b in grid[edge[1]])
        edge_slices[edge]=(start,len(pairs))
    pairs=np.asarray(pairs,dtype=int)
    if not len(pairs):raise ValueError('No neighbouring intervals')
    a,b=points[pairs[:,0]],points[pairs[:,1]]
    lengths=np.linalg.norm(a-b,axis=1)
    accepted=lengths<=4*spacing
    candidates=np.flatnonzero(accepted)
    sample=a[candidates,None,:]+np.array([.25,.5,.75])[None,:,None]*(b-a)[candidates,None,:]
    accepted[candidates]=mesh.contains(sample.reshape(-1,3)).reshape(-1,3).all(axis=1)
    matches={}
    for edge,(start,end) in edge_slices.items():
        left,right=grid[edge[0]],grid[edge[1]]
        cost=lengths[start:end].reshape(len(left),len(right)).copy()
        mask=accepted[start:end].reshape(cost.shape);cost[~mask]=1e6
        i,j=linear_sum_assignment(cost)
        mapping={left[u]:right[v] for u,v in zip(i,j) if mask[u,v]}
        matches[edge]=mapping;matches[edge[::-1]]={v:k for k,v in mapping.items()}
    faces=[]
    for a,b,c in triangles:
        ab,ac,bc=matches[(a,b)],matches[(a,c)],matches[(b,c)]
        for i,j in ab.items():
            k=ac.get(i)
            if k is not None and bc.get(j)==k:faces.append((i,j,k))
    if not faces:raise ValueError('No mutually consistent sheet triangles')
    faces=np.asarray(faces,dtype=int)
    face_inside=mesh.contains(points[faces].mean(axis=1))
    faces=faces[face_inside]
    if not len(faces):raise ValueError('No interior triangles')
    used=np.unique(faces);remap=np.full(len(points),-1);remap[used]=np.arange(len(used))
    split_points,split_faces,parent=split_vertex_fans(points[used],remap[faces])
    original_used_count=len(used);used=used[parent]
    surface=trimesh.Trimesh(split_points,split_faces,process=False)
    edge_counts=np.bincount(surface.edges_unique_inverse)
    if np.any(edge_counts>2):raise ValueError('Sheet correspondence created nonmanifold edges')
    arrays=dict(points=surface.vertices,faces=surface.faces,source_grid_indices=rays[used],
        source_axis_chord_m=chords[used],multiple_interval_rays=np.asarray(multi),
        rejected_odd_intersection_rays=np.asarray(odd),rejected_outside_midpoint_rays=rays[~inside])
    report=dict(method='multilayer paired intervals with interior edge samples and one-to-one correspondence',
        grid_shape=[len(y),len(x)],grid_spacing_m=spacing,rays=len(origins),intersected_rays=len(groups),
        multiple_interval_rays=len(multi),represented_multiple_interval_rays=int(np.isin(multi,np.unique(rays[used])).sum()),
        generated_midpoints=len(points),rejected_outside_midpoints=int((~inside).sum()),
        rejected_edge_candidates=int((~accepted).sum()),rejected_triangle_centroids=int((~face_inside).sum()),
        max_candidate_edge_length_m=4*spacing,edge_interior_sample_fractions=[.25,.5,.75],
        vertices=len(surface.vertices),triangles=len(surface.faces),boundary_edges=int((edge_counts==1).sum()),
        split_disconnected_vertex_fans=len(used)-original_used_count,
        connected_surface_components=len(surface.split(only_watertight=False)),nonmanifold_edges=0,
        frame_axes=axes.tolist(),frame_center_m=center.tolist(),
        unresolved='Finite interior samples do not prove continuous containment; sheet merges can remain disconnected')
    return surface,arrays,report
