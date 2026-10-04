"""Closest point on tetrahedral exterior triangles, not nearest FEM node."""
import numpy as np
import trimesh


class TissueSurface:
    def __init__(self,tetrahedra):
        t=np.asarray(tetrahedra,int).reshape(-1,4)
        faces=np.concatenate([t[:,f] for f in ((1,2,3),(0,3,2),(0,1,3),(0,2,1))])
        opposite=np.concatenate([t[:,i] for i in range(4)])
        _,inverse,count=np.unique(np.sort(faces,axis=1),axis=0,return_inverse=True,return_counts=True)
        if np.any(count>2):raise ValueError('Non-manifold tetrahedral boundary')
        mask=count[inverse]==1
        self.faces=faces[mask];self.opposite=opposite[mask]
        if not len(self.faces):raise ValueError('No exterior tissue triangles')

    def closest(self,points,tips):
        points=np.asarray(points,float);tips=np.asarray(tips,float).reshape(-1,3)
        if not np.isfinite(points).all() or not np.isfinite(tips).all():raise ValueError('Nonfinite geometry')
        triangles=points[self.faces]
        scale=float(np.ptp(points,axis=0).max());origin=points.mean(0)
        if scale<=0:raise ValueError('Zero-size tissue geometry')
        normalized=(triangles-origin)/scale
        rows=[]
        for tip in tips:
            closest=trimesh.triangles.closest_point(normalized,np.broadcast_to((tip-origin)/scale,(len(triangles),3)))*scale+origin
            distances=np.linalg.norm(closest-tip,axis=1);i=int(np.argmin(distances))
            tri=triangles[i];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0])
            length=np.linalg.norm(normal)
            if length<1e-14:raise ValueError('Degenerate nearest surface triangle')
            normal/=length
            if normal@(points[self.opposite[i]]-tri[0])>0:normal=-normal
            rows.append(dict(point_m=closest[i].tolist(),distance_m=float(distances[i]),
                outward_normal=normal.tolist(),triangle_nodes=self.faces[i].tolist(),
                local_outward_offset_m=float((tip-closest[i])@normal)))
        return rows
