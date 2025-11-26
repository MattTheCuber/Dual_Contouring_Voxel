# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 08:21:14 2019

@author: WilmesD
"""
from math import sqrt,sin,cos,pow,exp,asin,atan2,pi
from direct.showbase.ShowBase import ShowBase
import panda3d.core as pd
import numpy as np
from Voxel_Functions import make_face,make_mesh,StitchLoop

dims = 16
xmin = -dims
xmax = dims
ymin = -dims
ymax = dims
zmin = -dims
zmax = dims

def ball(x,y,z):
    a = 15 - sqrt(pow(x,2)+pow(y,2)+pow(z,2))
    #b = 4 - sqrt(pow(x-9,2)+pow(y,2)+pow(z,2))
    return a

def helix(x,y,z):
    if abs((z+5)-5*sin((pi/10)*x)) <= 2 and abs((y+5)-5*cos((pi/10)*x)) <= 2:# and abs(y) < 15 and abs(z) < 14 and abs(x) < 15:
        return 1
    else:
        return box(x,y,z)

def terrain(x,y,z):
#    if z < 0:
    if ((z+8) - (5*sin(x*pi/120)+2*cos(y*pi/11)-3*sin(x*y/90)-1.5*cos(x*2*pi/16))) < 1:# and abs(y+8) < 32 and abs(x+8) < 32:
#    if (z+8) - 5*sin(y*pi/20) < 0:
        return 1
    else:
        return helix(x,y,z)
    
def parabel(x,y,z):
    if (z - x*x) < 1 and abs(y) < 10 and abs(x) < 5 and abs(z) < 25:
        return 1
    else:
        return -1

def line(x,y,z):
    if abs(y) < 1:
        if abs(z-x) < 2:
            return 1
        else:
            return -1
    else:
        return -1
    
def box(x,y,z):
    if abs(z) < 15 and abs(y) < 15 and abs(x) < 15:
        if (19 - sqrt(pow(x,2)+pow(y,2)+pow(z,2))) > 0:
            return -1
        else:
            return 1
    else:
        return -10 #-10 gives sharper edges
    
def box2(x,y,z):
    if abs(z) <= 14.5 and abs(y) <= 14.5 and abs(x) <= 14.5:
        return 1
    else:
        return ball(x,y,z)#(10 - sqrt(pow(x-8,2)+pow(y-2,2)+pow(z-2,2)))

def makeGradients(f,x,y,z):
    c = 0.01
    return ((f(x+c,y,z)-f(x-c,y,z))/(2*c),
            (f(x,y+c,z)-f(x,y-c,z))/(2*c),
            (f(x,y,z+c)-f(x,y,z-c))/(2*c))

def interpolate_sign_change(v1,v2):
    return (0-v1)/(v2-v1)

def solve_lstsqrs_cost_function(positions,normals):
    A = np.array(normals)
    b = [v[0] * n[0] + v[1] * n[1] + v[2] * n[2] for v, n in zip(positions, normals)]
    #print(b)
    s,_,_,_ = np.linalg.lstsq(A,b,0.01)
    return s
    
def clipSolution(s,x,y,z,stepsize):
    d = stepsize
    if (s[0] - x) > d:
        s[0] = x + d
    elif (s[0] - x) < -d/2:
        s[0] = x
    if (s[1] - y) > d:
        s[1] = y + d
    elif (s[1] - y) < -d/2:
        s[1] = y
    if (s[2] - z) > d:
        s[2] = z + d
    elif (s[2] - z) < -d/2:
        s[2] = z
    return s

def find_vertex_in_voxel(f,x,y,z,stepsize,cx,cy,cz):
    #Look for sign change
    v = np.empty((stepsize+1,stepsize+1,stepsize+1), dtype=float)
    for dx in (0,stepsize):
        for dy in (0,stepsize):
            for dz in (0,stepsize):
                v[dx,dy,dz] = f(x+dx,y+dy,z+dz)
                
    #save exact points of sign changes in list
    change = []
    if cz == True:
        for dx in (0, stepsize):
            for dy in (0, stepsize):
                if (v[dx, dy, 0] > 0) != (v[dx, dy, stepsize] > 0):
                    change.append((x+dx,y+dy,z+interpolate_sign_change(v[dx, dy, 0],v[dx, dy, stepsize])))
        
    if cy == True:        
        for dx in (0, stepsize):
            for dz in (0, stepsize):
                if (v[dx, 0, dz] > 0) != (v[dx, stepsize, dz] > 0):
                    change.append((x+dx,y+interpolate_sign_change(v[dx, 0, dz],v[dx, stepsize, dz]),z+dz))
        
    if cx == True:        
        for dy in (0, stepsize):
            for dz in (0, stepsize):
                if (v[0, dy, dz] > 0) != (v[stepsize, dy, dz] > 0):
                    change.append((x+interpolate_sign_change(v[0, dy, dz],v[stepsize, dy, dz]),y+dy,z+dz))
                
    if len(change) <= 1:
        return None
    else:
        return (x,y,z)
    #print(len(change))
    #get normals from gradients (actually gradients == -normals, because the generated surface is on the sign change point, so the normal of the surface is the direction the value decreases a.k.a gradients)
    normals = []
    for v in change:
        n = makeGradients(f,v[0],v[1],v[2])
        normals.append(n)
        
    #add bias to cost function to reduce likelyhood that the resulting point lies outside the voxel-boundary
    mean = np.mean(change,axis=0)
    change.append(mean)
    normals.append([0.1,0,0])
    change.append(mean)
    normals.append([0,0.1,0])
    change.append(mean)
    normals.append([0,0,0.1])
        
    #Solve Least Squares cost function 
    #sum( dot(p-v[i], n[i])^2 ) with p = point, v[i] = sign-change-positions, n[i] = normals
    #Basically "Find the point that is as orthogonal to the normals as possible and minimizes the distance to the sign-change-points"
    s = solve_lstsqrs_cost_function(change, normals)
    s = clipSolution(s,x,y,z,stepsize)
    #print(abs(s-(x,y,z)))
    return s
    

def dual_contouring(f,xmin,xmax,ymin,ymax,zmin,zmax,vdata,stepsize):
    vert_array = {}
    vert_indices = {}
    edge_vertices = {}
    vdata.setNumRows(1000)
    vertexd = pd.GeomVertexWriter(vdata, "vertex")
    colord = pd.GeomVertexWriter(vdata, "color")
    texcoordd = pd.GeomVertexWriter(vdata, "texcoord")
    for x in range(xmin,xmax,stepsize):
        for y in range(ymin,ymax,stepsize):
            for z in range(zmin,zmax,stepsize):
                vert = find_vertex_in_voxel(f,x,y,z,stepsize,True,True,True)  
                coord = (x,y,z)                  
                if vert is None:
                    continue
                
                vertexd.add_data3f(vert[0],vert[1],vert[2])
                colord.add_data4f(abs(z)/5,abs(z)/5,abs(z)/5,1)
                if vertexd.getWriteRow() % 4 == 0:
#                    colord.add_data4f(1,1,1,1)
                    texcoordd.add_data2f(0,0)
                elif vertexd.getWriteRow() % 3 == 0:
#                    colord.add_data4f(0,0,1,1)
                    texcoordd.add_data2f(0,1)
                elif vertexd.getWriteRow() % 2 == 0:
#                    colord.add_data4f(0,1,0,1)
                    texcoordd.add_data2f(1,1)
                else:
#                    colord.add_data4f(1,0,0,1)
                    texcoordd.add_data2f(1,0)
                if x == xmax or y == ymax or z == zmax or x == xmax-1 or y == ymax-1 or z == zmax-1 or x == xmin or y == ymin or z == zmin:
                    edge_vertices[x,y,z] = vert
                vert_indices[coord] = len(vert_array)
                vert_array[coord] = vert
                
    faces = []
    for x in range(xmin,xmax,stepsize):
        for y in range(ymin,ymax,stepsize):
            for z in range(zmin,zmax,stepsize):
#                if x == xmin 
                
                if x > xmin and y > ymin:
                    inside1 = f(x,y,z) > 0
                    inside2 = f(x,y,z+stepsize) > 0
                    if inside1 != inside2:
                        faces.append(make_face(vert_indices[(x-stepsize,y-stepsize,z)],
                                               vert_indices[(x-0,y-stepsize,z)],
                                               vert_indices[(x-0,y-0,z)],
                                               vert_indices[(x-stepsize,y-0,z)],inside2,vdata))
                if x > xmin and z > zmin:
                    inside1 = f(x,y,z) > 0
                    inside2 = f(x,y+stepsize,z) > 0
                    if inside1 != inside2:
                        faces.append(make_face(vert_indices[(x-stepsize,y,z-stepsize)],
                                               vert_indices[(x-0,y,z-stepsize)],
                                               vert_indices[(x-0,y,z-0)],
                                               vert_indices[(x-stepsize,y,z-0)],inside1,vdata))
                if y > ymin and z > zmin:
                    inside1 = f(x,y,z) > 0
                    inside2 = f(x+stepsize,y,z) > 0
                    if inside1 != inside2:
                        faces.append(make_face(vert_indices[(x,y-stepsize,z-stepsize)],
                                               vert_indices[(x,y-0,z-stepsize)],
                                               vert_indices[(x,y-0,z-0)],
                                               vert_indices[(x,y-stepsize,z-0)],inside2,vdata))
                        
    return faces,edge_vertices.copy()  
    
def stitch(f,vertices,vdata,difx,dify,difz,dist,stepsize):
    faces = []
    all_vertices = {}
    vdata.setNumRows(2000)
#    print(list(vertices.keys()))
    for x in range((-dist+1)*difx,dist*difx,difx):
        for y in range((-dist+1)*dify,dist*dify,dify):
            for z in range((-dist+1)*difz,dist*difz,difz):
                all_vertices.update(vertices[x,y,z])
                
    faces = StitchLoop(f,all_vertices,vdata,difx,dify,difz,dist,stepsize)
    
    return faces

def chunking(f,xmin,xmax,ymin,ymax,zmin,zmax,dist):
    difx = dims
    dify = dims
    difz = dims
    dist = int(dist)
    gformat = pd.GeomVertexFormat.getV3cpt2()
    vdata = {}
    vertices = {}
#    steps = {}
    for x in range((-dist+1)*difx,dist*difx,difx):
        for y in range((-dist+1)*dify,dist*dify,dify):
            for z in range((-dist+1)*difz,dist*difz,difz):
                a = max(max(abs(x),abs(y)),abs(z))
#                print("x:"+str(x) + " " + "y:"+str(y)+ " " + "z:"+str(z))
#                print("Max coord : " + str(a))
                stepsize = 2**((a/difx)-1)
                if a < 1:
                    stepsize = 1.0
#                steps[(x,y,z)] = stepsize
                print("Voxelsize : " + str(stepsize))
                vdata[(x,y,z)] = pd.GeomVertexData('Triangle', gformat, pd.Geom.UHStatic)
                cvdata = vdata[(x,y,z)]
#                print(cvdata.getArray(0))
                faces,vertices[x,y,z] = dual_contouring(f,x,int(x+difx),y,int(y+dify),z,int(z+difz),cvdata,int(stepsize))
                node = make_mesh(cvdata,faces)
                nodepath = render.attachNewNode(node)
                nodepath.setRenderModeFilledWireframe((1,1,1,1),1)
                
    vdata[-1] = pd.GeomVertexData('Triangle', gformat, pd.Geom.UHStatic)
    stitches = stitch(f,vertices.copy(),vdata[-1],difx,dify,difz,dist,stepsize)
    node = make_mesh(vdata[-1],stitches)
    nodepath = render.attachNewNode(node)
    nodepath.setRenderModeFilledWireframe((1,1,1,1),1)#
    return nodepath

def destroyWindow(get):
    get.destroy()

class myapp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        #Enter your function as the first argument here
        chunking(terrain,xmin,xmax,ymin,ymax,zmin,zmax,2)
        par = []
        par.append(self)
        self.accept("q",destroyWindow,par)


if __name__ == "__main__":
    app = myapp()
    app.run()




