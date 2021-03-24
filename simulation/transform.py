#!/usr/bin/env python3

import numpy as np
from math import radians, sin, cos, acos
import pyproj


def expq(n):
    n *= 0.5
    nMag = np.sqrt( n.dot(n) )
    if nMag > 0.0:
        q = np.hstack( (cos(nMag), n/nMag*sin(nMag)) )
    else:
        q = np.array([1.0,0,0,0])
    return q

def logq(q):
    qw = q[0]
    qv = q[1:]
    qvMag = np.sqrt( qv.dot(qv) )
    if qvMag > 0.0:
        n = acos(qw)/qvMag * qv
    else:
        # qv must be [0,0,0] if qvMag == 0
        n = qv
    return n

def axisAng2quat(k,ang):
    return np.hstack(( cos(ang/2), k*sin(ang/2) ))

def qConj(q):
    q_out = np.array(q, copy=True)
    q_out[1:] = -q[1:]
    return q_out


def skew3(v):
    vx,vy,vz = v
    out = np.array([[  0, -vz,   vy],
                    [ vz,   0,  -vx],
                    [-vy,  vx,    0]])
    return out

def skew4L(v):
    if len(v)==3:
        v = np.hstack((0,v))
    w,x,y,z = v
    out = np.array([
        [w, -x, -y, -z],
        [x,  w, -z,  y],
        [y,  z,  w, -x],
        [z, -y,  x,  w],
    ])
    return out

def skew4R(v):
    if len(v)==3:
        v = np.hstack((0,v))
    w,x,y,z = v
    out = np.array([
        [w, -x, -y, -z],
        [x,  w,  z, -y],
        [y, -z,  w,  x],
        [z,  y, -x,  w],
    ])
    return out


def qRot(q,v):
    qPrime = qConj(q)
    v = np.hstack((0,v))
    vout = skew4L(q).dot(skew4R(qPrime)).dot(v)
    return vout[1:]

def dRotdq(q,v):
    qw,qx,qy,qz = q
    vx,vy,vz = v

    dRdq = np.array([
        [2*qw*vx + 2*qy*vz - 2*qz*vy,  2*qx*vx + 2*qy*vy + 2*qz*vz,  2*qw*vz + 2*qx*vy - 2*qy*vx, -2*qw*vy + 2*qx*vz - 2*qz*vx],
        [2*qw*vy - 2*qx*vz + 2*qz*vx, -2*qw*vz - 2*qx*vy + 2*qy*vx,  2*qx*vx + 2*qy*vy + 2*qz*vz,  2*qw*vx + 2*qy*vz - 2*qz*vy],
        [2*qw*vz + 2*qx*vy - 2*qy*vx,  2*qw*vy - 2*qx*vz + 2*qz*vx, -2*qw*vx - 2*qy*vz + 2*qz*vy,  2*qx*vx + 2*qy*vy + 2*qz*vz]
    ])

    return dRdq

def dVdq(q,v):
    qw,qx,qy,qz = q
    vx,vy,vz = v

    dv = np.array([
        [ 2*vx*qw + 2*vy*qz - 2*vz*qy, 2*vx*qx + 2*vy*qy + 2*vz*qz, -2*vx*qy + 2*vy*qx - 2*vz*qw, -2*vx*qz + 2*vy*qw + 2*vz*qx],
        [-2*vx*qz + 2*vy*qw + 2*vz*qx, 2*vx*qy - 2*vy*qx + 2*vz*qw,  2*vx*qx + 2*vy*qy + 2*vz*qz, -2*vx*qw - 2*vy*qz + 2*vz*qy],
        [ 2*vx*qy - 2*vy*qx + 2*vz*qw, 2*vx*qz - 2*vy*qw - 2*vz*qx,  2*vx*qw + 2*vy*qz - 2*vz*qy,  2*vx*qx + 2*vy*qy + 2*vz*qz],
    ])

    return dv



def quat2dcm(q):
    """
    Convert quaternion to DCM
    """

    # Extract components
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    # Reduce repeated calculations
    ww = w*w
    xx = x*x
    yy = y*y
    zz = z*z
    wx = w*x
    wy = w*y
    wz = w*z
    xy = x*y
    xz = x*z
    yz = y*z

    # Build Direction Cosine Matrix (DCM)
    dcm = np.array([
        [ww + xx - yy - zz,       2*(xy - wz),       2*(xz + wy)],
        [      2*(xy + wz), ww - xx + yy - zz,       2*(yz - wx)],
        [      2*(xz - wy),       2*(yz + wx), ww - xx - yy + zz]
    ])
    return dcm




def quat2euler321(q):

    # Extract components
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    # Reduce repeated calculations
    ww = w*w
    xx = x*x
    yy = y*y
    zz = z*z
    wx = w*x
    wy = w*y
    wz = w*z
    xy = x*y
    xz = x*z
    yz = y*z

    # Calculate angles
    #yaw = np.arctan2( 2*(xy-wz), 2*(ww+xx) - 1.0)
    #pitch = -np.arcsin(2*(xz - wy));
    #roll = np.arctan2(2*(yz-wx), 2*(ww+zz) - 1.0)

    yaw   = np.arctan2( 2*(xy + wz), ww + xx - yy - zz );
    pitch = -np.arcsin(2*(xz - wy));
    roll  = np.arctan2(2*(yz + wx) , ww - xx - yy + zz);
    return np.array([yaw, pitch, roll])

def euler3212quat(ang):
    """
    Converts the euler321 sequence to quaternion attitude representation

    euler321 refers to three sequential rotations about the axes specified by the number

    euler321 -> RotZ(ang1) -> RotY(ang2) -> RotX(ang3)

    This maps to the typical aerospace convention of
    euler321 -> RotZ(yaw) -> RotY(pitch) -> RotX(roll)

    """

    c1 = np.cos(ang[0]/2)
    s1 = np.sin(ang[0]/2)
    c2 = np.cos(ang[1]/2)
    s2 = np.sin(ang[1]/2)
    c3 = np.cos(ang[2]/2)
    s3 = np.sin(ang[2]/2)

    w = c1*c2*c3+s1*s2*s3
    x = c1*c2*s3-s1*s2*c3
    y = c1*s2*c3+s1*c2*s3
    z = s1*c2*c3-c1*s2*s3

    # Ensure unit quaternion
    q = np.array([w,x,y,z])
    q = q / np.sqrt(q.dot(q))

    return q

def dcm2quat(dcm):
    m = dcm.conj().transpose() # This method assumes row-vector and postmultiplication of that vector
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = [m[1, 2]-m[2, 1],  t,  m[0, 1]+m[1, 0],  m[2, 0]+m[0, 2]]
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = [m[2, 0]-m[0, 2],  m[0, 1]+m[1, 0],  t,  m[1, 2]+m[2, 1]]
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = [m[0, 1]-m[1, 0],  m[2, 0]+m[0, 2],  m[1, 2]+m[2, 1],  t]
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = [t,  m[1, 2]-m[2, 1],  m[2, 0]-m[0, 2],  m[0, 1]-m[1, 0]]

    q = np.array(q) #.astype('float64')
    q *= 0.5 / np.sqrt(t)
    q = q / np.sqrt(q.dot(q))
    return q


def dcm2quat_old(dcm):
    """
    Determine quaternion corresponding to dcm using
    the stanley method.

    Flips sign to always return shortest path quaterion
    so w >= 0

    Converts the 3x3 DCM into the quaterion where the
    first component is the real part
    """

    tr = np.trace(dcm)

    w = 0.25*(1+tr)
    x = 0.25*(1+2*dcm[0,0]-tr)
    y = 0.25*(1+2*dcm[1,1]-tr)
    z = 0.25*(1+2*dcm[2,2]-tr)

    kMax = np.argmax([w,x,y,z])
    print(kMax)

    if kMax == 0:
        w = np.sqrt(w)
        x = 0.25*(dcm[1,2]-dcm[2,1])/w
        y = 0.25*(dcm[2,0]-dcm[0,2])/w
        z = 0.25*(dcm[0,1]-dcm[1,0])/w

    elif kMax == 1:
        x = np.sqrt(x)
        w = 0.25*(dcm[1,2]-dcm[2,1])/x
        if w<0:
            x = -x
            w = -w
        y = 0.25*(dcm[0,1]+dcm[1,0])/x
        z = 0.25*(dcm[2,0]+dcm[0,2])/x

    elif kMax == 2:
        y = np.sqrt(y)
        w = 0.25*(dcm[2,0]-dcm[0,2])/y
        if w<0:
            y = -y
            w = -w
        x = 0.25*(dcm[0,1]+dcm[1,0])/y
        z = 0.25*(dcm[1,2]+dcm[2,1])/y

    elif kMax == 3:
        z = np.sqrt(z)
        w = 0.25*(dcm[0,1]-dcm[1,0])/z
        if w<0:
            z = -z
            w = -w
        x = 0.25*(dcm[2,0]+dcm[0,2])/z
        y = 0.25*(dcm[1,2]+dcm[2,1])/z

    # Ensure unit quaternion
    q = np.array([w,x,y,z])
    q = q / np.sqrt( q.dot(q) )

    return q


def ecef2lla(pos):
    # perform ecef -> lat,lon,alt transform
    transformer = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        )
    lon,lat,alt = transformer.transform(pos[0],pos[1],pos[2],
                                        radians=False)
    return np.array([lat,lon,alt])

def lla2ecef(lla):
    # perform lat,lon,alt -> ecef transform
    transformer = pyproj.Transformer.from_crs(
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        )
    # lla (lat, lon, alt)
    # transform() expects lon, lat, alt
    x,y,z = transformer.transform(lla[1],lla[0],lla[2],
                                  radians=False)
    return np.array([x,y,z])

