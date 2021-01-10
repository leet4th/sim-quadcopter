
import numpy as np
from math import radians, sin, cos, acos


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

