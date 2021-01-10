import numpy as np
import pygame
from operator import itemgetter

import time

RAD2DEG = 180.0/np.pi
DEG2RAD = 1.0/RAD2DEG

# Node stores each point of the block
class Node:
    def __init__(self, coordinates, color):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.z = coordinates[2]
        self.color = color

# Face stores 4 nodes that make up a face of the block
class Face:
    def __init__(self, nodes, color):
        self.nodeIndexes = nodes
        self.color = color

class OpenLoopAttitude():
    def __init__(self):
        self.dcm = np.eye(3)

    def setEuler321(self,ang):
        """
        Converts euler321 to dcm
        euler321 -> RotZ(ang1) -> RotY(ang2) -> RotX(ang3)
        """

        c1 = np.cos(ang[0])
        s1 = np.sin(ang[0])
        c2 = np.cos(ang[1])
        s2 = np.sin(ang[1])
        c3 = np.cos(ang[2])
        s3 = np.sin(ang[2])

        # Build Direction Cosine Matrix (DCM)
        dcm = np.zeros((3,3))
        dcm[0,0] = c2*c1
        dcm[0,1] = c2*s1
        dcm[0,2] = -s2
        dcm[1,0] = s3*s2*c1 - c3*s1
        dcm[1,1] = s3*s2*s1 + c3*c1
        dcm[1,2] = s3*c2
        dcm[2,0] = c3*s2*c1 + s3*s1
        dcm[2,1] = c3*s2*s1 - s3*c1
        dcm[2,2] = c3*c2

        self.dcm = dcm

    def setQuat(self, q):
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
        dcm = np.zeros((3,3))
        dcm[0,0] = ww + xx - yy - zz
        dcm[0,1] = 2*(xy + wz)
        dcm[0,2] = 2*(xz - wy)
        dcm[1,0] = 2*(xy - wz)
        dcm[1,1] = ww - xx + yy - zz
        dcm[1,2] = 2*(yz + wx)
        dcm[2,0] = 2*(xz + wy)
        dcm[2,1] = 2*(yz - wx)
        dcm[2,2] = ww - xx - yy + zz

        self.dcm = dcm

    def getDcm(self):
        return self.dcm

    def getEuler321(self):
        # euler321 -> RotZ(ang1) -> RotY(ang2) -> RotX(ang3)
        yaw = np.arctan2(self.dcm[0,1], self.dcm[0,0])
        pitch = -np.arcsin( self.dcm[0,2] )
        roll = np.arctan2( self.dcm[1,2], self.dcm[2,2] )

        return np.array([yaw, pitch, roll])

    def getQuaternion(self):
        """
        Determine quaternion corresponding to dcm using
        the stanley method.

        Flips sign to always return shortest path quaterion
        so w >= 0

        Converts the 3x3 DCM into the quaterion where the
        first component is the real part
        """

        dcm = self.dcm
        tr = np.trace(dcm)

        w = 0.25*(1+tr)
        x = 0.25*(1+2*dcm[0,0]-tr)
        y = 0.25*(1+2*dcm[1,1]-tr)
        z = 0.25*(1+2*dcm[2,2]-tr)

        kMax = np.argmax([w,x,y,z])

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
        mag = np.sum(q*q)
        return q / mag


# Wireframe stores the details of black
class Wireframe:
    def __init__(self,sys):
        self.nodes = []
        self.edges = []
        self.faces = []
        self.sys   = sys

    def addNodes(self, nodeList, colorList):
        for node, color in zip(nodeList, colorList):
            self.nodes.append(Node(node, color))

    def addFaces(self, faceList, colorList):
        for indexes, color in zip(faceList, colorList):
            self.faces.append(Face(indexes, color))

    def rotatePoint(self, point):
        #rotationMat = km.getRotMat(self.sys.xHat[0:4])
        dcm = self.sys.getDcm()
        return np.matmul(dcm, point)

    def convertToComputerFrame(self, point):
        computerFrameChangeMatrix = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
        return np.matmul(computerFrameChangeMatrix, point)

    def outputNodes(self):
        print("\n --- Nodes --- ")
        for i, node in enumerate(self.nodes):
            print(" %d: (%.2f, %.2f, %.2f) \t Color: (%d, %d, %d)" %
                 (i, node.x, node.y, node.z, node.color[0], node.color[1], node.color[2]))

    def outputFaces(self):
        print("\n --- Faces --- ")
        for i, face in enumerate(self.faces):
            print("Face %d:" % i)
            print("Color: (%d, %d, %d)" % (face.color[0], face.color[1], face.color[2]))
            for nodeIndex in face.nodeIndexes:
                print("\tNode %d" % nodeIndex)

class ProjectionViewer:
    """ Displays 3D objects on a Pygame screen """
    def __init__(self, width, height, wireframe):
        self.width = width
        self.height = height
        self.wireframe = wireframe
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Attitude Determination using Quaternions')
        self.background = (10,10,50) # Dark Blue
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.font = pygame.font.SysFont('Comic Sans MS', 30)

    def display(self):
        """ Draw the wireframes on the screen. """
        self.screen.fill(self.background)

        # Current Euler321
        yaw, pitch, roll = self.wireframe.sys.getEuler321()*RAD2DEG
        self.messageDisplay("Yaw:   %.1f" % yaw,
                            self.screen.get_width()*0.75,
                            self.screen.get_height()*0.05,
                            (220, 20, 60))      # Crimson
        self.messageDisplay("Pitch: %.1f" % pitch,
                            self.screen.get_width()*0.75,
                            self.screen.get_height()*0.1,
                            (0, 255, 255))     # Cyan
        self.messageDisplay("Roll:   %.1f" % roll,
                            self.screen.get_width()*0.75,
                            self.screen.get_height()*0.15,
                            (65, 105, 225))    # Royal Blue

        # Current Quaternion
        qw,qx,qy,qz = self.wireframe.sys.getQuaternion()
        self.messageDisplay("qw: %.4f" % qw,
                            self.screen.get_width()*0.05,
                            self.screen.get_height()*0,
                            (0, 128, 128))      # Teal
        self.messageDisplay("qx: %.4f" % qx,
                            self.screen.get_width()*0.05,
                            self.screen.get_height()*0.05,
                            (220, 20, 60))      # Crimson
        self.messageDisplay("qy: %.4f" % qy,
                            self.screen.get_width()*0.05,
                            self.screen.get_height()*0.1,
                            (0, 255, 255))     # Cyan
        self.messageDisplay("qz: %.4f" % qz,
                            self.screen.get_width()*0.05,
                            self.screen.get_height()*0.15,
                            (65, 105, 225))    # Royal Blue

        # Transform nodes to perspective view
        pvNodes = []
        pvDepth = []
        for node in self.wireframe.nodes:
            point = [node.x, node.y, node.z]
            newCoord = self.wireframe.rotatePoint(point)
            comFrameCoord = self.wireframe.convertToComputerFrame(newCoord)
            pvNodes.append(self.projectOthorgraphic(comFrameCoord[0], comFrameCoord[1], comFrameCoord[2],
                                                    self.screen.get_width(), self.screen.get_height(),
                                                    70, pvDepth))

        # Calculate the average Z values of each face.
        avg_z = []
        for face in self.wireframe.faces:
            n = pvDepth
            z = (n[face.nodeIndexes[0]] + n[face.nodeIndexes[1]] +
                 n[face.nodeIndexes[2]] + n[face.nodeIndexes[3]]) / 4.0
            avg_z.append(z)
        # Draw the faces using the Painter's algorithm:
        for idx, val in sorted(enumerate(avg_z), key=itemgetter(1)):
            face = self.wireframe.faces[idx]
            pointList = [pvNodes[face.nodeIndexes[0]],
                         pvNodes[face.nodeIndexes[1]],
                         pvNodes[face.nodeIndexes[2]],
                         pvNodes[face.nodeIndexes[3]]]
            pygame.draw.polygon(self.screen, face.color, pointList)

    def projectOthorgraphic(self, x, y, z, win_width, win_height, scaling_constant, pvDepth):
        # Normal Projection
        # In Pygame, the y axis is downward pointing.
        # In order to make y point upwards, a rotation around x axis by 180 degrees is needed.
        # This will result in y' = -y and z' = -z
        xPrime = x
        yPrime = -y
        xProjected = xPrime * scaling_constant + win_width / 2
        yProjected = yPrime * scaling_constant + win_height / 2
        # Note that there is no negative sign here because our rotation to computer frame
        # assumes that the computer frame is x-right, y-up, z-out
        # so this z-coordinate below is already in the outward direction
        pvDepth.append(z)
        return (round(xProjected), round(yProjected))

    def messageDisplay(self, text, x, y, color):
        textSurface = self.font.render(text, True, color, self.background)
        textRect = textSurface.get_rect()
        textRect.topleft = (x, y)
        self.screen.blit(textSurface, textRect)


def initializeCube(xLen,yLen,zLen,sys):
    xx = xLen/2
    yy = yLen/2
    zz = zLen/2

    block = Wireframe(sys)

    block_nodes = [(x, y, z) for x in (-xx, xx) for y in (-yy, yy) for z in (-zz, zz)]
    node_colors = [(255, 255, 255)] * len(block_nodes)
    block.addNodes(block_nodes, node_colors)
    block.outputNodes()

    faces = [
        (0, 2, 6, 4),
        (0, 1, 3, 2),
        (1, 3, 7, 5),
        (4, 5, 7, 6),
        (2, 3, 7, 6),
        (0, 1, 5, 4)
    ]

    colors = [
        (255,   0, 255), # Purple
        (255,   0,   0), # Red
        (  0, 255,   0), # Green
        (  0,   0, 255), # Blue
        (  0, 255, 255), # Cyan
        (255, 255,   0)  # Yellow
    ]

    block.addFaces(faces, colors)
    block.outputFaces()

    return block



if __name__ == '__main__':
    euler321 = np.array([[ang for ang in range(0,361)] for i in range(3)]) * DEG2RAD

    sys = OpenLoopAttitude()
    block = initializeCube(3,2,0.2,sys)
    pv = ProjectionViewer(640, 480, block)



    for ypr in euler321.T:

        pv.clock.tick(50)

        # sys update attitude
        block.sys.setEuler321(ypr)

        # display updated attitude
        pv.display()
        pygame.display.flip()



