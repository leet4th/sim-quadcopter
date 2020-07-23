import numpy as np

class BasicNavigation:

    def __init__(self, quad):
        self.__quad = quad
        
        self.update()
        
    def update(self):
        
        self.pos_B     = self.__quad.pos_B
        self.pos_L     = self.__quad.pos_L
        self.qToBfromL = self.__quad.qToBfromL
        self.vel_B     = self.__quad.vel_B
        self.vel_L     = self.__quad.vel_L
        self.wb        = self.__quad.wb
        self.wm        = self.__quad.wm
        self.velDot_B  = self.__quad.velDot_B
        self.velDot_L  = self.__quad.velDot_L
        self.qDot      = self.__quad.qDot
        self.wbDot     = self.__quad.wbDot
        self.wmDot     = self.__quad.wmDot