import numpy as np
from numpy import sin, cos, sqrt, degrees, radians, arctan, arctan2

class GeodeticModel():

    # WGS84 Model constants
    a = 6378137 # (m) Semimajor Axis
    b = 6356752.3142 # (m) Semiminor Axis
    esq = 6.69437999014 * 0.001 # First Eccentricity Squared
    e1sq = 6.73949674228 * 0.001 # Second Eccentricity Squared
    f = 1 / 298.257223563 # Flattening


    def __init__(self, lla0):

        lat0, lon0, alt0 = lla0

        # Save initial lla
        self._lla0 = lla0
        self._ecef0 = self.geodetic2ecef(lla0)

        # ECEF/NED transformation matrices
        ecef_x, ecef_y, ecef_z = self._ecef0
        phiP = arctan2( ecef_z,
                sqrt( ecef_x*ecef_x + ecef_y*ecef_y )
                )

        self.C_toLfromECEF = self.calcR0(phiP, lon0)
        self.C_toECEFfromL = self.calcR0(lat0, lon0).T

    def calcR0(self,lat,lon):
        sLat = sin( lat )
        cLat = cos( lat )
        sLon = sin( lon )
        cLon = cos( lon )

        R0 = np.array([
            [-sLat*cLon, -sLon, cLat*cLon],
            [-sLat*sLon,  cLon, cLat*sLon],
            [      cLat,     0,      sLat],
            ])
        return R0


    def geodetic2ecef(self,lla, wantDeg=False):
        """
        Convert geodetic (WGS84) coordeinates to ECEF using
        World Geodetic System 1984 (WGS84) model

        lat, lon in radians
        alt in meters

        """
        lat,lon,alt = lla

        if wantDeg:
            lat = radians(lat)
            lon = radians(lon)

        xi = sqrt(1 - self.esq * sin(lat)*sin(lat))
        x = (self.a / xi + alt) * cos(lat) * cos(lon)
        y = (self.a / xi + alt) * cos(lat) * sin(lon)
        z = (self.a / xi * (1-self.esq) + alt) * sin(lat)

        return np.array([x, y, z])

    def ecef2geodetic(self, ecef, wantDeg=False):
        """
        Convert ECEF coordinates to geodetic.
        J. Zhu, "Conversion of Earth-centered Earth-fixed coordinates
        to geodetic coordinates," IEEE Transactions on Aerospace and
        Electronic Systems, vol. 30, pp. 957-961, 1994.
        """
        x,y,z = ecef

        r = sqrt(x*x + y*y)
        Esq = self.a*self.a - self.b*self.b
        F = 54 * self.b*self.b * z*z
        G = r*r + (1-self.esq) * z*z - self.esq*Esq
        C = (self.esq*self.esq * F * r*r) / (G**3)
        S = np.cbrt(1 + C + sqrt(C*C + 2*C))
        P = F / (3 * (S + 1 / S + 1)**2 * G*G)
        Q = np.sqrt(1 + 2 * self.esq*self.esq * P)
        r_0 =  -(P * self.esq * r) / (1+Q) + np.sqrt(0.5 * self.a*self.a*(1 + 1.0 / Q) - \
            P * (1-self.esq) * z*z / (Q * (1+Q)) - 0.5 * P * r*r)
        t = (r - self.esq * r_0)**2
        U = np.sqrt(t + z*z)
        V = np.sqrt(t + (1-self.esq) * z*z)
        Z_0 = self.b*self.b * z / (self.a * V)
        alt = U * (1 - self.b*self.b / (self.a * V))
        lat = arctan((z + self.e1sq * Z_0) / r)
        lon = arctan2(y, x)

        if wantDeg:
            lat = degrees(lat)
            lon = degrees(lon)

        return np.array([lat, lon, alt])

    def ecef2ned(self,ecef):
        return np.matmul(self.C_toLfromECEF, ecef - self._ecef0)

    def ned2ecef(self,ned):
        return np.matmul(self.C_toECEFfromL, ned) + self._ecef0

    def geodetic2ned(self,lla):
        return self.ecef2ned( self.geodetic2ecef( lla ) )

    def ned2geodetic(self,ned):
        return self.ecef2geodetic( self.ned2ecef( ned ) )

if __name__ == "__main__":

    RAD2DEG = 180.0/np.pi
    DEG2RAD = 1/RAD2DEG

    # Initial LLA
    lat = 33.333 * DEG2RAD
    lon = -110 * DEG2RAD
    alt = 400 # m
    lla = np.array([lat,lon,alt])

    gm = GeodeticModel(lla*0.99999)
    ecef = gm.geodetic2ecef(lla)
    ned = gm.ecef2ned(ecef)
    ecef2 = gm.ned2ecef(ned)
    lla2 = gm.ecef2geodetic(ecef2)

    print(f"Geodetic Model")
    print(f"  lla    = {lla}")
    print(f"  ecef   = {ecef}")
    print(f"  ned    = {ned}")
    print(f"  ecef2  = {ecef2}")
    print(f"  lla2   = {lla2}")
    print(f"  llaErr = {(lla2-lla)/lla*100}")

