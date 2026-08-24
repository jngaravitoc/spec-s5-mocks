"""Plot a random density field on 1/4 of the sphere using healpy's orthographic projection."""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside = 64
npix = hp.nside2npix(nside)

# Random density field (log-normal so values stay positive).
density = np.random.lognormal(mean=0.0, sigma=0.3, size=npix)

# Split the sphere into 4 quadrants by longitude, the same way the
# Cartesian plane is split into quadrants I-IV: each wedge covers 1/4 of
# the sphere's surface area (a lune of angle theta covers theta/360).
#   Q1: 0-90 deg, Q2: 90-180 deg, Q3: 180-270 deg, Q4: 270-360 deg
theta, phi = hp.pix2ang(nside, np.arange(npix))
lon = np.degrees(phi)
quadrant4_mask = (lon >= 270) & (lon < 360)

density_quarter = np.full(npix, hp.UNSEEN)
density_quarter[quadrant4_mask] = density[quadrant4_mask]

hp.orthview(
    density_quarter,
    rot=(315, 0, 0),   # center the view on the middle of quadrant 4
    half_sky=True,      # show only the visible hemisphere, not front+back
    cmap="inferno",
    badcolor="white",   # hide masked-out pixels instead of showing gray
    bgcolor="white",
    title="Random density field on quadrant 4 of the sphere",
    unit="density",
)
hp.graticule()

plt.savefig("quarter_sphere_density.png", dpi=150, bbox_inches="tight", transparent=True)
plt.show()
