import gc
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric

from halo_data import load_halo, build_star_catalog, log_mem
from spec5.instrument.mock_observations import galactocentric_to_observed, observe_with_spec5

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

RMIN, RMAX = 50, 300
NPART = 5000000
PM_TO_KMS = 4.74047

LON_MIN, LON_MAX, LON_STEP = 240, 300, 2
LAT_MIN, LAT_MAX, LAT_STEP = -90, 0, 2

GALCEN_FRAME = Galactocentric(
    galcen_distance=8.122*u.kpc,
    galcen_v_sun=[12.9, 245.6, 7.78]*u.km/u.s,
    z_sun=0.0208*u.kpc,
)


def cartesian_to_galactic(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    l = np.degrees(np.arctan2(y, x)) % 360
    b = np.degrees(np.arcsin(z / r))
    return l, b


def load_lmc_orbit():
    lmc_orbit = np.loadtxt('../data/GC21M3b1_orbit_lmc.txt')
    mw_orbit = np.loadtxt('../data/GC21M3b1_orbit_mw.txt')
    x = lmc_orbit[:, 1] - mw_orbit[:, 1]
    y = lmc_orbit[:, 2] - mw_orbit[:, 2]
    z = lmc_orbit[:, 3] - mw_orbit[:, 3]
    return cartesian_to_galactic(x, y, z)


def compute_observed(halo):
    stars, _ = build_star_catalog(halo, rmin=RMIN, rmax=RMAX)
    log_mem('compute_observed: star catalog built')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        obs = galactocentric_to_observed(stars, star_type='giant')
        mock = observe_with_spec5(obs, star_type='giant', pm_model='gaia_dr5', seed=42)
    log_mem('compute_observed: mock pipeline done')

    good = mock['distance_obs'] >= 0
    obs = obs[good]
    mock = mock[good]

    icrs_full = SkyCoord(ra=obs['ra']*u.deg, dec=obs['dec']*u.deg, distance=mock['distance_obs']*u.kpc,
                          pm_ra_cosdec=mock['pmra_obs']*u.mas/u.yr, pm_dec=mock['pmdec_obs']*u.mas/u.yr,
                          radial_velocity=mock['vrad_obs']*u.km/u.s, frame='icrs')
    galcen_obs = icrs_full.transform_to(GALCEN_FRAME).represent_as('spherical', s='spherical')
    log_mem('compute_observed: galactocentric transform done')

    l_obs = galcen_obs.lon.deg
    b_obs = galcen_obs.lat.deg
    vrad_obs = galcen_obs.differentials['s'].d_distance.to(u.km/u.s).value

    good_vrad = np.isfinite(vrad_obs)
    l_obs = l_obs[good_vrad]
    b_obs = b_obs[good_vrad]
    vrad_obs = vrad_obs[good_vrad]
    pmra_obs = PM_TO_KMS * mock['pmra_obs'][good_vrad] * mock['distance_obs'][good_vrad]
    pmdec_obs = PM_TO_KMS * mock['pmdec_obs'][good_vrad] * mock['distance_obs'][good_vrad]

    return l_obs, b_obs, vrad_obs, pmra_obs, pmdec_obs


def bin_mean_vrad(l, b, vrad, lon_edges, lat_edges):
    counts, _, _ = np.histogram2d(l, b, bins=[lon_edges, lat_edges])
    sums, _, _ = np.histogram2d(l, b, bins=[lon_edges, lat_edges], weights=vrad)
    with np.errstate(invalid='ignore'):
        mean_vrad = sums / counts
    return mean_vrad, counts


def plot_vr_signal(mean_vrad, lon_edges, lat_edges, l_orbit, b_orbit, name):
    fig, ax = plt.subplots(figsize=(7, 6))
    mesh = ax.pcolormesh(lon_edges, lat_edges, mean_vrad.T, cmap='RdBu', shading='flat')
    ax.set_xlabel('Galactic Longitude $l$ (deg)')
    ax.set_ylabel('Galactic Latitude $b$ (deg)')
    ax.set_title('Observed mean radial velocity (50-300 kpc)')
    ax.set_xlim(lon_edges.max(), lon_edges.min())
    ax.set_ylim(lat_edges.min(), lat_edges.max())

    in_view = (l_orbit >= lon_edges.min()) & (l_orbit <= lon_edges.max()) & \
              (b_orbit >= lat_edges.min()) & (b_orbit <= lat_edges.max())
    ax.plot(l_orbit[in_view], b_orbit[in_view], color='black', lw=1.5, ls='--')
    ax.plot(l_orbit[-1], b_orbit[-1], marker='*', color='black', markersize=14, lw=0)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r'mean $v_{\rm rad}$ (km/s)')

    fig.savefig(f'../figures/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'../figures/{name}.png', bbox_inches='tight', dpi=200)


def main():
    log_mem('main: start')
    halo = load_halo(npart=NPART)
    log_mem('main: load_halo returned')

    distance = np.linalg.norm(halo['pos'], axis=1)
    dcut = (distance > RMIN) & (distance < RMAX)
    halo = {'pos': halo['pos'][dcut], 'vel': halo['vel'][dcut]}
    del distance, dcut
    gc.collect()
    log_mem(f"main: distance cut applied, {len(halo['pos'])} particles remain")

    l_orbit, b_orbit = load_lmc_orbit()
    print(f"LMC current position: l={l_orbit[-1]:.2f} deg, b={b_orbit[-1]:.2f} deg")

    l_obs, b_obs, vrad_obs, pmra_obs, pmdec_obs = compute_observed(halo)
    log_mem('main: compute_observed done')

    lon_edges = np.arange(LON_MIN, LON_MAX + LON_STEP, LON_STEP)
    lat_edges = np.arange(LAT_MIN, LAT_MAX + LAT_STEP, LAT_STEP)

    mean_vrad, counts = bin_mean_vrad(l_obs, b_obs, vrad_obs, lon_edges, lat_edges)
    log_mem('main: binning done')

    plot_vr_signal(mean_vrad, lon_edges, lat_edges, l_orbit, b_orbit, 'wake_vr_signal')
    log_mem('main: plot saved')


if __name__ == '__main__':
    main()
