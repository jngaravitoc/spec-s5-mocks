import warnings

import h5py
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric
from matplotlib.lines import Line2D

from halo_data import load_halo, build_star_catalog, log_mem
from spec5.instrument.mock_observations import galactocentric_to_observed, observe_with_spec5

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

RMIN, RMAX = 50, 300
NPART = 2000000
NSIDE = 32
SMOOTH_DEG = 10
PM_TO_KMS = 4.74047

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


def plot_wrapped_line(l, b, **kwargs):
    l_wrapped = ((np.asarray(l) + 180) % 360) - 180
    b = np.asarray(b)
    wrap_idx = np.where(np.abs(np.diff(l_wrapped)) > 180)[0] + 1
    for seg_l, seg_b in zip(np.split(l_wrapped, wrap_idx), np.split(b, wrap_idx)):
        hp.newprojplot(theta=np.radians(90 - seg_b), phi=np.radians(seg_l), **kwargs)


def footprint_boundary(mask, nside):
    inside_idx = np.where(mask)[0]
    neighbours = hp.get_all_neighbours(nside, inside_idx)
    valid = neighbours >= 0
    edge = np.zeros(mask.shape, dtype=bool)
    for col in range(neighbours.shape[1]):
        neigh = neighbours[valid[:, col], col]
        if not np.all(mask[neigh]):
            edge[inside_idx[col]] = True
    return hp.pix2ang(nside, np.where(edge)[0], lonlat=True)


def load_lmc_orbit():
    lmc_orbit = np.loadtxt('../data/GC21M3b1_orbit_lmc.txt')
    mw_orbit = np.loadtxt('../data/GC21M3b1_orbit_mw.txt')
    x = lmc_orbit[:, 1] - mw_orbit[:, 1]
    y = lmc_orbit[:, 2] - mw_orbit[:, 2]
    z = lmc_orbit[:, 3] - mw_orbit[:, 3]
    return cartesian_to_galactic(x, y, z)


def plot_density(l, b, nside, smooth_deg):
    pix = hp.ang2pix(nside, (90 - b) * np.pi / 180., l * np.pi / 180.)
    npix = hp.nside2npix(nside)
    idx, counts = np.unique(pix, return_counts=True)
    degsq = hp.nside2pixarea(nside, degrees=True)
    density_map = np.zeros(npix)
    density_map[idx] = counts / degsq
    density_smooth = np.clip(hp.smoothing(density_map, fwhm=np.radians(smooth_deg)), 0, None)

    mean_density = np.mean(density_smooth)
    contrast = density_smooth / mean_density - 1

    hp.projview(
        contrast,
        coord=["G"],
        graticule=True,
        graticule_labels=True,
        rot=(0, 0, 0),
        unit=r'$\rho/\bar\rho - 1$',
        ylabel='Galactic Latitude (b)',
        cb_orientation='horizontal',
        min=-0.3, max=0.3,
        latitude_grid_spacing=45,
        projection_type='mollweide',
        title='Outer halo density contrast (50-300 kpc)',
        cmap='RdBu_r',
        fontsize={
            "xlabel": 25, "ylabel": 25,
            "xtick_label": 20, "ytick_label": 20,
            "title": 25, "cbar_label": 20, "cbar_tick_label": 20,
        },
    )


def bin_average(l, b, q, nside, smooth_deg):
    pix = hp.ang2pix(nside, (90 - b) * np.pi / 180., l * np.pi / 180.)
    npix = hp.nside2npix(nside)
    idx = np.unique(pix)
    binned = np.zeros(npix)
    for i in idx:
        binned[i] = np.mean(q[pix == i])
    return hp.smoothing(binned, fwhm=np.radians(smooth_deg))


def plot_velocity_map(l, b, q, title, unit, nside, smooth_deg):
    smoothed = bin_average(l, b, q, nside, smooth_deg)
    vmax = np.max(np.abs(smoothed))

    hp.projview(
        smoothed,
        coord=["G"],
        graticule=True,
        graticule_labels=True,
        rot=(0, 0, 0),
        unit=unit,
        ylabel='Galactic Latitude (b)',
        cb_orientation='horizontal',
        min=-vmax, max=vmax,
        latitude_grid_spacing=45,
        projection_type='mollweide',
        title=title,
        cmap='RdBu',
        fontsize={
            "xlabel": 25, "ylabel": 25,
            "xtick_label": 20, "ytick_label": 20,
            "title": 25, "cbar_label": 20, "cbar_tick_label": 20,
        },
    )


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


def plot_footprints():
    with h5py.File('../data/specs5_footprints.h5', 'r') as f:
        wide_mask = f['wide_mask'][:]
        deep_mask = f['deep_mask'][:]
        nside = int(f.attrs['nside'])

    l_wide, b_wide = footprint_boundary(wide_mask, nside)
    plot_wrapped_line(l_wide, b_wide, marker='.', markersize=1, lw=0, color='black')

    l_deep, b_deep = footprint_boundary(deep_mask, nside)
    plot_wrapped_line(l_deep, b_deep, marker='.', markersize=1, lw=0, color='darkblue')


def plot_lmc_orbit():
    l_orbit, b_orbit = load_lmc_orbit()
    plot_wrapped_line(l_orbit, b_orbit, color='black', lw=1.5, ls='--')
    hp.newprojplot(theta=np.radians(90 - b_orbit[-1]), phi=np.radians(((l_orbit[-1] + 180) % 360) - 180),
                    marker='*', color='black', markersize=10, lw=0)


def add_legend():
    legend_elements = [
        Line2D([0], [0], color='black', lw=1.5, label='Spec-S5 Wide'),
        Line2D([0], [0], color='darkblue', lw=1.5, label='Spec-S5 Deep'),
        Line2D([0], [0], color='black', lw=1.5, ls='--', label='LMC orbit'),
        Line2D([0], [0], marker='*', color='black', lw=0, markersize=10, label='LMC ($t=0$)'),
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=14)


def finalize_and_save(name):
    plot_footprints()
    plot_lmc_orbit()
    add_legend()
    fig = plt.gcf()
    fig.savefig(f'../figures/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'../figures/{name}.png', bbox_inches='tight', dpi=200)


def main():
    log_mem('main: start')
    halo = load_halo(npart=NPART)
    log_mem('main: load_halo returned')
    distance = np.linalg.norm(halo['pos'], axis=1)
    dcut = (distance > RMIN) & (distance < RMAX)
    l, b = cartesian_to_galactic(halo['pos'][dcut, 0], halo['pos'][dcut, 1], halo['pos'][dcut, 2])

    plot_density(l, b, NSIDE, SMOOTH_DEG)
    finalize_and_save('wake_specs5_footprint')
    log_mem('main: density plot done')

    l_obs, b_obs, vrad_obs, pmra_obs, pmdec_obs = compute_observed(halo)
    log_mem('main: compute_observed done')

    plot_velocity_map(l_obs, b_obs, vrad_obs,
                       'Observed radial velocity (50-300 kpc)', 'km/s', NSIDE, SMOOTH_DEG)
    finalize_and_save('wake_specs5_vrad')
    log_mem('main: vrad plot done')

    plot_velocity_map(l_obs, b_obs, pmra_obs,
                       'Observed proper motion RA (50-300 kpc)', 'km/s', NSIDE, SMOOTH_DEG)
    finalize_and_save('wake_specs5_pmra')
    log_mem('main: pmra plot done')

    plot_velocity_map(l_obs, b_obs, pmdec_obs,
                       'Observed proper motion Dec (50-300 kpc)', 'km/s', NSIDE, SMOOTH_DEG)
    finalize_and_save('wake_specs5_pmdec')
    log_mem('main: pmdec plot done')


if __name__ == '__main__':
    main()
