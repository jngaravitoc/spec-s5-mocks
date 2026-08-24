import gc
import resource

import h5py
import numpy as np
import nba


def log_mem(stage):
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[mem] {stage}: peak RSS so far = {peak_gb:.2f} GB", flush=True)


def load_halo(sim_dir='../../mwlmc_sims',
              sim_file='MWLMC5_100M_b1_vir_OM3_G4_111.hdf5',
              npart=10000, center_npart=100000):
    sim = nba.ios.ReadGC21(sim_dir, sim_file)
    log_mem('load_halo: sim opened')

    center_data = sim.read_halo(['pos', 'vel', 'mass'],
                                 halo='MW', ptype='dm', randomsample=center_npart)
    log_mem('load_halo: centering read done')
    mw_center = nba.com.CenterHalo(center_data)
    pos_mw, vel_mw = mw_center.shrinking_sphere()
    log_mem('load_halo: shrinking_sphere done')

    del center_data, mw_center
    gc.collect()
    log_mem('load_halo: centering data freed')

    halo = sim.read_halo(['pos', 'vel'], halo='MW', ptype='dm', randomsample=npart)
    log_mem('load_halo: main read done')
    halo['pos'] = halo['pos'] - pos_mw
    halo['vel'] = halo['vel'] - vel_mw
    return halo


def build_star_catalog(halo, rmin=50, rmax=300, lum_mean_log=50, lum_sigma=0.5, seed=42):
    rng = np.random.default_rng(seed)

    distance = np.linalg.norm(halo['pos'], axis=1)
    dcut = np.where((distance > rmin) & (distance < rmax))[0]

    x = halo['pos'][dcut, 0]
    y = halo['pos'][dcut, 1]
    z = halo['pos'][dcut, 2]
    vx = halo['vel'][dcut, 0]
    vy = halo['vel'][dcut, 1]
    vz = halo['vel'][dcut, 2]

    lum = np.exp(rng.normal(np.log(lum_mean_log), lum_sigma, len(x)))

    dtype = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
             ('vx', 'f8'), ('vy', 'f8'), ('vz', 'f8'),
             ('luminosity', 'f8')]
    stars = np.array(list(zip(x, y, z, vx, vy, vz, lum)), dtype=dtype)

    params = dict(seed=seed, dcut_range=(rmin, rmax),
                  lum_mean_log=lum_mean_log, lum_sigma=lum_sigma)
    return stars, params


def write_stars_hdf5(stars, filename, seed, dcut_range,
                      lum_mean_log, lum_sigma, dataset_name='stars'):
    """
    Write a structured star catalog (as produced by build_star_catalog) to
    an HDF5 file as a compound-dtype dataset.

    Parameters
    ----------
    stars : np.ndarray
        Structured array with fields x, y, z, vx, vy, vz, luminosity.
    filename : str
        Output .h5 file path.
    seed : int
        RNG seed used for the synthetic luminosities (stored as metadata).
        Pass the same value used in build_star_catalog.
    dcut_range : tuple(float, float)
        (min, max) distance cut applied to halo positions (stored as metadata).
        Pass the same (rmin, rmax) used in build_star_catalog.
    lum_mean_log : float
        Mean (in linear space, before log) used for the lognormal luminosity.
    lum_sigma : float
        Sigma of the underlying normal distribution (in log space).
    dataset_name : str
        Name of the dataset inside the HDF5 file.

    Returns
    -------
    stars : np.ndarray
        The structured array that was written to disk.
    """
    with h5py.File(filename, 'w') as f:
        ds = f.create_dataset(dataset_name, data=stars, compression='gzip')
        ds.attrs['description'] = 'Stars selected by distance cut with synthetic luminosities'
        ds.attrs['distance_cut_min'] = dcut_range[0]
        ds.attrs['distance_cut_max'] = dcut_range[1]
        ds.attrs['n_stars'] = len(stars)
        ds.attrs['luminosity_model'] = 'lognormal'
        ds.attrs['luminosity_mean_log_input'] = lum_mean_log
        ds.attrs['luminosity_sigma'] = lum_sigma
        ds.attrs['seed'] = seed

    print(f"Wrote {len(stars)} stars to '{filename}' as dataset '{dataset_name}'")

    return stars


if __name__ == '__main__':
    halo = load_halo()
    stars, params = build_star_catalog(halo)
    write_stars_hdf5(stars, '../data/mwlmc_outer_halo_stars.h5', **params)
