import numpy as np
import helapy as hp

def mollweide_projection(l, b, l2, b2, title, bmin, bmax, nside, smooth, q=[0], **kwargs):
    """
    Makes mollweide plot using healpix
    Parameters:
    ----------- 
    l : numpy.array in degrees 
    b : numpy.array in degrees [-90, 90]
    """
 
    mwlmc_indices = hp.ang2pix(nside,  (90-b)*np.pi/180., l*np.pi/180.)
    npix = hp.nside2npix(nside)
 
    idx, counts = np.unique(mwlmc_indices, return_counts=True)
    degsq = hp.nside2pixarea(nside, degrees=True)
    # filling the full-sky map
    hpx_map = np.zeros(npix, dtype=float)
    if q[0] != 0 :    
        counts = np.zeros_like(idx, dtype=float)
        k=0
        for i in idx:
            pix_ids = np.where(mwlmc_indices==i)[0]
            counts[k] = np.mean(q[pix_ids])
            k+=1
        hpx_map[idx] = counts
    else :
       hpx_map[idx] = counts/degsq

    map_smooth = hp.smoothing(hpx_map, fwhm=smooth*np.pi/180)
   
    if ((bmin == 'auto') & (bmax == 'auto')):
        bmin = np.min(map_smooth)
        bmax = np.max(map_smooth)

    if 'cmap' in kwargs.keys():
        cmap = kwargs['cmap']
    else:
        cmap='viridis'
        
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plt.close()
    hp.projview(
      map_smooth,
      coord=["G"],
      graticule=True,
      graticule_labels=True,
      rot=(0, 0, 0),
      unit=" ",
      #xlabel="Galactic Longitude (l) ",
      ylabel="Galactic Latitude (b)",
      cb_orientation="horizontal",
      min=bmin,
      max=bmax,
      latitude_grid_spacing=45,
      projection_type="mollweide",
      title=title,
      cmap=cmap,
      fontsize={
              "xlabel": 25,
              "ylabel": 25,
              "xtick_label": 20,
              "ytick_label": 20,
              "title": 25,
              "cbar_label": 20,
              "cbar_tick_label": 20,
              },
      )
	
    #hp.newprojplot(theta=np.radians(90-(b2)), phi=np.radians(l2), marker="o", color="yellow", markersize=5, lw=0, mfc='none')
    if 'l3' in kwargs.keys():
        l3 = kwargs['l3']
        b3 = kwargs['b3']
        hp.newprojplot(theta=np.radians(90-(b3)), phi=np.radians(l3), marker="o", color="yellow", markersize=5, lw=0)
    elif 'l4' in kwargs.keys():
        l4 = kwargs['l4']
        b4 = kwargs['b4']
        hp.newprojplot(theta=np.radians(90-(b4)), phi=np.radians(l4), marker="*", color="r", markersize=8, lw=0)

    #newprojplot(theta=np.radians(90-(b2[0])), phi=np.radians(l2[0]-120), marker="*", color="r", markersize=5 )
    #newprojplot(theta=np.radians(90-(b2[1])), phi=np.radians(l2[1]-120), marker="*", color="w", markersize=2 )
    
    return fig

        

