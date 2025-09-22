from databroker import Broker, get_table
import numpy as np
import sys, os
import h5py
try:
    from .widgets.imgTools import rm_outlier_pixels
except ModuleNotFoundError:
    # for test purpose
    from widgets.imgTools import rm_outlier_pixels

try:
    csx_db = Broker.named('csx')
except FileNotFoundError:
    print("csx.yml not found. Unable to access CSX's database.", file=sys.stderr)
    csx_db = None
from csxtools.utils import get_fastccd_images, get_images_to_4D, get_fastccd_flatfield


# ***************************** "Public API" *****************************
# The following functions must exist in nsls2ptycho/core/*_databroker.py,
# but the function signatures do not need to agree across modules
# (obviously, it is impossible for all beamlines to have the same setup).
#   - load_metadata
#   - save_data
#   - get_single_image
#   - get_detector_names
# Other function must not be imported in the GUI.
# ************************************************************************


# CSX's fastccd detector has a vertical dark stripe
cs = 486     # pixel start point
cl = 28      # width
cedge = 988  # detector edge


def load_metadata(db, scan_num:int, det_name:str=''):
    '''
    Get all metadata for the given scan number and detector name

    Parameters:
        - db: 
            a Broker instance.
        - scan_num: int
            the scan number
        - det_name: str
            the detector name; not used in CSX

    Return:
        A dictionary that holds the metadata (except for those directly related to the image)
    '''
    header = db[scan_num]
    scan_data = header.table(fill=False, stream_name='primary')  # acquired counters during the scan/ct, images excluded
    scan_supp = header.table(fill=False, stream_name='baseline') # supplementary information, images excluded

    scan_type = header.start['plan_name']
    plan_args = header.start['plan_args']
    nz = header.start['num_points']

    # get energy_kev
    energy_kev = scan_supp.pgm_energy_setpoint.iat[0] / 1E+3

    # get scan_type, x_range, y_range, dr_x, dr_y
    if scan_type.endswith('grid_scan'):
        extents = header.start['extents']
        x_num = plan_args['args'][7]
        y_num = plan_args['args'][3]
        x_range = (extents[0][1] - extents[0][0]) * 1E+3
        y_range = (extents[1][1] - extents[1][0]) * 1E+3
        dr_x = x_range / (x_num - 1)
        dr_y = y_range / (y_num - 1)

        # get points
        points = np.zeros((2, nz))
        points[0] = scan_data.nanop_bx * 1E+3
        points[1] = scan_data.nanop_bz * 1E+3
    elif scan_type == 'spiral_continuous':
        extents = header.start['extents']
        # The customized plan messed up...
        x_range = (extents[0][1] - extents[0][0])/2 * 1E+3
        y_range = (extents[1][1] - extents[1][0])/2 * 1E+3
        # not available in CSX
        dr_x = 0
        dr_y = 0

        # get points
        points = np.zeros((2, nz))
        points[0] = scan_data.nanop_bx * 1E+3
        points[1] = scan_data.nanop_bz * 1E+3
    elif scan_type == 'scan_nd':  # for custom plans with a zone plate
        # this is a mesh scan, but a lot of info is missing in the plan...
        # set the missing values to zero
        x_range = 0
        y_range = 0
        dr_x = 0
        dr_y = 0

        # get points (from top motors, not bottom ones!)
        points = np.zeros((2, nz))
        points[0] = scan_data.nanop_tx * 1E+3
        points[1] = scan_data.nanop_tz * 1E+3
    else:
        raise NotImplementedError("Ask Wen Hu to explain to Leo Fang.")
    # get angle, ic
    angle = 90 - scan_supp.tardis_theta.iat[0]  # TODO(leofang): verify this
    
    # get ccd_pixel_um
    ccd_pixel_um = 30.

    # write everything we need to a dict
    metadata = dict()
    metadata['xray_energy_kev'] = energy_kev
    metadata['scan_type'] = scan_type
    metadata['dr_x'] = dr_x
    metadata['dr_y'] = dr_y
    metadata['x_range'] = x_range
    metadata['y_range'] = y_range
    metadata['points'] = points
    metadata['angle'] = angle
    metadata['ccd_pixel_um'] = ccd_pixel_um
    metadata['nz'] = nz
    metadata['nx'] = 960
    metadata['ny'] = 960
    metadata['z_m'] = 0.34

    return metadata


def save_data(db, param, scan_num:int, nx_prb:int, ny_prb:int, cx:int, cy:int, threshold=2., bad_pixels=None, zero_out=None, upsample=1):
    '''
    Save metadata and diffamp for the given scan number to a HDF5 file.

    Parameters:
        - db: 
            a Broker instance.
        - param: Param
            a Param instance containing the metadata and other information from the GUI
        - scan_num: int
            the scan number
        - nx_prb: int
            the x dimension of the ROI window
        - ny_prb: int
            the y dimension of the ROI window
        - cx: int
            x index of the center of mass
        - cy: int
            y index of the center of mass
        - threshold: float, optional
            the threshold of raw image, below which the data is removed
        - bad_pixels: list of two lists, optional
            the data structure is [[x1, x2, ...], [y1, y2, ...]]. If given, they will be removed from the images.
        - zero_out: list of tuples, optional
            zero out the given rois [(x0, y0, w0, h0), (x1, y1, w1, h1), ...]
        - upsample: detector upsampling ratio, optional
            positive integer

    Notes:
        1. the detector distance is assumed existent as param.z_m
    '''
    if bad_pixels is not None:
        print("[WARNING] Bad-pixel removal is not yet supported for CSX.", file=sys.stderr)
        bad_pixels = None

    det_distance_m = param.z_m
    det_pixel_um = param.ccd_pixel_um / upsample
    num_frame = param.nz
    angle = param.angle
    lambda_nm = param.lambda_nm

    x_pixel_m = lambda_nm * 1.e-9 * det_distance_m / (nx_prb * det_pixel_um * 1e-6)
    y_pixel_m = lambda_nm * 1.e-9 * det_distance_m / (ny_prb * det_pixel_um * 1e-6)
    x_depth_of_field_m = lambda_nm * 1.e-9 / (nx_prb / 2 * det_pixel_um*1.e-6 / det_distance_m)**2
    y_depth_of_field_m = lambda_nm * 1.e-9 / (ny_prb / 2 * det_pixel_um*1.e-6 / det_distance_m)**2
    #x_depth_of_field_m = 1e-6
    #y_depth_of_field_m = 1e-6
    
    # retrieve scan coordinates
    points = param.points
    ## retrieve scan coordinates from Databroker
    #scan_data = db[scan_num].table(fill=False, stream_name='primary')  # acquired counters during the scan/ct, images excluded
    #points = np.zeros((2, num_frame))
    #points[0] = scan_data.nanop_bx * 1000.
    #points[1] = scan_data.nanop_bz * 1000.

    # get the slicerator corresponding to the scan number
    key = _expand_partial_key(scan_num)
    itr = scan_image_itr_cache[key]
    assert num_frame == len(itr)

    # get raw data
    images_stack = get_images_to_4D(itr)
    raw_mean_data = _preprocess_image(images_stack, bad_pixels, zero_out)

    # construct data array
    diffamp = np.empty((num_frame, nx_prb//2*2, ny_prb//2*2))
    diffamp[...] = np.rot90(raw_mean_data[:, cy-ny_prb//2:cy+ny_prb//2, cx-nx_prb//2:cx+nx_prb//2], axes=(2, 1))  # equivalent to np.flipud(arr).T
    diffamp = np.fft.fftshift(diffamp, axes=(1, 2))
    diffamp = np.sqrt(diffamp)
    diffamp[diffamp < threshold] = 0.
    assert diffamp.shape == (num_frame, nx_prb, ny_prb)
    if upsample > 1:
        diffamp = np.kron(diffamp,np.ones((1,upsample,upsample)))
    print('array size:', diffamp.shape)

    # create a folder
    try:
        os.mkdir(param.working_directory + '/h5_data/')
    except FileExistsError:
        pass 

    file_path = param.working_directory + '/h5_data/scan_' + str(scan_num) + '.h5'
    with h5py.File(file_path, 'w') as hf:
        dset = hf.create_dataset('diffamp', data=diffamp)
        dset = hf.create_dataset('points', data=points)
        dset = hf.create_dataset('x_range', data=param.x_range)
        dset = hf.create_dataset('y_range', data=param.y_range)
        dset = hf.create_dataset('dr_x', data=param.dr_x)
        dset = hf.create_dataset('dr_y', data=param.dr_y)
        dset = hf.create_dataset('z_m',data=det_distance_m)
        dset = hf.create_dataset('lambda_nm',data=lambda_nm)
        dset = hf.create_dataset('ccd_pixel_um',data=det_pixel_um)
        dset = hf.create_dataset('angle',data=angle)
        dset = hf.create_dataset('x_pixel_m',data=x_pixel_m)
        dset = hf.create_dataset('y_pixel_m',data=y_pixel_m)
        dset = hf.create_dataset('x_depth_field_m',data=x_depth_of_field_m)
        dset = hf.create_dataset('y_depth_field_m',data=y_depth_of_field_m)

    # symlink so ptycho can find it
    try:
        symlink_path = param.working_directory + '/scan_' + str(scan_num) + '.h5'
        os.symlink(file_path, symlink_path)
    except FileExistsError:
        os.remove(symlink_path)
        os.symlink(file_path, symlink_path)

'''
For actual scans:
    key : value = (scan num, dark8 num, dark2 num, dark1 num) : slicerator

For flat-field scans:
    key : value = (scan num, dark8 num, dark2 num, dark1 num) : flat image
'''
scan_image_itr_cache = {}


def _load_scan_image_itr(db, scan_num:int, dark8:int=None, dark2:int=None, dark1:int=None, key_flat:tuple=None):
    bgnd8 = db[dark8] if (dark8 is not None) else None
    bgnd2 = db[dark2] if (dark2 is not None) else None
    bgnd1 = db[dark1] if (dark1 is not None) else None
    if (bgnd8 is None) and (bgnd2 is None) and (bgnd1 is None):
        dark_headers = None
    else:
        dark_headers = (bgnd8, bgnd2, bgnd1)

    flat_scan_num, flat_scan_dark8, flat_scan_num_dark2, flat_scan_dark1 = key_flat
    if flat_scan_num is not None:
        flat_im = scan_image_itr_cache.get(key_flat)  # load flat field image from cache
        if flat_im is None:
            flat_im = get_fastccd_flatfield(db[flat_scan_num], dark=(db[flat_scan_dark8], db[flat_scan_num_dark2], db[flat_scan_dark1]))
            scan_image_itr_cache[key_flat] = flat_im
    else:
        flat_im = None

    silcerator = get_fastccd_images(db[scan_num], dark_headers=dark_headers, flat=flat_im)
    return silcerator


def get_single_image(db, frame_num, scan_num:int, dark8:int=None, dark2:int=None, dark1:int=None,
                     flat_scan_num:int=None, flat_scan_dark8:int=None, flat_scan_num_dark2:int=None, flat_scan_dark1:int=None):
    # TODO: use mds_table here
    key_flat = (flat_scan_num, flat_scan_dark8, flat_scan_num_dark2, flat_scan_dark1)
    key = (scan_num, dark8, dark2, dark1, key_flat)

    if key in scan_image_itr_cache:
        return _preprocess_image(scan_image_itr_cache[key][frame_num])
    else:
        itr = _load_scan_image_itr(db, scan_num, dark8, dark2, dark1, key_flat)
        scan_image_itr_cache[key] = itr
        return _preprocess_image(itr[frame_num])


def _preprocess_image(img, bad_pixels=None, zero_out=None):
    # TODO: support this
    assert bad_pixels is None

    # average over the axis corresponding to the same scan point,
    # and then remove the stripe
    if img.ndim == 3:
        axis = 0
        stack = np.hstack
    elif img.ndim == 4:
        axis = 1
        stack = np.dstack
    else:
        raise ValueError("The image array's shape is not supported.")

    img = np.nan_to_num(img, copy=False)
    img[img < 0.] = 0.  # needed due to dark subtraction
    img = np.mean(img, axis=axis)
    img = stack((img[..., :cs], img[..., cs+cl:cedge]))

    # The coordinates we got from GUI have the stripe removed
    if zero_out is not None:
        for blue_roi in zero_out:
            x0, y0, w, h = blue_roi
            img[..., y0:y0+h, x0:x0+w] = 0.

    return img


def _expand_partial_key(scan_num:int):
    # get the full key based on scan_num
    # We don't wanna guess the dark IDs, so we rely on cached info
    related_keys = {}
    for key in scan_image_itr_cache:
        if scan_num in key:
            # sort key based on the number of provided dark IDs
            # prefer a complete info
            if key[4][0] is not None:  # found match with flat-field
                related_keys[5] = key
                break  # shortcut
            if key[3] is not None:
                related_keys[4] = key
                break  # shortcut
            elif key[2] is not None:
                related_keys[3] = key
            elif key[1] is not None:
                related_keys[2] = key
            else:
                related_keys[1] = key

    if 5 in related_keys:
        key = related_keys[5]
    elif 4 in related_keys:
        key = related_keys[4]
    elif 3 in related_keys:
        key = related_keys[3]
    elif 2 in related_keys:
        key = related_keys[2]
    elif 1 in related_keys:
        key = related_keys[1]
        print("[WARNING] Proceeding without dark IDs...", file=sys.stderr)
    else:
        raise ValueError("Data for scan number", scan_num, "not found. Forget to click load?")
    print("Found", key) # "from", related_keys)

    return key


def get_detector_names(db, scan_num:int):
    '''
    Returns
    -------
        list: detectors used in the scan or available in the beamline
    '''
    # TODO: make it more robust, like reading from databroker?
    return ['fccd']
