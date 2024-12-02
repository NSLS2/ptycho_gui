from databroker.v0 import Broker
from . import CompositeBroker
from databroker.headersource.mongo import MDS
import numpy as np
import sys, os, warnings, pandas
import h5py
try:
    from .widgets.imgTools import rm_outlier_pixels
except ModuleNotFoundError:
    # for test purpose
    from widgets.imgTools import rm_outlier_pixels

#from hxntools.handlers import register
from .scan_info import ScanInfo
try:
    # new mongo database
    hxn_db = CompositeBroker.db
    #register(hxn_db)
except FileNotFoundError:
    print("hxn.yml not found. Unable to access HXN's database.", file=sys.stderr)
    hxn_db = None


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


def array_ensure_positive_elements(arr, name="array"):
    """
    Replace all zero or negative values in the array with the closest positive values.
    Works only with 1D arrays. The array is processed in the reversed order to replace
    zeros with the following (not preceding) values, because they are more likely to
    be from the same subscan (HXN).

    The function will do nothing if there are no zeros or negative values in the array.

    Parameters
    ----------
    arr: numpy.ndarray
        Reference to 1D numpy array. The values are modified in place.
    name: str
        Data name to use in error messages.

    Returns
    -------
    None
    """
    n_items_to_replace = sum(arr <= 0)

    # Exit right away if there scaler data is valid (this should be true for correctly recorded scans)
    if not n_items_to_replace:
        return

    # TODO: the algorithm may be implemented more efficiently if needed. Scalers are loaded only once
    #       per reconstruction, and the correction does not introduce noticable delay. Rewrite
    #       the function if performance becomes an issue.
    v_closest_positive = None
    for v in np.flip(
        arr
    ):  # Initialize the algorithm with some valid value in case the 1st element is zero.
        if v > 0:
            v_closest_positive = v
            break

    n_replaced = 0
    if v_closest_positive is not None:
        for n in reversed(range(arr.size)):
            if arr[n] <= 0:
                print(
                    f"{name.capitalize()} value {arr[n]} with index {n} is replaced "
                    f"with the closest value {v_closest_positive}."
                )
                arr[n] = v_closest_positive
                n_replaced += 1
                if n_replaced == n_items_to_replace:
                    break
            else:
                v_closest_positive = arr[n]
    else:
        print(
            f"The {name} contains no positive non-zero values. Computations are likely to fail."
        )


def load_metadata(db, scan_num:int, det_name:str):
    '''
    Get all metadata for the given scan number and detector name

    Parameters:
        - db:
            a Broker instance. For HXN experiments they are db1, db2, and db_old
        - scan_num: int
            the scan number
        - det_name: str
            the detector name

    Return:
        A dictionary that holds the metadata (except for those directly related to the image)
    '''
    sid = scan_num
    header = db[sid]
    warnings.filterwarnings("ignore", category = pandas.errors.PerformanceWarning)
    
    if 'plan_args' in header.start:
        plan_args = header.start['plan_args']
        scan_type = header.start['plan_name']
        scan_motors = header.start['motors']
        items = [det_name, 'sclr1_ch3', 'sclr1_ch4'] + scan_motors
        bl = db.get_table(header, stream_name='baseline')
        df = db.get_table(header, fields=items, fill=False)
        #images = db_old.get_images(db_old[sid], name=det_name)

        # get energy_kev
        dcm_th = bl.dcm_th[1]
        energy_kev = 12.39842 / (2.*3.1355893 * np.sin(dcm_th * np.pi / 180.))

        # get scan_type, x_range, y_range, dr_x, dr_y
        if scan_type == 'FlyPlan2D':
            x_range = plan_args['scan_end1']-plan_args['scan_start1']
            y_range = plan_args['scan_end2']-plan_args['scan_start2']
            x_num = plan_args['num1']
            y_num = plan_args['num2']
            dr_x = 1.*x_range/x_num
            dr_y = 1.*y_range/y_num
            x_range = x_range - dr_x
            y_range = y_range - dr_y
        elif scan_type == 'rel_spiral_fermat' or scan_type == 'fermat':
            x_range = plan_args['x_range']
            y_range = plan_args['y_range']
            dr_x = plan_args['dr']
            dr_y = 0
        else:
            x_range = plan_args['args'][2]-plan_args['args'][1]
            y_range = plan_args['args'][6]-plan_args['args'][5]
            x_num = plan_args['args'][3]
            y_num = plan_args['args'][7]
            dr_x = 1.*x_range/x_num
            dr_y = 1.*y_range/y_num
            x_range = x_range - dr_x
            y_range = y_range - dr_y

        # get points
        num_frame, count = np.shape(df)
        points = np.zeros((2, num_frame))
        points[0] = np.array(df[scan_motors[0]])
        points[1] = np.array(df[scan_motors[1]])

        # get angle, ic
        if scan_motors[1] == 'ssy':
            angle = 0#bl.zpsth[1]
            ic = np.asfarray(df['sclr1_ch3'])
        elif scan_motors[1] == 'zpssy':
            if 'tomo_angle_offset'in header.start:
                angle_offset = header.start['tomo_angle_offset']
            else:
                angle_offset = -0.2
            if 'x_scale_factor'in header.start:
                x_scale_factor = header.start['x_scale_factor']
            else:
                x_scale_factor = 0.9542
            if 'z_scale_factor'in header.start:
                z_scale_factor = header.start['z_scale_factor']
            else:
                z_scale_factor = 1.0309
            angle = bl.zpsth[1] - angle_offset
            if scan_motors[0] == 'zpssx':
                points[0] = points[0] * x_scale_factor
                dr_x *= x_scale_factor
            elif scan_motors[0] == 'zpssz':
                points[0] = points[0] * z_scale_factor
                dr_x *= z_scale_factor
            ic = np.asfarray(df['sclr1_ch4'])
        else:
            angle = bl.dsth[1]
            ic = np.asfarray(df['sclr1_ch4'])
        array_ensure_positive_elements(ic, name="scaler")

        if 'merlin2' in header.start['detectors']:
            # get ccd_pixel_um
            ccd_pixel_um = 55.

            z_m = 0.5
        elif 'eiger1' in header.start['detectors']:
            ccd_pixel_um = 75.

            z_m = 2.05
        else: 
            print("[WARNING] Unknown detector! Input detector pixel size and detector distance manually in ptycho recon.")

            ccd_pixel_um = 55.

            z_m = 0.5
        # get diffamp dimensions (uncropped!)
        nz, = df[det_name].shape
        mds_table = df[det_name]

        # get nx and ny by looking at the first image
        # img = db.reg.retrieve(mds_table.iat[0])[0]
        # nx, ny = img.shape # can also give a ValueError; TODO: come up a better way!
    elif 'scan' in header.start:
        if 'panda1' in header.start['scan']['detectors']:
            scan_type = header.start['plan_name']
            scan_doc = header.start['scan']
            scan_motors = [scan_doc['fast_axis']['motor_name'], scan_doc['slow_axis']['motor_name']]
            if header.start['plan_name'].startswith('pt'):
                items = [det_name, 'sclr3_ch4', 'inenc1_val', 'inenc2_val', 'inenc3_val', 'inenc4_val']
                ic_chan = 'sclr3_ch4'
            else:
                items = [det_name, 'sclr1_ch4', 'inenc1_val', 'inenc2_val', 'inenc3_val', 'inenc4_val']
                ic_chan = 'sclr1_ch4'
            bl = db.get_table(header, stream_name='baseline')
            df = db.get_table(header, fields=items, fill=False)
            #images = db_old.get_images(db_old[sid], name=det_name)

            # get energy_kev
            dcm_th = bl.dcm_th[1]
            energy_kev = 12.39842 / (2.*3.1355893 * np.sin(dcm_th * np.pi / 180.))

            if scan_motors[0].endswith('ssy'):
                y_range = np.abs(scan_doc['scan_input'][1] - scan_doc['scan_input'][0])
                x_range = np.abs(scan_doc['scan_input'][4] - scan_doc['scan_input'][3])
                y_num = scan_doc['scan_input'][2]
                x_num = scan_doc['scan_input'][5]        
            elif not header.start['plan_name'].startswith('pt'):
                x_range = np.abs(scan_doc['scan_input'][1] - scan_doc['scan_input'][0])
                y_range = np.abs(scan_doc['scan_input'][4] - scan_doc['scan_input'][3])
                x_num = scan_doc['scan_input'][2]
                y_num = scan_doc['scan_input'][5]       
            else:
                x_range = np.abs(scan_doc['scan_input'][1])
                y_range = np.abs(scan_doc['scan_input'][4] - scan_doc['scan_input'][3])
                x_num = scan_doc['scan_input'][2]
                y_num = scan_doc['scan_input'][5]       
            # get x_range, y_range, dr_x, dr_y
            dr_x = 1.*x_range/x_num
            dr_y = 1.*y_range/y_num
            x_range = x_range
            y_range = y_range

            # get points
            scan_dim = scan_doc['shape']
            # scan_dim[1] -= 1
            num_frame = scan_dim[0]*scan_dim[1]

            if False: #header.start['plan_name'].startswith('pt'):
                points = np.zeros([2,num_frame])
                pos_tmp = np.zeros([1,num_frame],dtype=np.int64)
                pos_tmp = db.reg.retrieve(df['inenc1_val'].iat[0])
                points[0,:] = (pos_tmp-pos_tmp[0])*1e-4
                pos_tmp = db.reg.retrieve(df['inenc3_val'].iat[0])
                points[1,:] = (pos_tmp-pos_tmp[0])*6e-5
                if np.sum(np.abs(points[1,:]))<1e-5:
                    points[1,:] = np.linspace(scan_doc['scan_input'][3],scan_doc['scan_input'][4],num_frame)
                #points[:,:-1] = (points[:,:-1]+points[:,1:])/2
                if 'detector_distance' in header.start['scan']:
                        z_m = header.start['scan']['detector_distance']
                else:
                        z_m = 1.055
                angle = 0
            else:
                points = np.zeros([2,num_frame])
                pos_tmp = np.zeros([1,num_frame],dtype=np.int64)
                from hxntools.scan_info import get_scan_positions
                pos0,pos1 = get_scan_positions(header)
                points[0,:] = pos0
                points[1,:] = pos1
                #if scan_motors[0].endswith('ssx')
                #    pos_tmp = db.reg.retrieve(df['inenc2_val'].iat[0])
                #    pos_tmp = (pos_tmp-pos_tmp[0])*(-9.7e-5)
                #elif scan_motors[0] == 'zpssz':
                #    pos_tmp = - db.reg.retrieve(df['inenc4_val'].iat[0])
                #    pos_tmp = (pos_tmp-pos_tmp[0])*1.006e-4
                #points[0,:] = pos_tmp[:]
                #if scan_motors[1] == 'zpssy':
                #    pos_tmp = db.reg.retrieve(df['inenc4_val'].iat[0])
                #    pos_tmp = (pos_tmp-pos_tmp[0])*(-1.04e-4)
                #points[1,:] = pos_tmp[:]
                #for i in range(int(x_num)):
                #    points[0,i:int(x_num)*int(y_num):int(x_num)] = scan_doc['scan_input'][1] + (scan_doc['scan_input'][1] - scan_doc['scan_input'][0])/x_num*i
                #for i in range(int(y_num)):
                #    points[1,i*int(x_num):(i+1)*int(x_num)] = scan_doc['scan_input'][3] + (scan_doc['scan_input'][4] - scan_doc['scan_input'][3])/y_num*i
                #points[1,:] = np.linspace(scan_doc['scan_input'][3],scan_doc['scan_input'][4],num_frame)
                #points[:,:-1] = (points[:,:-1]+points[:,1:])/2
                #points[:,3:-4] = (points[:,:-7]+points[:,1:-6]+points[:,2:-5]+points[:,3:-4]+points[:,4:-3]+points[:,5:-2]+points[:,6:-1]+points[:,7:])/8
                if 'detector_distance' in header.start['scan']:
                        z_m = header.start['scan']['detector_distance']
                else:
                        z_m = 2.05
                try:
                    if scan_motors[0].startswith('zp'):
                        angle = bl.zpsth[1]
                    elif scan_motors[0].startswith('d'):
                        angle = bl.dsth[1]
                    else:
                        angle = 0
                except:
                    angle = 0
            ic = np.zeros(num_frame)
            ic[:] = np.resize(db.reg.retrieve(df[ic_chan].iat[0]),num_frame)
            array_ensure_positive_elements(ic, name="scaler")

            # get ccd_pixel_um
            ccd_pixel_um = 75.


            # get diffamp dimensions (uncropped!)
            nz = num_frame
            mds_table = df[det_name]

            # get nx and ny by looking at the first image
            # img = db.reg.retrieve(mds_table.iat[0])[0]
            # nx, ny = img.shape # can also give a ValueError; TODO: come up a better way!
        else:
            scan_type = header.start['plan_name']
            scan_doc = header.start['scan']
            scan_motors = [scan_doc['fast_axis']['motor_name'], scan_doc['slow_axis']['motor_name']]
            items = [det_name, 'sclr1_ch4', 'enc1', 'enc2']
            bl = db.get_table(header, stream_name='baseline')
            df = db.get_table(header, fields=items, fill=False)
            #images = db_old.get_images(db_old[sid], name=det_name)

            # get energy_kev
            dcm_th = bl.dcm_th[1]
            energy_kev = 12.39842 / (2.*3.1355893 * np.sin(dcm_th * np.pi / 180.))

            if scan_motors[0].endswith('ssy'):
                y_range = np.abs(scan_doc['scan_input'][1] - scan_doc['scan_input'][0])
                x_range = np.abs(scan_doc['scan_input'][4] - scan_doc['scan_input'][3])
                y_num = scan_doc['scan_input'][2]
                x_num = scan_doc['scan_input'][5]        
            else:
                x_range = np.abs(scan_doc['scan_input'][1] - scan_doc['scan_input'][0])
                y_range = np.abs(scan_doc['scan_input'][4] - scan_doc['scan_input'][3])
                x_num = scan_doc['scan_input'][2]
                y_num = scan_doc['scan_input'][5]       
            # get x_range, y_range, dr_x, dr_y
            dr_x = 1.*x_range/x_num
            dr_y = 1.*y_range/y_num
            x_range = x_range
            y_range = y_range

            # get points
            scan_dim = scan_doc['shape']
            # scan_dim[1] -= 1
            num_frame = scan_dim[0]*scan_dim[1]

            if df['enc1'].shape[0] == 1:
                points = np.zeros([2,num_frame])
                points[0,:] = db.reg.retrieve(df['enc1'].iat[0])
                points[1,:] = db.reg.retrieve(df['enc2'].iat[0])
                points[1,:] = np.linspace(scan_doc['scan_input'][3],scan_doc['scan_input'][4],num_frame)
                points[:,:-1] = (points[:,:-1]*1.2+points[:,1:]*0.8)/2
                ic = np.zeros(num_frame)
                ic[:] = db.reg.retrieve(df['sclr1_ch4'].iat[0])
                array_ensure_positive_elements(ic, name="scaler")
            else:
                points = np.zeros([2] + scan_dim)
                for i in range(scan_dim[1]):
                    points[0,:,i] = db.reg.retrieve(df['enc1'].iat[i])
                    points[1,:,i] = db.reg.retrieve(df['enc2'].iat[i])
                points = np.transpose(points,(0,2,1)).reshape((2,num_frame))
                #ic
                ic = np.zeros(scan_dim)
                for i in range(scan_dim[1]):
                    ic[:,i] = db.reg.retrieve(df['sclr1_ch4'].iat[i])
                ic = ic.T.reshape(num_frame)
                array_ensure_positive_elements(ic, name="scaler")
            
            # angle
            try:
                angle = bl.zpsth[1]
            except:
                angle = 0
            
            

            # get ccd_pixel_um
            ccd_pixel_um = 75.

            if 'detector_distance' in header.start['scan']:
                    z_m = header.start['scan']['detector_distance']
            else:
                    z_m = 1.055

            # get diffamp dimensions (uncropped!)
            nz = num_frame
            mds_table = df[det_name]

            # get nx and ny by looking at the first image
            # img = db.reg.retrieve(mds_table.iat[0])[0]
            # nx, ny = img.shape # can also give a ValueError; TODO: come up a better way!

    handler = db.reg.get_spec_handler(mds_table.iat[0].split('/')[0])
    if hasattr(handler,'_filename'):
        filename = handler._filename
    else:
        filename = handler._handle.filename

    with h5py.File(filename,'r') as f:
        shape = f['entry/data/data'].shape
        nx = shape[1]
        ny = shape[2]


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
    metadata['ic'] = ic
    metadata['ccd_pixel_um'] = ccd_pixel_um
    metadata['z_m'] = z_m
    metadata['nz'] = nz
    metadata['nx'] = nx
    metadata['ny'] = ny
    metadata['mds_table'] = mds_table

    return metadata


def save_data(db, param, scan_num:int, n:int, nn:int, cx:int, cy:int, threshold=0., bad_pixels=None, zero_out=None, upsample=1, save_diff = False):
    '''
    Save metadata and diffamp for the given scan number to a HDF5 file.

    Parameters:
        - db:
            a Broker instance.
        - param: Param
            a Param instance containing the metadata and other information from the GUI
        - scan_num: int
            the scan number
        - n: int
            the x dimension of the ROI window (=nx_prb)
        - nn: int
            the y dimension of the ROI window (=ny_prb)
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
    det_distance_m = param.z_m
    det_pixel_um = param.ccd_pixel_um/upsample
    num_frame = param.nz
    angle = param.angle
    lambda_nm = param.lambda_nm
    ic = param.ic
    #energy_kev = param.energy_kev

    #print('energy:', energy_kev)
    #print('angle: ', angle)
    #lambda_nm = 1.2398/energy_kev
    x_pixel_m = lambda_nm * 1.e-9 * det_distance_m / (n * det_pixel_um * 1e-6)
    y_pixel_m = lambda_nm * 1.e-9 * det_distance_m / (nn * det_pixel_um * 1e-6)
    x_depth_of_field_m = lambda_nm * 1.e-9 / (n/2 * det_pixel_um*1.e-6 / det_distance_m)**2
    y_depth_of_field_m = lambda_nm * 1.e-9 / (nn/2 * det_pixel_um*1.e-6 / det_distance_m)**2
    #print('pixel size: ', x_pixel_m, y_pixel_m)
    #print('depth of field: ', x_depth_of_field_m, y_depth_of_field_m)
    
    for i in range(num_frame):
        if ic[i]<0.1*np.mean(ic):
            ic[i]=ic[np.mod(i+1,num_frame)]

    if save_diff:
        # get data array
        data = np.zeros((num_frame, n//2*2, nn//2*2)) # nz*nx*ny
        mask = []
        nstack=1
        for i in range(num_frame):
            #print(param.mds_table.iat[i], file=sys.stderr)
            if i==0:
                img_stack = db.reg.retrieve(param.mds_table.iat[0])
                nstack = img_stack.shape[0]
            elif i%nstack == 0:
                if (num_frame - nstack)>3:
                    img_stack = db.reg.retrieve(param.mds_table.iat[i//nstack])
            img = img_stack[i%nstack]
            #img = np.rot90(img, axes=(1,0)) #equivalent to tt = np.flipud(tt).T
            ny, nx = np.shape(img)

            img = img * ic[0] / ic[i]

            if bad_pixels is not None:
                img = rm_outlier_pixels(img, bad_pixels[0], bad_pixels[1])

            if zero_out is not None:
                for blue_roi in zero_out:
                    x0 = blue_roi[0]
                    y0 = blue_roi[1]
                    w = blue_roi[2]
                    h = blue_roi[3]
                    img[y0:y0+h, x0:x0+w] = 0.

            if n < nx:
                # assuming n=nn???
                #print(nx, ny, file=sys.stderr)
                #print(cx-n//2, cx+n//2, cy-nn//2, cy+nn//2, file=sys.stderr)
                #tmptmp = img[cx-n//2:cx+n//2, cy-nn//2:cy+nn//2]
                tmptmp = img[cy-nn//2:cy+nn//2, cx-n//2:cx+n//2]
                #print(tmptmp.shape, file=sys.stderr)
            else:
                raise Exception("zero padding not completed yet")
                # # is this part necessary???
                # #tmptmp = t
                # tmptmp = np.zeros((n, n))
                # #tmptmp[3:-3,:] = t[:,cy-n//2:cy+n//2]
                # tmptmp[4:-8, :] = img[:, cy-n//2:cy+n//2]

            #if i == 0:
            #    import matplotlib.pyplot as plt
            #    plt.imshow(tmptmp, vmin=np.min(img), vmax=np.max(img))
            #    plt.savefig("ttttt.png")
            #    return

            tmptmp = np.rot90(tmptmp, axes=(1,0)) #equivalent to np.flipud(tmptmp).T
            if not np.sum(tmptmp) > 0.:
                mask.append(i)
            data[i] = np.fft.fftshift(tmptmp)

        if len(mask) > 0:
            print("Removing the dark frames:", mask, file=sys.stderr)
            data = np.delete(data, mask, axis=0)
            param.points = np.delete(param.points, mask, axis=1)
            param.nz = param.nz - len(mask)
        data[data < threshold] = 0.
        data = np.sqrt(data)

        if upsample != 1:
            if upsample%1 == 0:
                data = np.kron(data,np.ones((1,int(upsample),int(upsample))))
            elif upsample%0.5 == 0 and data.shape[1]%2 == 0:
                usample = upsample*2
                data = np.kron(data,np.ones((1,int(usample),int(usample))))
                data = data.reshape(data.shape[0],int(data.shape[1]/2),2,int(data.shape[2]/2),2)
                data = np.mean(data,axis=(2,4))
            else:
                raise NotImplementedError("Only upsampling ratios divisible by 0.5 are supported")

        # data array got
        print('array size:', np.shape(data))
    else:
        handler = db.reg.get_spec_handler(param.mds_table.iat[0].split('/')[0])
        if hasattr(handler,'_filename'):
            raw_data_filename = [handler._filename]
            raw_data_frame_counts = [1]
        else:
            raw_data_filename = [handler._handle.filename]
            raw_data_frame_counts = [1]
        for i in range(1,param.mds_table.size):
            handler = db.reg.get_spec_handler(param.mds_table.iat[i].split('/')[0])
            if hasattr(handler,'_filename'):
                filename = handler._filename
            else:
                filename = handler._handle.filename
            if raw_data_filename[-1] != filename:
                raw_data_filename.append(filename)
                raw_data_frame_counts.append(1)
            else:
                raw_data_frame_counts[-1] += 1
        raw_data_filename_abs = [os.path.realpath(filename) for filename in raw_data_filename]

        raw_data_roi = np.array([[cy-nn//2,cy+nn//2],[cx-n//2,cx+n//2]])        

        if bad_pixels is None:
            bad_pixels = []


    # create a folder
    try:
        os.mkdir(param.working_directory + '/h5_data/')
    except FileExistsError:
        pass

    file_path = param.working_directory + '/h5_data/scan_' + str(scan_num) + '.h5'
    with h5py.File(file_path, 'w') as hf:
        if save_diff:
            dset = hf.create_dataset('raw_data/flag',data=False)
            dset = hf.create_dataset('diffamp', data=data, compression='szip')
        else:
            dset = hf.create_dataset('raw_data/flag',data=True)
            dset = hf.create_dataset('raw_data/filename',data=raw_data_filename_abs)
            dset = hf.create_dataset('raw_data/framecounts',data=raw_data_frame_counts)
            dset = hf.create_dataset('raw_data/roi',data=raw_data_roi)
            dset = hf.create_dataset('raw_data/badpixels',data=bad_pixels)
            dset = hf.create_dataset('raw_data/threshold',data=threshold)
            dset = hf.create_dataset('raw_data/upsample',data=upsample)           
        dset = hf.create_dataset('points', data=param.points)
        dset = hf.create_dataset('x_range', data=param.x_range)
        dset = hf.create_dataset('y_range', data=param.y_range)
        dset = hf.create_dataset('dr_x', data=param.dr_x)
        dset = hf.create_dataset('dr_y', data=param.dr_y)
        dset = hf.create_dataset('z_m', data=det_distance_m)
        dset = hf.create_dataset('lambda_nm', data=lambda_nm)
        dset = hf.create_dataset('ccd_pixel_um', data=det_pixel_um)
        dset = hf.create_dataset('angle', data=angle)
        dset = hf.create_dataset('ic', data=ic)
        dset = hf.create_dataset('x_pixel_m', data=x_pixel_m)
        dset = hf.create_dataset('y_pixel_m', data=y_pixel_m)
        dset = hf.create_dataset('x_depth_field_m', data=x_depth_of_field_m)
        dset = hf.create_dataset('y_depth_field_m', data=y_depth_of_field_m)

    # symlink so ptycho can find it
    try:
        symlink_path = param.working_directory + '/scan_' + str(scan_num) + '.h5'
        os.symlink(os.path.realpath(file_path), symlink_path)
    except FileExistsError:
        os.remove(symlink_path)
        os.symlink(os.path.realpath(file_path), symlink_path)


def get_single_image(db, frame_num, mds_table):
    length = mds_table.size
    if length == 1:
        if frame_num == -1:
            img_raw = db.reg.retrieve(mds_table.iat[0])
        else:
            dataset = db.reg.get_spec_handler(mds_table.iat[0].split('/')[0]).dataset()
            nframes = dataset.shape[0]
            if frame_num >= nframes:
                message = "[ERROR] The {0}-th frame doesn't exist. "
                message += "Available frames for the chosen scan: [0, {1}]."
                raise ValueError(message.format(frame_num, nframes-1))
            img_raw = np.array((dataset[frame_num],),dtype=dataset.dtype)
    elif frame_num >= length:
        message = "[ERROR] The {0}-th frame doesn't exist. "
        message += "Available frames for the chosen scan: [0, {1}]."
        raise ValueError(message.format(frame_num, length-1))
    elif frame_num == -1:
        img_single = db.reg.retrieve(mds_table.iat[0])
        img_raw = np.zeros((length,)+img_single.shape[1:],dtype = img_single.dtype)
        img_raw[0] = img_single[0]
        for i in range(1,length):
            img_raw[i] = db.reg.retrieve(mds_table.iat[i])[0]
    else:
        img_raw = db.reg.retrieve(mds_table.iat[frame_num])
    overflow_value = np.iinfo(img_raw.dtype).max
    img = np.mean(img_raw,axis=0)
    return img,overflow_value


def get_detector_names(db, scan_num:int):
    '''
    Returns
    -------
        list: detectors used in the scan or available in the beamline
    '''
    # TODO: a better way without ScanInfo?
    scan = ScanInfo(db[scan_num])
    return [key for key in scan.filestore_keys]
