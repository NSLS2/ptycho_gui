import logging
import collections
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def _eval(scan_args):
    '''Evaluate scan arguments, replacing OphydObjects with NamedObjects'''

    class NamedObject:
        def __init__(self, name):
            self.name = name

    def no_op():
        def no_op_inner(*args, name=None, **kwargs):
            if name is not None:
                return NamedObject(name)

        return no_op_inner

    return eval(scan_args, collections.defaultdict(no_op))


scan_types = dict(
    v0=dict(step_1d=('InnerProductAbsScan', 'HxnInnerAbsScan',
                     'InnerProductDeltaScan', 'HxnInnerDeltaScan', 'AbsScan',
                     'HxnAbsScan', 'DeltaScan', 'HxnDeltaScan'),
            step_2d=('OuterProductAbsScan', 'HxnOuterAbsScan', 'relative_mesh',
                     'absolute_mesh'),
            spiral=('HxnFermatPlan', 'relative_fermat', 'absolute_fermat',
                    'relative_spiral', 'absolute_spiral'),
            fly=('FlyPlan1D', 'FlyPlan2D'),
            ),
    v1=dict(fly=('FlyPlan1D', 'FlyPlan2D', 'FlyStep1D'),
            # everything else can be done with plan_patterns
            ),
    v2=dict(fly=('generator'),
            ),
)


def get_scan_info(header):
    start_doc = header['start']
    if 'scan_args' in start_doc:
        return _get_scan_info_bs_v0(header)
    elif 'plan_args' in start_doc:
        return _get_scan_info_bs_v1(header)
    elif 'scan' in start_doc:
        return _get_scan_info_bs_v2(header)
    else:
        raise RuntimeError('Unknown start document information')


def _get_scan_info_bs_v0(header):
    info = {'num': 0,
            'dimensions': [],
            'motors': [],
            'range': [],
            'pyramid': False,
            }

    start_doc = header['start']
    try:
        scan_args = start_doc['scan_args']
    except KeyError:
        try:
            scan_args = start_doc['plan_args']
        except KeyError:
            logger.error('No scan args for scan %s', start_doc['uid'])
            return info

    try:
        scan_type = start_doc['scan_type']
    except KeyError:
        try:
            scan_type = start_doc['plan_type']
        except KeyError:
            logger.error('No plan type for scan %s', start_doc['uid'])
            return info

    motors = None
    range_ = None
    pyramid = False
    motor_keys = None
    exposure_time = 0.0
    dimensions = []

    scan_type_info = scan_types['v0']
    step_1d = scan_type_info['step_1d']
    step_2d = scan_type_info['step_2d']
    spiral_scans = scan_type_info['spiral']
    fly_scans = scan_type_info['fly']

    if scan_type in fly_scans:
        dimensions = start_doc['dimensions']
        try:
            motors = start_doc['motors']
        except KeyError:
            motors = start_doc['axes']

        pyramid = start_doc['fly_type'] == 'pyramid'
        exposure_time = float(scan_args.get('exposure_time', 0.0))

        logger.debug('Scan %s (%s) is a fly-scan (%s) of axes %s '
                     'with per-frame exposure time of %.3f s',
                     start_doc['scan_id'], start_doc['uid'], scan_type,
                     motors, exposure_time)
        try:
            range_ = start_doc['scan_range']
        except KeyError:
            try:
                range_ = [(float(start_doc['scan_start']),
                           float(start_doc['scan_end']))]
            except (KeyError, ValueError):
                pass
    elif scan_type in step_2d:
        logger.debug('Scan %s (%s) is an ND scan (%s)', start_doc['scan_id'],
                     start_doc['uid'], scan_type)

        try:
            args = _eval(scan_args['args'])
        except Exception:
            pass

        # 2D mesh scan
        try:
            motors = [arg.name for arg in args[::5]]
        except Exception:
            motors = []

        exposure_time = float(start_doc.get('exposure_time', 0.0))
        try:
            dimensions = args[3::5]
            range0 = args[1::5]
            range1 = args[2::5]
            range_ = list(zip(range0, range1))
        except Exception:
            dimensions = []
            range_ = []

    elif scan_type in spiral_scans:
        motor_keys = ['x_motor', 'y_motor']
        dimensions = [int(start_doc['num'])]
        exposure_time = float(scan_args.get('exposure_time', 0.0))
        logger.debug('Scan %s (%s) is a fermat scan (%s) %d points, '
                     'with per-point exposure time of %.3f s',
                     start_doc['scan_id'], start_doc['uid'], scan_type,
                     dimensions[0], exposure_time)
        try:
            range_ = [(float(start_doc['x_range']),
                       float(start_doc['y_range']))]
        except (KeyError, ValueError):
            pass

    elif scan_type in step_1d or 'num' in start_doc:
        logger.debug('Scan %s (%s) is a 1D scan (%s)', start_doc['scan_id'],
                     start_doc['uid'], scan_type)
        exposure_time = float(start_doc.get('exposure_time', 0.0))
        # 1D scans
        try:
            dimensions = [int(start_doc['num'])]
        except KeyError:
            # some scans with the bluesky md changes didn't save num
            dimensions = []
        motor_keys = ['motor']
    else:
        msg = 'Unrecognized scan type (uid={} {})'.format(start_doc['uid'],
                                                          scan_type)
        raise RuntimeError(msg)

    if motor_keys:
        motors = []
        for key in motor_keys:
            try:
                motors.append(_eval(start_doc[key]).name)
            except Exception:
                pass

    num = np.product(dimensions)

    info['num'] = num
    info['dimensions'] = dimensions
    info['motors'] = motors
    info['range'] = range_
    info['pyramid'] = pyramid
    info['exposure_time'] = exposure_time
    return info

def _get_scan_info_bs_v1(header):
    start_doc = header['start']
    info = {'num': 0,
            'dimensions': [],
            'motors': [],
            'range': {},
            'pyramid': False,
            }

    plan_type = start_doc['plan_type']
    plan_name = start_doc['plan_name']

    range_ = None
    pyramid = False

    plan_type_info = scan_types['v1']
    fly_scans = plan_type_info['fly']

    motors = start_doc['motors']

    if plan_type in fly_scans:
        logger.debug('Scan %s (%s) is a fly scan (%s %s)', start_doc['scan_id'],
                     start_doc['uid'], plan_type, plan_name)
        shape = start_doc['shape']
        pyramid = start_doc['fly_type'] == 'pyramid'
        range_ = dict(zip(motors, start_doc['scan_range']))
    else:
        try:
            pattern_module = start_doc['plan_pattern_module']
            pattern = start_doc['plan_pattern']
            pattern_args = start_doc['plan_pattern_args']
        except KeyError as ex:
            msg = ('Unrecognized plan type/name (uid={} name={} type={}; '
                   'missing key: {!r})'.format(start_doc['uid'], plan_name,
                                               plan_type, ex))
            raise RuntimeError(msg)

        module = __import__(pattern_module, fromlist=[''])
        pattern_fcn = getattr(module, pattern)
        logger.debug('Calling plan pattern function: %s with kwargs %s',
                     pattern_fcn, pattern_args)
        points = pattern_fcn(**pattern_args)

        if isinstance(points, (list, np.ndarray)):
            num = len(points)
            range_ = {motors[0]: (np.min(points), np.max(points))}
        else:
            cyc = points
            num = len(cyc)
            try:
                # cycler >= 0.10
                positions = cyc.by_key()
            except AttributeError:
                positions = collections.defaultdict(lambda: [])
                for pt in iter(cyc):
                    for mtr, pos in pt.items():
                        positions[mtr].append(pos)

            range_ = {_eval(mtr).name: (np.min(positions[mtr]),
                                        np.max(positions[mtr]))
                      for mtr in cyc.keys}

        shape = start_doc.get('shape', None)
        if shape is None:
            shape = [num]

    info['num'] = np.product(shape)
    info['dimensions'] = shape
    info['motors'] = motors
    info['range'] = range_
    info['pyramid'] = pyramid
    return info

def _get_scan_info_bs_v2(header):
    start_doc = header['start']
    info = {'num': 0,
            'dimensions': [],
            'motors': [],
            'range': {},
            'pyramid': False,
            }

    plan_type = start_doc['plan_type']
    plan_name = start_doc['plan_name']

    range_ = None
    pyramid = False

    plan_type_info = scan_types['v2']
    fly_scans = plan_type_info['fly']

    scan_doc = start_doc['scan']
    motors = [scan_doc['fast_axis']['motor_name'], scan_doc['slow_axis']['motor_name']]

    if plan_type in fly_scans:
        logger.debug('Scan %s (%s) is a fly scan (%s %s)', start_doc['scan_id'],
                     start_doc['uid'], plan_type, plan_name)
        shape = scan_doc['shape']
        range_ = dict(zip(motors, [scan_doc['scan_input'][0:2],scan_doc['scan_input'][3:5]]))

    info['num'] = np.product(shape)
    info['dimensions'] = shape
    info['motors'] = motors
    info['range'] = range_
    info['pyramid'] = pyramid
    return info

class ScanInfo(object):
    def __init__(self, header, stream='primary'):
        self.header = header
        self.start_doc = header['start']
        self.descriptors = header['descriptors']
        self.key = None

        if 'scan_args' in self.start_doc and stream == 'primary':
            stream = None

        self.stream = stream
        for key, value in get_scan_info(self.header).items():
            logger.debug('Scan info %s=%s', key, value)
            setattr(self, key, value)

    @property
    def filestore_keys(self):
        for desc in self.descriptors:
            for key, info in desc['data_keys'].items():
                try:
                    external = info['external']
                except KeyError:
                    continue

                try:
                    source, info = external.split(':', 1)
                except Exception:
                    pass
                else:
                    source = source.lower()
                    if source in ('filestore', ):
                        yield key

    @property
    def scan_id(self):
        return self.start_doc['scan_id']

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.header)

    def iterate_over_stream(self, stream_name, fill=False, key=None, **kwargs):
        if key is None:
            key = self.key

        if key:
            for event in self.header.events(fill=False,
                                            stream_name=stream_name):
                yield event['data'][key]

    def __iter__(self):
        yield from self.iterate_over_stream(self.stream, fill=False)


def combine_tables_on_time(header, names, *, method='ffill', **kwargs):
    '''Combine a fast-changing dataframe from databroker with one (or more)
    slow-changing ones, given its header.

    Parameters
    ----------
    header : Header
    names : list
        List of broker event stream names. The first will be taken as the
        primary (and in fact should probably be 'primary')
    method : {'ffill', 'bfill', 'nearest'}, optional
        Reindexing method
    **kwargs : dict, optional
        Passed to metadatastore.get_table

    Returns
    -------
    df : pd.DataFrame
    '''
    dfs = [header.table(stream_name=name, **kwargs)
           for name in names]

    primary_df = dfs[0]
    primary_index = primary_df.index

    try:
        times = primary_df['time']
    except KeyError:
        return dfs[0]

    dfs = ([primary_df] +
           [other.set_index('time').reindex(times, method=method)
            for other in dfs[1:]
            if 'time' in other])

    for df in dfs[1:]:
        df.index = primary_index
    return pd.concat(dfs, axis=1)


def get_combined_table(headers, name='primary',
                       combine_table_names=None,
                       **kwargs):
    # Functions the same as get_table, but also combines ('primary' and
    # 'motor2') when necessary (or whatever is in combine_table_names)

    if combine_table_names is None:
        if name == 'primary':
            combine_table_names = ['primary', 'motor2']
        else:
            combine_table_names = []

    if not combine_table_names:
        return [h.table(stream_name=name, **kwargs)
                for h in headers]

    try:
        headers.items()
    except AttributeError:
        pass
    else:
        headers = [headers]

    dfs = [combine_tables_on_time(header, names=combine_table_names, **kwargs)
           for header in headers]

    if dfs:
        return pd.concat(dfs)
    else:
        # edge case: no data
        return pd.DataFrame()
