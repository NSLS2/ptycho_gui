from databroker.v0 import Broker
from databroker.headersource.mongo import MDS
from databroker.assets.mongo import Registry
from databroker.headersource.core import doc_or_uid_to_uid
from databroker.assets.handlers import HandlerBase
from jsonschema import validate as js_validate
from bluesky_kafka import Publisher
from collections import deque

import numpy as np
import pandas as pd
import warnings
import uuid
import certifi
import os
import six
import pymongo
import h5py
from datetime import datetime

import logging

# Load data from Eiger
db1_name = 'rs'
db1_addr = 'mongodb://xf03id1-mdb01:27017,xf03id1-mdb02:27017,xf03id1-mdb03:27017'
bootstrap_servers = os.getenv("BLUESKY_KAFKA_BOOTSTRAP_SERVERS", None)
kafka_password = '^!Y45VFy&AtBv&^w'
if kafka_password is None:
    msg = "The 'BLUESKY_KAFKA_PASSWORD' environment variable must be set."
    raise RuntimeError(msg)

_mds_config_db1 = {'host': db1_addr,
                   'port': 27017,
                   'database': 'datastore-2',
                   'timezone': 'US/Eastern'}

_fs_config_db1 = {'host': db1_addr,
                  'port': 27017,
                  'database': 'filestore-2'}
kafka_publisher = Publisher(
        topic="hxn.bluesky.datum.documents",
        bootstrap_servers=bootstrap_servers,
        key=str(uuid.uuid4()),
        producer_config={
                "acks": 1,
                "message.timeout.ms": 3000,
                "queue.buffering.max.kbytes": 10 * 1048576,
                "compression.codec": "snappy",
                "ssl.ca.location": certifi.where(),
                "security.protocol": "SASL_SSL",
                "sasl.mechanisms": "SCRAM-SHA-512",
                "sasl.username": "beamline",
                "sasl.password": kafka_password,
                },
        flush_on_stop_doc=True,
    ) if not os.environ.get('AZURE_TESTING') else None   # Disable on CI

#f_benchmark = open("/tmp/benchmark.out", "a+")
datum_counts = {}

def sanitize_np(val):
    "Convert any numpy objects into built-in Python types."
    if isinstance(val, (np.generic, np.ndarray)):
        if np.isscalar(val):
            return val.item()
        return val.tolist()
    return val

def apply_to_dict_recursively(d, f):
    for key, val in d.items():
        if hasattr(val, 'items'):
            d[key] = apply_to_dict_recursively(val, f)
        d[key] = f(val)

class CompositeRegistry(Registry):
    '''Composite registry.'''

    def _register_resource(self, col, uid, spec, root, rpath, rkwargs,
                              path_semantics):

        run_start=None
        ignore_duplicate_error=False
        duplicate_exc=None

        if root is None:
            root = ''

        resource_kwargs = dict(rkwargs)
        if spec in self.known_spec:
            js_validate(resource_kwargs, self.known_spec[spec]['resource'])

        resource_object = dict(spec=str(spec),
                               resource_path=str(rpath),
                               root=str(root),
                               resource_kwargs=resource_kwargs,
                               path_semantics=path_semantics,
                               uid=uid)

        try:
            col.insert_one(resource_object)
        except duplicate_exc:
            if ignore_duplicate_error:
                warnings.warn("Ignoring attempt to insert Datum with duplicate "
                          "datum_id, assuming that both ophyd and bluesky "
                          "attempted to insert this document. Remove the "
                          "Registry (`reg` parameter) from your ophyd "
                          "instance to remove this warning.")
            else:
                raise

        resource_object['id'] = resource_object['uid']
        resource_object.pop('_id', None)
        ret = resource_object['uid']

        return ret

    def register_resource(self, spec, root, rpath, rkwargs,
                              path_semantics='posix'):

        uid = str(uuid.uuid4())
        datum_counts[uid] = 0
        method_name = "register_resource"
        col = self._resource_col
        ret = self._register_resource(col, uid, spec, root, rpath,
                                      rkwargs, path_semantics=path_semantics)

        return ret

    def _insert_datum(self, col, resource, datum_id, datum_kwargs, known_spec,
                     resource_col, ignore_duplicate_error=False,
                     duplicate_exc=None):
        if ignore_duplicate_error:
            assert duplicate_exc is not None
        if duplicate_exc is None:
            class _PrivateException(Exception):
                pass
            duplicate_exc = _PrivateException
        try:
            resource['spec']
            spec = resource['spec']

            if spec in known_spec:
                js_validate(datum_kwargs, known_spec[spec]['datum'])
        except (AttributeError, TypeError):
            pass
        resource_uid = self._doc_or_uid_to_uid(resource)
        if type(datum_kwargs) == str and '/' in datum_kwargs:
            datum_kwargs = {'point_number': datum_kwargs.split('/')[-1]}

        datum = dict(resource=resource_uid,
                     datum_id=str(datum_id),
                     datum_kwargs=dict(datum_kwargs))
        apply_to_dict_recursively(datum, sanitize_np)
        # We are transitioning from ophyd objects inserting directly into a
        # Registry to ophyd objects passing documents to the RunEngine which in
        # turn inserts them into a Registry. During the transition period, we allow
        # an ophyd object to attempt BOTH so that configuration files are
        # compatible with both the new model and the old model. Thus, we need to
        # ignore the second attempt to insert.
        try:
            kafka_publisher('datum', datum)
            #col.insert_one(datum)
        except duplicate_exc:
            if ignore_duplicate_error:
                warnings.warn("Ignoring attempt to insert Resource with duplicate "
                              "uid, assuming that both ophyd and bluesky "
                              "attempted to insert this document. Remove the "
                              "Registry (`reg` parameter) from your ophyd "
                              "instance to remove this warning.")
            else:
                raise
        # do not leak mongo objectID
        datum.pop('_id', None)

        return datum


    def register_datum(self, resource_uid, datum_kwargs, validate=False):

        if validate:
            raise RuntimeError('validate not implemented yet')

        res_uid = resource_uid
        datum_count = datum_counts[res_uid]

        datum_uid = res_uid + '/' + str(datum_count)
        datum_counts[res_uid] = datum_count + 1

        col = self._datum_col
        datum = self._insert_datum(col, resource_uid, datum_uid, datum_kwargs, {}, None)
        ret = datum['datum_id']

        return ret

    def _doc_or_uid_to_uid(self, doc_or_uid):

        if not isinstance(doc_or_uid, six.string_types):
            try:
                doc_or_uid = doc_or_uid['uid']
            except TypeError:
                pass

        return doc_or_uid

    def _bulk_insert_datum(self, col, resource, datum_ids,
                           datum_kwarg_list):

        resource_id = self._doc_or_uid_to_uid(resource)

        to_write = []

        d_uids = deque()

        for d_id, d_kwargs in zip(datum_ids, datum_kwarg_list):
            dm = dict(resource=resource_id,
                      datum_id=str(d_id),
                      datum_kwargs=dict(d_kwargs))
            apply_to_dict_recursively(dm, sanitize_np)
            to_write.append(pymongo.InsertOne(dm))
            d_uids.append(dm['datum_id'])

        col.bulk_write(to_write, ordered=False)

        return d_uids

    def bulk_register_datum_table(self, resource_uid, dkwargs_table, validate=False):

        res_uid = resource_uid['uid']
        datum_count = datum_counts[res_uid]

        if validate:
            raise RuntimeError('validate not implemented yet')

        d_ids = [res_uid + '/' + str(datum_count+j) for j in range(len(dkwargs_table))]
        datum_counts[res_uid] = datum_count + len(dkwargs_table)

        dkwargs_table = pd.DataFrame(dkwargs_table)
        datum_kwarg_list = [ dict(r) for _, r in dkwargs_table.iterrows()]

        method_name = "bulk_register_datum_table"

        self._bulk_insert_datum(self._datum_col, resource_uid, d_ids, datum_kwarg_list)
        return d_ids
class CompositeBroker(Broker):
    """wrapper for two databases"""

    # databroker.headersource.MDSROTemplate
    def _bulk_insert_events(self, event_col, descriptor, events, validate, ts):

        descriptor_uid = doc_or_uid_to_uid(descriptor)

        to_write = []
        for ev in events:
            data = dict(ev['data'])

            # Replace any filled data with the datum_id stashed in 'filled'.
            for k, v in six.iteritems(ev.get('filled', {})):
                if v:
                    data[k] = v
            # Convert any numpy types to native Python types.
            apply_to_dict_recursively(data, sanitize_np)
            timestamps = dict(ev['timestamps'])
            apply_to_dict_recursively(timestamps, sanitize_np)

            # check keys, this could be expensive
            if validate:
                if data.keys() != timestamps.keys():
                    raise ValueError(
                        BAD_KEYS_FMT.format(data.keys(),
                                            timestamps.keys()))
            ev_uid = ts + '-' + ev['uid']

            ev_out = dict(descriptor=descriptor_uid, uid=ev_uid,
                          data=data, timestamps=timestamps,
                          time=ev['time'],
                          seq_num=ev['seq_num'])

            to_write.append(pymongo.InsertOne(ev_out))

        event_col.bulk_write(to_write, ordered=True)

    # databroker.headersource.MDSROTemplate
    # databroker.headersource.MDSRO(MDSROTemplate)
    def _insert(self, name, doc, event_col, ts):
        for desc_uid, events in doc.items():
            # If events is empty, mongo chokes.
            if not events:
                continue
            self._bulk_insert_events(event_col,
                                     descriptor=desc_uid,
                                     events=events,
                                     validate=False, ts=ts)


    def insert(self, name, doc):

        if name == "start":
            #f_benchmark.write("\n scan_id: {} \n".format(doc['scan_id']))
            #f_benchmark.flush()
            datum_counts = {}

        ts =  str(datetime.now().timestamp())

        if name in {'bulk_events'}:
            ret2 = self._insert(name, doc, db1.mds._event_col, ts)
        else:
            ret2 = db1.insert(name, doc)
        return ret2
    
class SISHDF5Handler(HandlerBase):
    HANDLER_NAME = "SIS_HDF51_FLY_STREAM_V1"

    def __init__(self, resource_fn, *, frame_per_point):
        self._frame_per_point = frame_per_point
        self._handle = h5py.File(resource_fn, "r", libver='latest', swmr=True)

    def __call__(self, *, column, point_number):
        n_first = point_number * self._frame_per_point
        n_last = n_first + self._frame_per_point
        ds = self._handle[column]
        ds.id.refresh()
        return ds[n_first:n_last]

    # def close(self):
    #     self._handle.close()
    #     self._handle = None
    #     super().close()
class BulkMerlin(HandlerBase):
    HANDLER_NAME = 'MERLIN_FLY_STREAM_V2'

    def __init__(self, resource_fn, *, frame_per_point):
        self._frame_per_point = frame_per_point
        self._handle = h5py.File(resource_fn, "r", libver='latest', swmr=True)

    def __call__(self, point_number):
        n_first = point_number * self._frame_per_point
        n_last = n_first + self._frame_per_point
        ds = self._handle['entry/instrument/detector/data']
        ds.id.refresh()
        return ds[n_first:n_last, :, :]

class ZebraHDF5Handler(HandlerBase):
    HANDLER_NAME = "ZEBRA_HDF51_FLY_STREAM_V1"

    def __init__(self, resource_fn, *, frame_per_point):
        self._frame_per_point = frame_per_point
        self._handle = h5py.File(resource_fn, "r", libver='latest', swmr=True)

    def __call__(self, *, column, point_number):
        n_first = point_number * self._frame_per_point
        n_last = n_first + self._frame_per_point
        ds = self._handle[column]
        ds.id.refresh()
        return ds[n_first:n_last]
    
class PandAHandlerHDF5(HandlerBase):
    """The handler to read HDF5 files produced by PandABox."""

    specs = {"PANDA"}

    def __init__(self, filename):
        self._name = filename

    def __call__(self, field):
        with h5py.File(self._name, "r") as f:
            entry = f[f"/{field}"]
            return entry[:]


# imort XRF_DATA_KEY for back-compat
from databroker.assets.handlers import (Xspress3HDF5Handler,
                                        XS3_XRF_DATA_KEY as XRF_DATA_KEY)


logger = logging.getLogger(__name__)

FMT_ROI_KEY = 'entry/instrument/detector/NDAttributes/CHAN{}ROI{}'
from databroker.assets.handlers import HandlerBase, ImageStack


class HDF5DatasetSliceHandler(HandlerBase):
    """
    Handler for data stored in one Dataset of an HDF5 file.
    Parameters
    ----------
    filename : string
        path to HDF5 file
    key : string
        key of the single HDF5 Dataset used by this Handler
    frame_per_point : integer, optional
        number of frames to return as one datum, default 1
    swmr : bool, optional
        Open the hdf5 file in SWMR read mode. Only used when mode = 'r'.
        Default is False.
    """
    def __init__(self, filename, key, frame_per_point=1):
        self._fpp = frame_per_point
        self._filename = filename
        self._key = key
        self._file = None
        self._dataset = None
        self._data_objects = {}
        self.open()

    def get_file_list(self, datum_kwarg_gen):
        return [self._filename]

    def __call__(self, point_number):
        # Don't read out the dataset until it is requested for the first time.
        if not self._dataset:
            self._dataset = self._file[self._key]

        if point_number not in self._data_objects:
            start = point_number * self._fpp
            stop = (point_number + 1) * self._fpp
            self._data_objects[point_number] = ImageStack(self._dataset,
                                                          start, stop)
        return self._data_objects[point_number]

    def open(self):
        import h5py
        if self._file:
            return

        self._file = h5py.File(self._filename, 'r')

    def close(self):
        super(HDF5DatasetSliceHandler, self).close()
        self._file.close()
        self._file = None


class TimepixHDF5Handler(HDF5DatasetSliceHandler):
    """
    Handler for the 'AD_HDF5' spec used by Area Detectors.
    In this spec, the key (i.e., HDF5 dataset path) is always
    '/entry/detector/data'.
    Parameters
    ----------
    filename : string
        path to HDF5 file
    frame_per_point : integer, optional
        number of frames to return as one datum, default 1
    """
    _handler_name = 'TPX_HDF5'
    specs = {_handler_name}

    # TODO this is only different due to the hardcoded key being different?
    hardcoded_key = '/entry/instrument/detector/data'

    def __init__(self, filename, frame_per_point=1):
        super(TimepixHDF5Handler, self).__init__(
                filename=filename, key=self.hardcoded_key,
                frame_per_point=frame_per_point)

class ROIHDF5Handler(HandlerBase):
    HANDLER_NAME = "ROI_HDF51_FLY"

    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, "r", libver='latest', swmr=True)

    def __call__(self, *, det_elem):
        ds = self._handle[det_elem]
        ds.id.refresh()
        return ds[:]

    # def close(self):
    #     self._handle.close()
    #     self._handle = None
    #     super().close()




mds_db1 = MDS(_mds_config_db1, auth=False)
db1 = Broker(mds_db1, CompositeRegistry(_fs_config_db1))
db = CompositeBroker(mds_db1, CompositeRegistry(_fs_config_db1))

db.reg.register_handler(SISHDF5Handler.HANDLER_NAME, SISHDF5Handler, overwrite=True)
db.reg.register_handler(BulkMerlin.HANDLER_NAME, BulkMerlin,  overwrite=True)
db.reg.register_handler(ZebraHDF5Handler.HANDLER_NAME, ZebraHDF5Handler, overwrite=True)
db.reg.register_handler("PANDA", PandAHandlerHDF5, overwrite=True)
db.reg.register_handler(Xspress3HDF5Handler.HANDLER_NAME,
                            Xspress3HDF5Handler, overwrite=True)
db.reg.register_handler(TimepixHDF5Handler._handler_name,
                            TimepixHDF5Handler, overwrite=True)
db.reg.register_handler(ROIHDF5Handler.HANDLER_NAME, ROIHDF5Handler, overwrite=True)

try:
    from hxntools.handlers import register
    register(db)
except:
    pass
