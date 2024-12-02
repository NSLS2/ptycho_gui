import json
import os, sys
import platform


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

load_metadata = None
save_data = None
get_single_image = None
get_detector_names = None
db = None


def _load_HXN():
    from . import HXN_databroker as hxn_databroker
    global load_metadata, save_data, get_single_image, get_detector_names, db
    load_metadata = hxn_databroker.load_metadata
    save_data = hxn_databroker.save_data
    get_single_image = hxn_databroker.get_single_image
    get_detector_names = hxn_databroker.get_detector_names
    db = hxn_databroker.hxn_db
    print("HXN's Databroker is enabled.", file=sys.stderr)


def _load_CSX():
    from . import CSX_databroker as csx_databroker
    global load_metadata, save_data, get_single_image, get_detector_names, db
    load_metadata = csx_databroker.load_metadata
    save_data = csx_databroker.save_data
    get_single_image = csx_databroker.get_single_image
    get_detector_names = csx_databroker.get_detector_names
    db = csx_databroker.csx_db
    print("CSX's Databroker is enabled.", file=sys.stderr)


try:
    hostname = platform.node()
    config_path = os.path.expanduser("~") + "/.ptycho_gui/nsls2ptycho.json"

    if hostname.startswith('xf03id'):
        beamline_name = 'HXN'
    elif hostname.startswith('xf23id'):
        beamline_name = 'CSX'
    elif os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            beamline_config = json.load(f)
            beamline_name = beamline_config['beamline_name']
    else:
        beamline_name = None

    if beamline_name == 'HXN':
        _load_HXN()
    elif beamline_name == 'CSX':
        _load_CSX()
    else:
        raise RuntimeError("[WARNING] Cannot detect the beamline name. Databroker is disabled.")
except ImportError:
    print("[WARNING] Databroker is not found and so disabled.", file=sys.stderr)
except RuntimeError as ex:
    print(ex, file=sys.stderr)


del config_path, hostname, json, os, platform, sys
