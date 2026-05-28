# -*- coding: utf-8 -*-
import re
import sys
from nsls2ptycho.remote_worker import main
from nsls2ptycho._version import __version__
print(f"Ptycho reconstruction worker for version {__version__}")
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
