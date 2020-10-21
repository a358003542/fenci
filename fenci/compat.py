#!/usr/bin/env python
# -*-coding:utf-8-*-

import os

if os.name == 'nt':
    from shutil import move as _replace_file
else:
    _replace_file = os.rename
