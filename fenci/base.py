#!/usr/bin/env python
# -*-coding:utf-8-*-

from abc import ABC, abstractmethod
import os
import tempfile


class BaseSegment(ABC):

    @abstractmethod
    def initialize(self):
        pass

    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def _get_cache_file(self):
        cache_file = os.path.join(self.tmp_dir or tempfile.gettempdir(),
                                  self.cache_file)
        self.tmp_dir = os.path.dirname(cache_file)
        return cache_file
