#!/usr/bin/env python
# -*-coding:utf-8-*-
import shutil
from abc import ABC, abstractmethod
import os
import tempfile
import logging

from .const import DEFAULT_MODEL_NAME, DEFAULT_SIMPLE_MODEL
from . import __softname__
from .utils import get_resource_path

logger = logging.getLogger(__name__)


class BaseSegment(ABC):
    tmp_dir = None

    @abstractmethod
    def initialize(self):
        pass

    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def _get_model_file(self):
        model_file = os.path.join(self.tmp_dir or tempfile.gettempdir(), self.model_file)
        self.tmp_dir = os.path.dirname(model_file)
        return model_file

    def _copy_default_model_file(self):
        model_file = self._get_model_file()
        default_model_file = get_resource_path(__softname__, DEFAULT_MODEL_NAME)
        shutil.copyfile(default_model_file, model_file)

    def _copy_model_simple_file(self):
        model_file = self._get_model_file()
        default_model_simple_file = get_resource_path(__softname__, DEFAULT_SIMPLE_MODEL)
        shutil.copyfile(default_model_simple_file, model_file)

    def reset_model(self, model='default'):
        """
        将模型文件重设为默认值
        """
        if model == 'default':
            self._copy_default_model_file()
        elif model == 'simple':
            self._copy_model_simple_file()
        else:
            logger.warning("Wrong model type given, default or simple.")
