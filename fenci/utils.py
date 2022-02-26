#!/usr/bin/env python
# -*-coding:utf-8-*-

import os
import json
import logging
import re
from pathlib import Path
import tempfile
import shutil

logger = logging.getLogger(__name__)


def normalized_path(path='.') -> str:
    """
    默认支持 ~ 符号

    返回的是字符串

    which default support the `~`
    """
    if isinstance(path, Path):
        return str(path.expanduser())
    elif isinstance(path, str):
        if path.startswith('~'):
            path = os.path.expanduser(path)
        return path
    else:
        raise TypeError


def strdecode(sentence):
    """
    字符串decode 只尝试utf8，和 gbk 否则将抛出异常
    :param sentence:
    :return:
    """
    if not isinstance(sentence, str):
        try:
            sentence = sentence.decode('utf-8')
        except UnicodeDecodeError:
            try:
                sentence = sentence.decode('gbk')
            except UnicodeDecodeError:
                logger.error('I have tried utf8 and gbk encoding, all failed.')
                raise UnicodeDecodeError
    return sentence


def write_json(file, data):
    """
    采用更稳妥的写文件方式，先在另外一个临时文件里面写，确保写操作无误之后再更改文件名
    """
    fp = tempfile.NamedTemporaryFile(mode='wt', encoding='utf8', delete=False)
    try:
        json.dump(data, fp, indent=4, ensure_ascii=False)
        fp.close()
    except Exception as e:
        logger.error(f"write data to tempfile {fp.name} failed!!! \n"
                     f"{e}")
    finally:
        shutil.move(fp.name, file)


def get_json_file(json_filename, default_data=None):
    """
    try get json file or auto-create the json file with default_data
    """
    default_data = default_data if default_data is not None else {}

    if not os.path.exists(json_filename):
        data = default_data
        write_json(json_filename, data)

    return json_filename


def get_json_data(json_filename, default_data=None):
    """
    get json file data
    """
    with open(get_json_file(json_filename, default_data=default_data),
              encoding='utf8') as f:
        data = json.load(f)
        return data


def get_json_value(json_filename, key, default_data=None):
    """
    get value by key in json file if your json file stored value as one dict.
    """
    data = get_json_data(json_filename, default_data=default_data)
    if not isinstance(data, dict):
        raise Exception(
            "the target json file must stored whole data as one dict.")

    return data.get(key)


def set_json_value(json_filename, key, value, default_data=None):
    """
    set value by key and value in json file if your json file stored value
    as one dict.
    """
    data = get_json_data(json_filename, default_data=default_data)

    if not isinstance(data, dict):
        raise Exception(
            "the target json file must stored whole data as one dict.")

    data[key] = value
    write_json(get_json_file(json_filename), data)


def find_trainning_files(root, regexp, **kwargs):
    items = []

    for dirname, subdirs, fileids in os.walk(root, **kwargs):
        for fileid in fileids:
            if re.match(regexp, fileid):
                items.append(os.path.join(dirname, *subdirs, fileid))
    return items


def read_training_content(root, regexp):
    return ''.join([open(file, encoding='utf8').read() for file in
                    find_trainning_files(root, regexp)])
