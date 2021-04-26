# -*- coding: utf-8 -*-

"""
Requirements
------------
sudo -H pip3 install gitpython
# sudo -H pip3 install python-git
"""

import sys
import os
import os.path as osp
import yaml
from easydict import EasyDict
import logging
import datetime
import subprocess
import git

cfg_file = osp.join(osp.dirname(__file__), 'open3d_freetech_config.yaml')

if not osp.exists(cfg_file):
    print('cwd:            {}'.format(os.getcwd()))
    print('path not exist: {}'.format(cfg_file))
    raise ValueError


################################################################################
# Do string concatenation in YAML
#   https://stackoverflow.com/questions/5484016/how-can-i-do-string-concatenation-or-string-replacement-in-yaml
################################################################################
# define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)

    # return ''.join([str(i) for i in seq])
    if len(seq) == 0:
        return ''
    elif len(seq) == 1:
        return str(seq[0])
    elif len(seq) == 2:
        return osp.join(str(seq[0]), str(seq[1]))
    else:
        path = osp.join(str(seq[0]), str(seq[1]))
        for idx in range(2, len(seq)):
            path = osp.join(path, str(seq[idx]))
        return path


# register the tag handler
yaml.add_constructor('!join', join)


################################################################################
# Load YAML
################################################################################
with open(cfg_file, encoding='utf-8') as f_yaml:
    cfg_in_yaml = EasyDict(yaml.load(f_yaml, Loader=yaml.Loader))


################################################################################
# Configuration
################################################################################
class CommonAPI:
    def __init__(self, cfg_dict):
        self.cfg = cfg_dict
        repo_path = osp.join(osp.dirname(__file__), '../')
        repo = git.Repo(repo_path)
        head_commit_hex = repo.head.commit.hexsha[:8]
        dirty_info = 'dirty_{}'.format(repo.is_dirty())
        self.repo_info = '{}_{}'.format(head_commit_hex, dirty_info)

    # ======================================================================== #
    # prepare data
    # ======================================================================== #
    def setup_data(self):
        if not osp.exists(self.get_home_dir):
            os.makedirs(self.get_home_dir)

        # config
        if not osp.exists(self.get_config_dir):
            os.makedirs(self.get_config_dir)

        # copy config file
        src_file = cfg_file
        base_name = osp.basename(src_file)
        dst_file = osp.join(self.get_config_dir, base_name)
        dt = datetime.datetime.now()
        str_name, str_ext = osp.splitext(dst_file)
        str_date = dt.strftime('_%Y%m%d_')
        dst_file = str_name + str_date + self.repo_info + str_ext

        if sys.platform != 'win32':
            subprocess.call(['cp', src_file, dst_file])
        else:
            os.system('copy "{}" "{}"'.format(src_file, dst_file))
            # os.system('xcopy /s {} {}'.format(src_file, dst_file))
        # ===============================================
        # or:
        #   import shutil
        #   shutil.copyfile("path-to-src", "path-to-dst")
        # ===============================================

        # output
        if not osp.exists(self.get_ou_dir):
            os.makedirs(self.get_ou_dir)

        # log
        if not osp.exists(self.get_log_dir):
            os.makedirs(self.get_log_dir)

        # temp
        if not osp.exists(self.get_temp_dir):
            os.makedirs(self.get_temp_dir)

    @property
    def get_date_str(self):
        """
        References
        ----------
        https://strftime.org/
        """
        dt = datetime.datetime.now()
        return dt.strftime('_%Y%m%d_%A')

    # ======================================================================== #
    # < Get Dir >
    # ======================================================================== #
    @property
    def get_home_dir(self):
        # return self.cfg.WORKSPACE.HOME
        return osp.join(osp.dirname(__file__), '../../Open3D_freetech_Home')

    @property
    def get_ou_dir(self):
        return osp.join(self.get_home_dir, self.cfg.WORKSPACE.OUTPUT.DIR)

    @property
    def get_log_dir(self):
        return osp.join(self.get_home_dir, self.cfg.WORKSPACE.LOG.DIR)

    @property
    def get_log_file(self):
        log_name = osp.join(self.get_log_dir, self.cfg.WORKSPACE.LOG.FILE)
        dt = datetime.datetime.now()
        str_name, str_ext = osp.splitext(log_name)
        str_date = dt.strftime('_%Y%m%d_')
        log_name = str_name + str_date + self.repo_info + str_ext
        return log_name

    @property
    def get_temp_dir(self):
        return osp.join(self.get_home_dir,
                        self.cfg.WORKSPACE.TEMP.DATA_TEMP)
    # ======================================================================== #
    # </Get Dir >
    # ======================================================================== #

    # ======================================================================== #
    # < Get Config >
    # ======================================================================== #
    @property
    def get_config_dir(self):
        return osp.join(self.get_home_dir, self.cfg.WORKSPACE.CONFIG.DIR)

    @property
    def get_is_log_enabled(self):
        return self.cfg.WORKSPACE.CONFIG.ENABLE_LOG

    @property
    def get_is_release(self):
        return self.cfg.WORKSPACE.CONFIG.RELEASE

    @property
    def get_is_quiet_enabled(self):
        return self.cfg.WORKSPACE.CONFIG.ENABLE_QUIET
    # ======================================================================== #
    # </Get Config >
    # ======================================================================== #

    # ======================================================================== #
    # < Get Param >
    # ======================================================================== #
    # ======================================================================== #
    # </Get Param >
    # ======================================================================== #

    # ======================================================================== #
    # Log setting
    # ======================================================================== #
    def setup_log(self):
        if not self.get_is_log_enabled:
            return

        # Initialize logging
        # simple_format = '%(levelname)s >>> %(message)s'
        medium_format = (
            '%(levelname)s : %(filename)s[%(lineno)d]'
            ' >>> %(message)s'
        )
        logging.basicConfig(
            filename=self.get_log_file,
            filemode='w',
            level=logging.INFO,
            format=medium_format
        )
        logging.info('@{} created at {}'.format(
            self.get_log_file,
            datetime.datetime.now())
        )


################################################################################
# Global config
################################################################################
common_api = CommonAPI(cfg_in_yaml)
common_api.setup_data()
common_api.setup_log()


################################################################################
# Utility
################################################################################
def view_api(obj, brief=True):
    """
    Print api of object.
    """
    logging.warning('view_api( {} )'.format(type(obj)))

    if brief:
        return

    obj_dir = dir(obj)

    api_base = list()
    api_protect = list()
    api_public = list()

    for item_dir in obj_dir:
        if item_dir.startswith('__') or item_dir.endswith('__'):
            api_base.append(item_dir)
        elif item_dir.startswith('_'):
            api_protect.append(item_dir)
        else:
            api_public.append(item_dir)

    enable_sort = False
    if enable_sort:
        api_base.sort()
        api_public.sort()
        api_protect.sort()

    enable_log = True
    if enable_log:
        logging.info('{} {} API {}'.format('=' * 20, type(obj), '=' * 20))

        logging.info('{} public api {}'.format('-' * 10, '-' * 10))
        for item_dir in api_public:
            logging.info('  --> {}'.format(item_dir))

        logging.info('{} protect api {}'.format('-' * 10, '-' * 10))
        for item_dir in api_protect:
            logging.info('  --> {}'.format(item_dir))

        logging.info('{} base api {}'.format('-' * 10, '-' * 10))
        for item_dir in api_base:
            logging.info('  --> {}'.format(item_dir))


def get_specific_files(name_dir, name_ext, with_dir=False):
    """
    Get files with specific extension.

    Parameters
    ----------
    name_dir : str
    name_ext : str
    with_dir : bool

    Returns
    -------
    list of str
    """
    specific_files = list()
    for item_file in os.listdir(name_dir):
        str_base, str_ext = osp.splitext(item_file)
        if str_ext.lower() == name_ext.lower():
            if with_dir:
                specific_files.append(osp.join(name_dir, item_file))
            else:
                specific_files.append(item_file)
    specific_files.sort()
    return specific_files
