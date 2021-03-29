import os
import time
import platform
import json
PLATFORM_WINDOWS = 'Windows'

if platform.system() == PLATFORM_WINDOWS:
    # conda install -c anaconda pywin32
    import win32file, win32con, pywintypes
else:
    import fcntl


class FileUtils(object):

    @staticmethod
    def lock_file(f):
        while True:
            try:
                if platform.system() == PLATFORM_WINDOWS:
                    # hfile = win32file._get_osfhandle(f.fileno())
                    # win32file.LockFileEx(hfile, win32con.LOCKFILE_FAIL_IMMEDIATELY | win32con.LOCKFILE_EXCLUSIVE_LOCK,
                    #                      0, 0xffff0000, pywintypes.OVERLAPPED())
                    break
                else:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
            except:
                time.sleep(0.1)

    @staticmethod
    def writeJSON(path, obj):
        with open(path, 'w') as fp:
            FileUtils.lock_file(fp)
            json.dump(obj, fp, indent=4)
            FileUtils.lock_file(fp)

    @staticmethod
    def unlock_file(f):
        while True:
            try:
                if platform.system() == PLATFORM_WINDOWS:
                    # hfile = win32file._get_osfhandle(f.fileno())
                    # win32file.UnlockFileEx(hfile, 0, 0, 0xffff0000, pywintypes.OVERLAPPED())
                    break
                else:
                    fcntl.flock(f, fcntl.LOCK_UN)
                break
            except:
                time.sleep(0.1)

    @staticmethod
    def deleteDir(dirPath, is_delete_dir_path = False):
        if os.path.exists(dirPath):
            deleteFiles = []
            deleteDirs = []
            for root, dirs, files in os.walk(dirPath):
                for f in files:
                    deleteFiles.append(os.path.join(root, f))
                for d in dirs:
                    deleteDirs.append(os.path.join(root, d))
            for f in deleteFiles:
                os.remove(f)
            for d in deleteDirs:
                os.rmdir(d)
            if is_delete_dir_path:
                os.rmdir(dirPath)

    @staticmethod
    def createDir(dirPath):
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)