import argparse
import ctypes
import os
import subprocess
import sys
import time


def wait_for_pid(pid, poll_sec):
    if os.name == 'nt':
        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, int(pid))
        if handle:
            ctypes.windll.kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
            ctypes.windll.kernel32.CloseHandle(handle)
            return
    while True:
        try:
            os.kill(int(pid), 0)
        except OSError:
            return
        time.sleep(poll_sec)


def main():
    parser = argparse.ArgumentParser(description='Wait for a process to finish, then run stage-1 config queue.')
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--poll-sec', type=float, default=30.0)
    args = parser.parse_args()

    wait_for_pid(args.pid, args.poll_sec)
    env = os.environ.copy()
    if os.name == 'nt':
        env.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    cmd = [
        sys.executable,
        'scripts/run_stage1_ablation.py',
        '--configs',
        *args.configs,
        '--log-dir',
        args.log_dir,
    ]
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
