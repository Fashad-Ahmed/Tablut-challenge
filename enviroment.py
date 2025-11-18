from tablut_socket import TablutSocket
import subprocess
import os
import time
import logging


class Environment:
    def __init__(self):
        self._server_process = None
        self._random_black_process = None
        # simple logger
        self._logger = logging.getLogger("Environment")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

    def run_server(self, server=None):
        """Start the external Java TablutServer by running 'ant server' in the
        TablutServer directory. The subprocess is stored in
        self._server_process so it can be terminated later.

        If the server is already running (self._server_process not None and
        still alive) this is a no-op.
        """
        if self._server_process is not None:
            # check if still running
            if self._server_process.poll() is None:
                self._logger.debug("Tablut server already running (pid=%s)", self._server_process.pid)
                return self._server_process
            else:
                # previous process exited; clear it
                self._logger.debug("Previous Tablut server process exited with code %s", self._server_process.returncode)
                self._server_process = None

        # Build the command and working directory
        repo_root = os.path.dirname(os.path.abspath(__file__))
        server_dir = os.path.join(repo_root, 'TablutServer', 'Tablut')

        if not os.path.isdir(server_dir):
            self._logger.error("TablutServer directory not found at %s", server_dir)
            raise FileNotFoundError(f"TablutServer directory not found: {server_dir}")

        cmd = ['ant', 'server']

        # Prepare log files
        logs_dir = os.path.join(server_dir, 'logs')
        try:
            os.makedirs(logs_dir, exist_ok=True)
        except Exception:
            logs_dir = server_dir

        stdout_log = open(os.path.join(logs_dir, 'tablut_server_stdout.log'), 'a')
        stderr_log = open(os.path.join(logs_dir, 'tablut_server_stderr.log'), 'a')

        self._logger.info("Starting Tablut server with command: %s in %s", ' '.join(cmd), server_dir)

        try:
            # Start the server in background
            self._server_process = subprocess.Popen(
                cmd,
                cwd=server_dir,
                stdout=stdout_log,
                stderr=stderr_log,
                env=os.environ.copy()
            )
        except FileNotFoundError as e:
            # 'ant' not found
            stdout_log.close()
            stderr_log.close()
            self._logger.exception("Failed to start Tablut server: %s", e)
            raise

        # small wait to see if it dies quickly
        time.sleep(0.2)
        if self._server_process.poll() is not None:
            code = self._server_process.returncode
            self._logger.error("Tablut server process exited immediately with code %s", code)
            raise RuntimeError(f"Tablut server failed to start, exit code {code}")

        self._logger.info("Tablut server started (pid=%s)", self._server_process.pid)
        return self._server_process
    

    def run_random_black(self):
        if self._random_black_process is not None:
            # check if still running
            if self._random_black_process.poll() is None:
                return self._random_black_process
            else:
                # previous process exited; clear it
                self._random_black_process = None

        # Build the command and working directory
        repo_root = os.path.dirname(os.path.abspath(__file__))
        server_dir = os.path.join(repo_root, 'TablutServer', 'Tablut')

        if not os.path.isdir(server_dir):
            self._logger.error("TablutServer directory not found at %s", server_dir)
            raise FileNotFoundError(f"TablutServer directory not found: {server_dir}")

        cmd = ['ant', 'randomblack']
        try:
            # Start the server in background
            self._random_black_process = subprocess.Popen(
                cmd,
                cwd=server_dir,
                env=os.environ.copy()
            )
        except FileNotFoundError as e:
            raise

        # small wait to see if it dies quickly
        time.sleep(0.2)
        if self._random_black_process.poll() is not None:
            code = self._random_black_process.returncode
            raise RuntimeError(f"Tablut random black failed to start, exit code {code}")

        return self._random_black_process

    def stop_server(self):
        """Terminate the server process if running."""
        if self._server_process is None:
            return
        if self._server_process.poll() is None:
            self._logger.info("Terminating Tablut server (pid=%s)", self._server_process.pid)
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except Exception:
                self._logger.warning("Killing Tablut server (pid=%s)", self._server_process.pid)
                self._server_process.kill()
        self._server_process = None

    def reset(self):
        self.stop_server()
        self.run_server()

    def get_moves(self):
        pass