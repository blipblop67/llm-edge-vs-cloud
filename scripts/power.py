"""Jetson board power from the INA3221 sensor (VDD_IN = total module input)."""
import glob
import os
import threading
import time


def find_vdd_in():
    """Find the voltage and current files for the VDD_IN channel."""
    for label in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label"):
        with open(label) as f:
            if f.read().strip() == "VDD_IN":
                folder = os.path.dirname(label)
                n = os.path.basename(label)[2:-6]          # "in1_label" -> "1"
                return (os.path.join(folder, f"in{n}_input"),
                        os.path.join(folder, f"curr{n}_input"))
    return None


class PowerSampler:
    def __init__(self, interval_s=0.05):
        self.paths = find_vdd_in()
        self.interval_s = interval_s
        self.samples = []                  # list of (timestamp, watts)
        self._stop = threading.Event()
        self._thread = None

    def read_watts(self):
        v_path, c_path = self.paths
        with open(v_path) as fv, open(c_path) as fc:
            return int(fv.read()) * int(fc.read()) / 1e6   # mV * mA -> W

    def _loop(self):
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(), self.read_watts()))
            time.sleep(self.interval_s)

    def start(self):
        if self.paths is None:
            raise RuntimeError("VDD_IN sensor not found: is this a Jetson?")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        time.sleep(0.3)                    # collect a few samples first
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def window(self, t0, t1):
        """All readings taken between t0 and t1."""
        return [w for t, w in list(self.samples) if t0 <= t <= t1]

    def average(self, t0, t1):
        w = self.window(t0, t1)
        return sum(w) / len(w) if w else None

    def idle(self, seconds=5):
        """Average power while nothing is running: the baseline."""
        t0 = time.perf_counter()
        time.sleep(seconds)
        return self.average(t0, time.perf_counter())
