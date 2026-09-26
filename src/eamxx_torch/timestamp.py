"""EAMxx time stamps, calendar and time intervals (host-side, no tensors).

Translation of, at E3SM 80fe0a71:
  components/eamxx/src/share/util/eamxx_time_stamp.{hpp,cpp}
      is_leap_year, days_in_month, TimeStamp (operator+=, frac_of_year_in_days,
      days_in_curr_year, comparisons, operator-), TimeInterval
  components/eamxx/src/share/core/eamxx_config.cpp  use_leap_year (default: on,
      because the reference is built with SCREAM_HAS_LEAP_YEAR=ON)

Python floats are IEEE doubles and the operation order is kept, so interval
lengths and interpolation weights are bitwise identical to the C++.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

SECONDS_PER_DAY = 86400  # constants::seconds_per_day

_use_leap_year = True


def set_use_leap_year(flag: bool) -> None:
    global _use_leap_year
    _use_leap_year = flag


def is_leap_year(yy: int) -> bool:
    if _use_leap_year:
        if yy % 4 == 0:
            if yy % 100 != 0:
                return True
            elif (yy // 100) % 4 == 0:
                return True
    return False


_NONLEAP = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_LEAP = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def days_in_month(yy: int, mm: int) -> int:
    if not 1 <= mm <= 12:
        raise ValueError("Month out of bounds")
    return (_LEAP if is_leap_year(yy) else _NONLEAP)[mm - 1]


def _c_int_div(a: int, b: int) -> int:
    """C++ integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _c_int_mod(a: int, b: int) -> int:
    return a - b * _c_int_div(a, b)


@dataclass
class TimeStamp:
    date: list[int] = field(default_factory=lambda: [None, None, None])  # yy, mm, dd
    time: list[int] = field(default_factory=lambda: [None, None, None])  # h, min, sec
    num_steps: int = 0
    sec_fraction: float = 0.0

    @classmethod
    def make(cls, yy, mm, dd, h=0, mi=0, s=0, num_steps=0) -> "TimeStamp":
        if not (0 <= yy <= 9999 and 1 <= mm <= 12 and 1 <= dd <= days_in_month(yy, mm)
                and 0 <= s < 60 and 0 <= mi < 60 and 0 <= h < 24 and num_steps >= 0):
            raise ValueError("TimeStamp out of bounds")
        return cls([yy, mm, dd], [h, mi, s], num_steps, 0.0)

    @classmethod
    def from_string(cls, s: str) -> "TimeStamp":
        """Parse 'YYYY-MM-DD-XXXXX' (seconds of day), as util::str_to_time_stamp."""
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})-(\d{5})", s)
        if not m:
            raise ValueError(f"bad time stamp {s!r}")
        yy, mm, dd, tod = map(int, m.groups())
        return cls.make(yy, mm, dd, tod // 3600, (tod % 3600) // 60, tod % 60)

    def is_valid(self) -> bool:
        return None not in self.date

    def copy(self) -> "TimeStamp":
        return TimeStamp(list(self.date), list(self.time), self.num_steps, self.sec_fraction)

    def sec_of_day(self) -> int:
        return self.time[0] * 3600 + self.time[1] * 60 + self.time[2]

    def to_string(self) -> str:
        return f"{self.date[0]:04d}-{self.date[1]:02d}-{self.date[2]:02d}-{self.sec_of_day():05d}"

    def frac_of_year_in_days(self) -> float:
        doy = float(self.date[2] - 1)
        doy += (self.sec_of_day() + self.sec_fraction) / 86400.0
        for m in range(1, self.date[1]):
            doy += days_in_month(self.date[0], m)
        return doy

    def days_in_curr_year(self) -> int:
        return 366 if is_leap_year(self.date[0]) else 365

    def get_month(self) -> int:
        return self.date[1]

    def iadd(self, seconds: float) -> "TimeStamp":
        """TimeStamp::operator+=(double seconds)."""
        if seconds < 0:
            raise ValueError("Cannot rewind time")
        if not self.is_valid():
            raise ValueError("Uninitialized time stamp")
        self.sec_fraction += seconds
        frac, whole = math.modf(self.sec_fraction)
        self.sec_fraction, seconds = frac, whole
        if self.sec_fraction < 1e-3:
            self.sec_fraction = 0.0
        elif self.sec_fraction > 0.999:
            self.sec_fraction = 0.0
            seconds += 1
        self.num_steps += 1

        spd = SECONDS_PER_DAY
        add_days = math.floor(seconds / spd)
        add_secs = seconds - add_days * spd
        # int += double in C++: compute in double, truncate toward zero.
        sec = int(self.time[2] + add_secs)
        dd = int(self.date[2] + add_days)
        mi, hour = self.time[1], self.time[0]
        mm, yy = self.date[1], self.date[0]

        carry = _c_int_div(sec, 60)
        if carry != 0:
            sec = _c_int_mod(sec, 60)
            mi += carry
            carry = _c_int_div(mi, 60)
            if carry != 0:
                mi = _c_int_mod(mi, 60)
                hour += carry
                carry = _c_int_div(hour, 24)
                if carry != 0:
                    hour = _c_int_mod(hour, 24)
                    dd += carry
        while dd > days_in_month(yy, mm):
            dd -= days_in_month(yy, mm)
            mm += 1
            if mm > 12:
                yy += 1
                mm = 1
        self.date = [yy, mm, dd]
        self.time = [hour, mi, sec]
        return self

    def __add__(self, seconds: float) -> "TimeStamp":
        return self.copy().iadd(seconds)

    # Comparisons (operator==, <, <=): year first, then fraction of year.
    def __eq__(self, other) -> bool:
        return (self.date == other.date and self.time == other.time
                and self.sec_fraction == other.sec_fraction)

    def __lt__(self, other) -> bool:
        if self.date[0] < other.date[0]:
            return True
        if self.date[0] == other.date[0]:
            return self.frac_of_year_in_days() < other.frac_of_year_in_days()
        return False

    def __le__(self, other) -> bool:
        if self.date[0] < other.date[0]:
            return True
        if self.date[0] == other.date[0]:
            return self.frac_of_year_in_days() <= other.frac_of_year_in_days()
        return False

    def seconds_from(self, other: "TimeStamp") -> int:
        """operator-(ts1, ts2) in whole seconds (fractions ignored, as in C++)."""
        if self < other:
            return -other.seconds_from(self)
        spd = SECONDS_PER_DAY

        def abs_seconds(ts):
            days = sum(366 if is_leap_year(y) else 365 for y in range(0, ts.date[0]))
            days += sum(days_in_month(ts.date[0], m) for m in range(1, ts.date[1]))
            days += ts.date[2] - 1
            return days * spd + ts.sec_of_day()

        return abs_seconds(self) - abs_seconds(other)

    def days_from(self, other: "TimeStamp") -> float:
        return self.seconds_from(other) / 86400.0


YEARLY_PERIODIC = "yearly_periodic"
LINEAR = "linear"


@dataclass
class TimeInterval:
    """util::TimeInterval."""

    beg: TimeStamp
    end: TimeStamp
    timeline: str = LINEAR
    length: float = -1.0

    def __post_init__(self):
        pass

    @classmethod
    def make(cls, beg, end, timeline, do_compute_length=True) -> "TimeInterval":
        ti = cls(beg, end, timeline)
        if do_compute_length:
            ti.compute_length()
        return ti

    def contains(self, t: TimeStamp) -> bool:
        if self.timeline == LINEAR:
            return self.beg <= t and t <= self.end
        t_frac = t.frac_of_year_in_days()
        end_frac = self.end.frac_of_year_in_days()
        beg_frac = self.beg.frac_of_year_in_days()
        across_nye = self.beg.get_month() > self.end.get_month()
        if not across_nye:
            return beg_frac <= t_frac and t_frac <= end_frac
        return t_frac >= beg_frac or t_frac <= end_frac

    def compute_length(self) -> None:
        if self.timeline == LINEAR:
            self.length = self.end.days_from(self.beg)
        else:
            across_nye = self.beg.get_month() > self.end.get_month()
            frac_beg = self.beg.frac_of_year_in_days()
            frac_end = self.end.frac_of_year_in_days()
            if across_nye:
                year = float(self.end.days_in_curr_year())
                self.length = frac_end + (year - frac_beg)
            else:
                self.length = frac_end - frac_beg

    def advance(self, new_end: TimeStamp) -> None:
        self.beg = self.end
        self.end = new_end
        self.compute_length()


def parse_cf_time_units(units: str) -> tuple[TimeStamp, int]:
    """parse_cf_time_units: '<unit> since <date> [<time>]' -> (ref, seconds per unit)."""
    m = re.fullmatch(r"\s*(\w+)\s+since\s+(\d{1,4})-(\d{1,2})-(\d{1,2})"
                     r"(?:[ T](\d{1,2}):(\d{1,2}):(\d{1,2})(?:\.\d*)?|-(\d{5}))?\s*", units)
    if not m:
        raise ValueError(f"unsupported CF time units: {units!r}")
    unit = m.group(1).lower()
    mult = {"seconds": 1, "second": 1, "minutes": 60, "minute": 60,
            "hours": 3600, "hour": 3600, "days": 86400, "day": 86400}[unit]
    yy, mm, dd = int(m.group(2)), int(m.group(3)), int(m.group(4))
    if m.group(8) is not None:
        tod = int(m.group(8))
        h, mi, s = tod // 3600, (tod % 3600) // 60, tod % 60
    elif m.group(5) is not None:
        h, mi, s = int(m.group(5)), int(m.group(6)), int(m.group(7))
    else:
        h = mi = s = 0
    return TimeStamp.make(yy, mm, dd, h, mi, s), mult
