#!/usr/bin/env python3
#
# Copyright 2020 Seth Troisi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Search records for starting point with several nearby records"""

import re
import sys

GAP_RE = re.compile(r"[0-9]* ([0-9]*\.[0-9]*) ([0-9]*) \*")

fn = "records_337.txt" if len(sys.argv) != 2 else sys.argv[1]

with open(fn) as f:
    data = f.readlines()

records = []
for line in data:
    m = GAP_RE.match(line)
    if m and float(m.group(1)) > 27:
        records.append(int(m.group(2)))

print(f"Found {len(records)} gaps in {fn!r}")

records = sorted(set(records))
for d in range(2, 10):
    distance, start = min((records[i] - records[i-d], records[i-d]) for i in range(d, len(records)))
    print(f"Start at {start} has {d} records in next {distance:,} m")
