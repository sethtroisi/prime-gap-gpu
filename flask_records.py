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

import re
import os

from flask import Flask, Response


app = Flask(__name__)

RECORDS_RE = re.compile("^records_[1-9][0-9]*.txt$")

@app.route("/records/<name>")
def merits(name):
    if RECORDS_RE.fullmatch(name):
        if os.path.exists(name):
            with open(name) as f:
                return Response(f.read(), mimetype="text/plain")

    return "Not Found"


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        # host = "::",
        port=5080,
        debug=False,
    )
