"""
Turn a .dwg into a .dxf that ezdxf can actually open, using whichever converter is present.

Why this exists, in order of how much it matters:

1. THE PIPELINE'S FIRST STAGE WAS NOT REPRODUCIBLE. The working file the extractor reads,
   Assets/converted/la_meridiana_clean.dxf, was produced by a manual command that was never
   committed. Nothing in the repo could rebuild it from the DWG. This module is that step,
   written down.

2. RAW LibreDWG OUTPUT DOES NOT OPEN. Converting the real client DWG with dwg2dxf 0.13.4
   yields a DXF that ezdxf rejects outright -- `DXFStructureError: Invalid sort handle code
   331, expected 5` -- and `ezdxf.recover` does not rescue it either. The cause is 155
   malformed SORTENTSTABLE objects. Those record draw order, which nothing here uses, so
   dropping them is safe and is what makes the file loadable.

3. ONE CONVERTER IS A SINGLE POINT OF FAILURE. DWG is a closed format that changes with each
   AutoCAD release; LibreDWG is a clean-room reimplementation and does not read everything.
   On this one file it silently discarded a lot -- counted from its own stderr:

       871  unknown objects skipped outright
       529  HATCH objects with truncated handle streams
       426  "Invalid data type in TABLE entity"
        14  AEC_DISPROPSXSECTIONCOMMON

   The last line is the one to watch. AEC_* are AutoCAD Architecture's native wall, door and
   window objects. If an architect draws with those rather than plain polylines, LibreDWG
   cannot read the very things we need, and layer-based extraction will quietly read
   whatever ordinary geometry happens to sit nearby instead. This file did not depend on
   them. Another firm's file might.

   ODA File Converter is from the Open Design Alliance, whose libraries most CAD software is
   built on, and reads essentially every DWG version. It is free but must be installed by
   hand, so it is tried first when present and skipped when not.

UNVERIFIED: the ODA backend has not been run -- ODA File Converter is not installed in this
environment and cannot be installed unattended (manual download, licence acceptance). Its
command line is written from the vendor's documented usage. Treat it as untested until it
has converted a real file. The LibreDWG backend and the verification step below ARE tested
against the real client DWG.
"""
import glob
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

# dwg2dxf can be installed with its shared library outside the loader's default path, which
# fails at exec time with "libredwg.so.0: cannot open shared object file" -- exactly the
# state this machine was found in. Add the usual local prefix rather than requiring the
# caller to have exported LD_LIBRARY_PATH.
_EXTRA_LIB_DIRS = [os.path.expanduser("~/.local/lib"), "/usr/local/lib"]


def _env_with_libs() -> Dict[str, str]:
    env = dict(os.environ)
    existing = env.get("LD_LIBRARY_PATH", "")
    dirs = [d for d in _EXTRA_LIB_DIRS if os.path.isdir(d)]
    env["LD_LIBRARY_PATH"] = os.pathsep.join([*dirs, existing]) if existing else os.pathsep.join(dirs)
    return env


def strip_objects(dxf_path: str, out_path: str, drop_types: List[str]) -> int:
    """
    Remove whole DXF objects of the given types. Returns how many were dropped.

    A DXF is alternating (group code, value) lines. An object begins with group code 0 and
    its type, and runs until the next group code 0. So: copy pairs through, and when a 0 pair
    names a type we are dropping, skip everything until the next 0 pair.
    """
    dropped = 0
    with open(dxf_path, "r", errors="replace") as src, open(out_path, "w") as dst:
        skipping = False
        buf = []
        while True:
            code = src.readline()
            if not code:
                break
            value = src.readline()
            if not value:
                if not skipping:
                    dst.write(code)
                break
            if code.strip() == "0":
                if value.strip() in drop_types:
                    skipping = True
                    dropped += 1
                else:
                    skipping = False
            if not skipping:
                buf.append(code)
                buf.append(value)
                if len(buf) >= 8192:
                    dst.write("".join(buf))
                    buf.clear()
        dst.write("".join(buf))
    return dropped


def verify_dxf(path: str) -> Dict[str, Any]:
    """
    Open the DXF and describe what is in it. A converter that 'succeeded' but produced
    something unreadable, or readable but empty, has not succeeded.
    """
    import collections

    from ezdxf import recover

    result: Dict[str, Any] = {"path": path, "loadable": False, "error": None,
                              "entities": 0, "layers": 0, "audit_errors": 0,
                              "top_types": [], "has_aec": False}
    try:
        doc, auditor = recover.readfile(path)
    except Exception as exc:                      # noqa: BLE001 - any parse failure is a fail
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    msp = doc.modelspace()
    types = collections.Counter(e.dxftype() for e in msp)
    layers = collections.Counter(e.dxf.layer for e in msp if e.dxf.hasattr("layer"))
    result.update(
        loadable=True,
        entities=int(sum(types.values())),
        layers=len(layers),
        audit_errors=len(auditor.errors),
        top_types=types.most_common(6),
        has_aec=any("AEC" in t.upper() for t in types),
    )
    return result


def _convert_oda(dwg_path: str, out_path: str) -> Optional[str]:
    """ODA File Converter. Batch-only: it takes directories, not filenames."""
    exe = shutil.which("ODAFileConverter") or shutil.which("ODAFileConverter.exe")
    if not exe:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        indir, outdir = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(indir)
        os.makedirs(outdir)
        shutil.copy(dwg_path, indir)
        # <in> <out> <outver> <outfmt> <recurse> <audit> [filter]
        subprocess.run([exe, indir, outdir, "ACAD2018", "DXF", "0", "1", "*.DWG"],
                       check=False, capture_output=True, timeout=600)
        produced = glob.glob(os.path.join(outdir, "*.dxf")) + glob.glob(os.path.join(outdir, "*.DXF"))
        if not produced:
            return None
        shutil.copy(produced[0], out_path)
    return "ODAFileConverter"


def _convert_libredwg(dwg_path: str, out_path: str) -> Optional[str]:
    """LibreDWG's dwg2dxf, then strip the draw-order objects that make the result unopenable."""
    exe = shutil.which("dwg2dxf") or os.path.expanduser("~/.local/bin/dwg2dxf")
    if not os.path.exists(exe):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw.dxf")
        proc = subprocess.run([exe, "-o", raw, dwg_path], env=_env_with_libs(),
                              capture_output=True, text=True, timeout=600)
        if not os.path.exists(raw) or os.path.getsize(raw) == 0:
            return None
        dropped = strip_objects(raw, out_path, ["SORTENTSTABLE"])
        skipped = proc.stderr.count("Unknown object, skipping")
        return f"LibreDWG (dropped {dropped} SORTENTSTABLE, converter skipped {skipped} unknown objects)"


BACKENDS = [
    ("oda", _convert_oda),
    ("libredwg", _convert_libredwg),
]


def convert(dwg_path: str, out_path: str) -> Dict[str, Any]:
    """
    Convert dwg_path to out_path, trying each backend until one produces a DXF that opens.

    Returns a report. `ok` is False unless a backend produced a file ezdxf could read with at
    least one entity in it -- a converter exiting 0 is not evidence of a usable result, which
    is precisely how the unopenable DXF got this far in the first place.
    """
    if not os.path.exists(dwg_path):
        raise FileNotFoundError(dwg_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    attempts = []
    for name, fn in BACKENDS:
        try:
            detail = fn(dwg_path, out_path)
        except (subprocess.SubprocessError, OSError) as exc:
            attempts.append({"backend": name, "available": True, "note": f"failed: {exc}"})
            continue
        if detail is None:
            attempts.append({"backend": name, "available": False, "note": "not installed"})
            continue

        check = verify_dxf(out_path)
        attempts.append({"backend": name, "available": True, "note": detail, "check": check})
        if check["loadable"] and check["entities"] > 0:
            return {"ok": True, "backend": name, "detail": detail,
                    "out_path": out_path, "check": check, "attempts": attempts}

    return {"ok": False, "backend": None, "out_path": out_path, "attempts": attempts}
