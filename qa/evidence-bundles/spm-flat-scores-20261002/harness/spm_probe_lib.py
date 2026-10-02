"""Minimal GGUF metadata reader (keys and values only, no tensors)."""
import struct


def read_gguf_metadata(path):
    with open(path, "rb") as fh:
        def u32():
            return struct.unpack("<I", fh.read(4))[0]

        def u64():
            return struct.unpack("<Q", fh.read(8))[0]

        def string():
            return fh.read(u64()).decode("utf-8", "replace")

        scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                inner, count = u32(), u64()
                return [value(inner) for _ in range(count)]
            fmt = scalar[kind]
            return struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0]

        assert fh.read(4) == b"GGUF"
        version = u32()
        _tensors, kv_count = u64(), u64()
        meta = {}
        for _ in range(kv_count):
            key = string()
            meta[key] = value(u32())
        return version, meta
