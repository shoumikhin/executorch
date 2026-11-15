# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import io
import os
import shutil
import stat
import tempfile
import weakref
from typing import Container, IO, List, Optional, Union


class FileBackedData:
    """A byte buffer that stays on disk until explicitly closed or consumed.

    ``write_to_file(outfile, consume=True)`` moves data that owns its file
    (``cleanup=True``, as from ``move_from``) into ``outfile`` instead of copying it.
    Chunks are copied from the end back to the start, and the file is truncated after
    each chunk is written and synced to ``outfile``; the part of ``outfile`` not yet
    written stays a hole. On a file system with sparse files, the data is then on disk
    once, plus one chunk. The move needs a regular file that is not a link and has no
    other name, and a regular output on the same file system that writes at the
    offsets it seeks to. Anything else, and Windows (where files are not sparse by
    default), gets a copy. Afterwards the file is empty, and neither this nor any other
    ``FileBackedData`` on it can be read again.

    A consuming write that fails before its first truncation leaves the data intact,
    so it can be written again. One that fails later leaves the file in place and says,
    in a note on the error, which bytes are still in it and where the rest is in
    ``outfile``.
    """

    _COPY_CHUNK_SIZE = 8 * 1024 * 1024

    def __init__(
        self,
        path: str,
        cleanup: bool = False,
        sha256: Optional[bytes] = None,
    ) -> None:
        self._path = path
        self._size = os.path.getsize(path)
        self._sha256 = sha256
        self._finalizer = (
            weakref.finalize(self, self._remove, path) if cleanup else None
        )

    @staticmethod
    def _remove(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    @classmethod
    def move_from(cls, path: str, sha256: Optional[bytes] = None) -> "FileBackedData":
        """Take ownership of ``path`` without loading its contents."""
        directory = os.path.dirname(path) or "."
        fd, owned_path = tempfile.mkstemp(
            prefix=".executorch_", suffix=".data", dir=directory
        )
        os.close(fd)
        try:
            os.replace(path, owned_path)
        except Exception:
            os.remove(owned_path)
            raise
        return cls(owned_path, cleanup=True, sha256=sha256)

    def __len__(self) -> int:
        return self._size

    def _check_size(self, size: int) -> None:
        # The size, not a flag, shows a consumed file, so every object on it sees it.
        if size != self._size:
            raise RuntimeError(
                f"{self._path} has {size} bytes, expected {self._size}; a consuming "
                "write may have moved them"
            )

    def check(self) -> None:
        """Raise if the file no longer holds the data, as after a consuming write."""
        self._check_size(os.stat(self._path).st_size)

    def _open(self) -> IO[bytes]:
        f = open(self._path, "rb")
        try:
            self._check_size(os.fstat(f.fileno()).st_size)
        except BaseException:
            f.close()
            raise
        return f

    def prefix(self, size: int) -> bytes:
        with self._open() as f:
            return f.read(size)

    def sha256(self) -> bytes:
        if self._sha256 is None:
            digest = hashlib.sha256()
            with self._open() as f:
                while chunk := f.read(self._COPY_CHUNK_SIZE):
                    digest.update(chunk)
            self._sha256 = digest.digest()
        return self._sha256

    def to_bytes(self) -> bytes:
        with self._open() as f:
            return f.read()

    def write_to_file(
        self, outfile: io.BufferedIOBase, *, consume: bool = False
    ) -> Optional[int]:
        """Write the data to ``outfile``; with ``consume``, move it there when it can
        (see the class docstring). Returns the offset in ``outfile`` the data was moved
        to, or None if it was copied."""
        source = self._open_to_consume(outfile) if consume else None
        if source is None:
            with self._open() as f:
                shutil.copyfileobj(f, outfile, length=self._COPY_CHUNK_SIZE)
            return None
        try:
            return self._consume_into(outfile, source)
        finally:
            os.close(source)

    def _open_to_consume(self, outfile: io.BufferedIOBase) -> Optional[int]:
        """The file opened for a move into ``outfile``, or None if it must be copied."""
        owned = self._finalizer is not None and self._finalizer.alive
        if not owned or os.name == "nt":
            return None
        if not isinstance(outfile, (io.BufferedWriter, io.BufferedRandom, io.FileIO)):
            return None
        import fcntl  # POSIX only; Windows returned above.

        try:
            output_fd = outfile.fileno()
            output = os.fstat(output_fd)
            appends = fcntl.fcntl(output_fd, fcntl.F_GETFL) & os.O_APPEND
            # O_NOFOLLOW refuses a link, and the checks below look at the file opened.
            fd = os.open(self._path, os.O_RDWR | os.O_NOFOLLOW)
        except OSError:
            return None
        source = os.fstat(fd)
        # With O_APPEND every write lands at the end, not at the chunk's offset.
        # Truncating changes every name of the file, and on another file system it
        # frees nothing for the output.
        if (
            appends
            or not stat.S_ISREG(output.st_mode)
            or source.st_nlink != 1
            or source.st_dev != output.st_dev
            or source.st_size != self._size
        ):
            os.close(fd)
            return None
        return fd

    def _consume_into(self, outfile: io.BufferedIOBase, source: int) -> int:
        start = outfile.tell()
        end = self._size
        try:
            while end > 0:
                begin = max(end - self._COPY_CHUNK_SIZE, 0)
                chunk = os.pread(source, end - begin, begin)
                if len(chunk) != end - begin:
                    raise OSError(
                        f"read {len(chunk)} of {end - begin} bytes at offset {begin} "
                        f"of {self._path}"
                    )
                outfile.seek(start + begin)
                written = outfile.write(chunk)
                if written != len(chunk):
                    raise OSError(
                        f"wrote {written} of {len(chunk)} bytes at offset "
                        f"{start + begin}"
                    )
                # The chunk has to be on the output's disk before the source gives it up.
                outfile.flush()
                os.fsync(outfile.fileno())
                # Before the truncation, so the note is true even if an interrupt
                # arrives right after it.
                end = begin
                os.ftruncate(source, begin)
        except BaseException as e:
            if end < self._size:
                # The note sends the caller to the rest of the data, so it must
                # outlive this object.
                if self._finalizer is not None:
                    self._finalizer.detach()
                e.add_note(
                    f"Bytes [0, {end}) are still in {self._path}; bytes "
                    f"[{end}, {self._size}) are at offset {start + end} of the output."
                )
            raise
        outfile.seek(start + self._size)
        return start

    def close(self) -> None:
        if self._finalizer is not None:
            self._finalizer()

    def __enter__(self) -> "FileBackedData":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


CordBuffer = Union[bytes, FileBackedData]


class Cord:
    """A `bytes`-like sequence of bytes, stored non-contiguously.

    Users can use a Cord to assemble large files and data blobs using references
    to and slices of other data, instead of copying and appending that data to a
    `bytes` or `bytearray` object.
    """

    def __init__(self, data: Optional[Union[CordBuffer, "Cord"]] = None) -> None:
        """Initialize Cord data structure."""
        self._buffers: List[CordBuffer] = []
        self._byte_size: int = 0

        if data is not None:
            self.append(data)

    def __len__(self):
        """Number of bytes in the Cord."""
        return self._byte_size

    def __bytes__(self) -> bytes:
        """Return the contents of the Cord as a single `bytes` object."""
        return b"".join(
            item if isinstance(item, bytes) else item.to_bytes()
            for item in self._buffers
        )

    def append(self, data: Union[CordBuffer, "Cord"]) -> None:
        """Append a bytes or Cord to the current Cord."""
        if isinstance(data, (bytes, FileBackedData)):
            self._buffers.append(data)
            self._byte_size += len(data)
        elif isinstance(data, Cord):
            self._buffers.extend(data._buffers)
            self._byte_size += len(data)
        else:
            raise TypeError(
                f"Can only append bytes, FileBackedData, or Cords, received {type(data)}"
            )

    def file_backed(self) -> List[FileBackedData]:
        """The file-backed buffers of the Cord."""
        return [item for item in self._buffers if isinstance(item, FileBackedData)]

    def write_to_file(
        self,
        outfile: io.BufferedIOBase,
        *,
        consume: bool = False,
        keep: Container[FileBackedData] = (),
    ) -> None:
        """Write the Cord to a file.

        With ``consume``, file-backed buffers that own their files, except those in
        ``keep``, are moved into ``outfile`` rather than copied (see
        ``FileBackedData``), so the Cord can't be written or read again afterwards. A
        buffer that appears more than once must be in ``keep``. If the write fails, a
        note on the error lists the buffers already moved, whose only copy is now in
        ``outfile``, and where they are.
        """
        moved: List[str] = []
        try:
            for item in self._buffers:
                if isinstance(item, bytes):
                    outfile.write(item)
                    continue
                offset = item.write_to_file(
                    outfile, consume=consume and item not in keep
                )
                if offset is not None:
                    moved.append(f"{item._path} ({len(item)} bytes) at offset {offset}")
        except BaseException as e:
            if moved:
                e.add_note(
                    "Moved into the output before the failure: " + "; ".join(moved)
                )
            raise
