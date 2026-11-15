# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import gzip
import io
import os
import tempfile
import threading
import unittest
from unittest import mock

from executorch.exir._serialize._cord import Cord, FileBackedData

# Windows always copies, so it can't run the tests of a move.
_needs_move = unittest.skipIf(os.name == "nt", "Windows copies instead of moving")


class TestCord(unittest.TestCase):
    def test_cord_init(self) -> None:
        cord_empty = Cord()
        self.assertEqual(0, len(cord_empty))

        cord = Cord(b"HelloWorld")
        self.assertEqual(10, len(cord))
        self.assertEqual(b"HelloWorld", bytes(cord))

        cord2 = Cord(cord)
        self.assertEqual(10, len(cord2))
        self.assertEqual(b"HelloWorld", bytes(cord))

        # Confirm no copies were made.
        self.assertEqual(id(cord._buffers[0]), id(cord2._buffers[0]))

    def test_cord_append(self) -> None:
        cord = Cord()
        cord.append(b"Hello")
        self.assertEqual(5, len(cord))
        self.assertEqual(b"Hello", bytes(cord))

        cord.append(b"World")
        self.assertEqual(10, len(cord))
        self.assertEqual(b"HelloWorld", bytes(cord))

    def test_cord_append_cord(self) -> None:
        cord = Cord()
        cord.append(b"Hello")
        cord.append((b"World"))

        cord2 = Cord()
        cord2.append(b"Prefix")
        cord2.append(cord)

        self.assertEqual(16, len(cord2))
        self.assertEqual(b"PrefixHelloWorld", bytes(cord2))

        # Confirm that no copies were made when appending a Cord.
        self.assertEqual(id(cord2._buffers[1]), id(cord._buffers[0]))
        self.assertEqual(id(cord2._buffers[2]), id(cord._buffers[1]))

    def test_cord_write_to_file(self) -> None:
        cord = Cord()
        cord.append(b"Hello")
        cord.append(b"World")

        outfile = io.BytesIO()
        cord.write_to_file(outfile)
        self.assertEqual(b"HelloWorld", outfile.getvalue())

    def test_file_backed_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, "source.bin")
            with open(source_path, "wb") as f:
                f.write(b"FileBacked")

            with FileBackedData.move_from(source_path) as data:
                self.assertFalse(os.path.exists(source_path))
                self.assertEqual(10, len(data))

                cord = Cord(b"Prefix")
                cord.append(data)
                outfile = io.BytesIO()
                cord.write_to_file(outfile)
                self.assertEqual(b"PrefixFileBacked", outfile.getvalue())
                self.assertEqual(b"PrefixFileBacked", bytes(cord))

            self.assertEqual([], os.listdir(directory))

    # 10 full chunks and part of one; the part is written last and fits in a buffer.
    _CHUNK = 8192
    _PAYLOAD = bytes(range(256)) * (10 * _CHUNK // 256) + b"x" * 1000

    def _owned(self, directory: str) -> FileBackedData:
        path = os.path.join(directory, "source.bin")
        with open(path, "wb") as f:
            f.write(self._PAYLOAD)
        return FileBackedData.move_from(path)

    def _assert_intact(self, data: FileBackedData) -> None:
        self.assertEqual(self._PAYLOAD, data.to_bytes())

    @_needs_move
    def test_consume_moves_owned_data_without_a_second_copy(self) -> None:
        payload, chunk = self._PAYLOAD, self._CHUNK
        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            source = data._path
            out_path = os.path.join(directory, "out.bin")
            on_disk = []

            class Recording(io.FileIO):
                def write(self, b) -> int:
                    offset = self.tell() - len(b"Prefix")
                    size = super().write(b)
                    if 0 <= offset < len(payload):
                        # The source must still hold the range that reaches the output.
                        assert os.path.getsize(source) >= offset + len(b), offset
                        blocks = os.stat(source).st_blocks
                        on_disk.append(
                            (blocks + os.fstat(self.fileno()).st_blocks) * 512
                        )
                    return size

            cord = Cord(b"Prefix")
            cord.append(data)
            cord.append(b"Suffix")
            with mock.patch.object(FileBackedData, "_COPY_CHUNK_SIZE", chunk):
                with io.BufferedWriter(Recording(out_path, "wb")) as outfile:
                    cord.write_to_file(outfile, consume=True)

            with open(out_path, "rb") as result:
                self.assertEqual(b"Prefix" + payload + b"Suffix", result.read())
            self.assertEqual(0, os.path.getsize(source))
            # Allocated blocks, so the unwritten part of the output must stay a hole.
            self.assertLessEqual(max(on_disk), len(payload) + chunk + 4 * 4096)
            with self.assertRaises(RuntimeError):
                data.to_bytes()

    @_needs_move
    def test_only_a_consuming_write_consumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            cord = Cord(data)
            for _ in range(2):
                with open(os.path.join(directory, "out.bin"), "wb") as outfile:
                    cord.write_to_file(outfile)
            self.assertEqual(self._PAYLOAD, bytes(cord))
            with open(os.path.join(directory, "out.bin"), "wb") as outfile:
                cord.write_to_file(outfile, consume=True)
            with self.assertRaises(RuntimeError):
                bytes(cord)

    def test_consume_is_keyword_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            with open(os.path.join(directory, "out.bin"), "wb") as outfile:
                with self.assertRaises(TypeError):
                    data.write_to_file(outfile, True)
                with self.assertRaises(TypeError):
                    Cord(data).write_to_file(outfile, True)
            self._assert_intact(data)

    def test_consume_copies_data_that_does_not_own_its_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "source.bin")
            with open(path, "wb") as f:
                f.write(self._PAYLOAD)
            data = FileBackedData(path)
            with open(os.path.join(directory, "out.bin"), "wb") as outfile:
                data.write_to_file(outfile, consume=True)
            self._assert_intact(data)

    def test_consume_copies_into_outputs_it_cannot_move_into(self) -> None:
        class WriteOnly:
            def __init__(self) -> None:
                self.data = b""

            def write(self, b) -> int:
                self.data += bytes(b)
                return len(b)

        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            out_path = os.path.join(directory, "out.bin")
            binary = os.O_BINARY if os.name == "nt" else 0
            appending = os.open(
                out_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | binary
            )
            outputs = {
                "BytesIO": lambda: io.BytesIO(),
                "buffered BytesIO, no file descriptor": lambda: io.BufferedWriter(
                    io.BytesIO()
                ),
                "devnull": lambda: open(os.devnull, "wb"),
                "O_APPEND opened as wb": lambda: os.fdopen(appending, "wb"),
                "gzip": lambda: gzip.GzipFile(out_path + ".gz", "wb"),
                "write-only": WriteOnly,
            }
            for name, make in outputs.items():
                with self.subTest(name):
                    outfile = make()
                    with mock.patch.object(FileBackedData, "_COPY_CHUNK_SIZE", 4096):
                        data.write_to_file(outfile, consume=True)
                    if hasattr(outfile, "close"):
                        outfile.close()
                    self._assert_intact(data)
            with open(out_path, "rb") as result:
                self.assertEqual(self._PAYLOAD, result.read())

    def test_consume_copies_into_a_pipe_that_reports_it_can_seek(self) -> None:
        class ReportsSeekable(io.BufferedWriter):
            # On Windows a pipe reports seekable(), so the move can't rely on it.
            def seekable(self) -> bool:
                return True

        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            read, write = os.pipe()
            # The data is bigger than a pipe's buffer, so read it while it is written.
            received = []
            with open(read, "rb") as infile:
                reader = threading.Thread(target=lambda: received.append(infile.read()))
                reader.start()
                with ReportsSeekable(io.FileIO(write, "wb")) as outfile:
                    data.write_to_file(outfile, consume=True)
                reader.join()
            self.assertEqual([self._PAYLOAD], received)
            self._assert_intact(data)

    @_needs_move
    def test_consume_copies_into_a_named_pipe_next_to_the_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            fifo = os.path.join(directory, "out.fifo")
            os.mkfifo(fifo)
            received = []

            def read() -> None:
                with open(fifo, "rb") as infile:
                    received.append(infile.read())

            reader = threading.Thread(target=read)
            reader.start()
            with open(fifo, "wb") as outfile:
                data.write_to_file(outfile, consume=True)
            reader.join()
            self.assertEqual([self._PAYLOAD], received)
            self._assert_intact(data)

    def test_consume_copies_a_file_with_another_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            os.link(data._path, os.path.join(directory, "other_name.bin"))
            with open(os.path.join(directory, "out.bin"), "wb") as outfile:
                data.write_to_file(outfile, consume=True)
            self._assert_intact(data)

    def test_consume_copies_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            with open(os.path.join(directory, "out.bin"), "wb") as outfile:
                with mock.patch.object(os, "name", "nt"):
                    data.write_to_file(outfile, consume=True)
            self._assert_intact(data)

    @_needs_move
    def test_a_consuming_write_that_fails_first_keeps_the_data(self) -> None:
        class Short(io.FileIO):
            def write(self, b) -> int:
                return super().write(bytes(b)[:-1])

        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            out_path = os.path.join(directory, "out.bin")
            with Short(out_path, "wb") as outfile:
                with self.assertRaises(OSError):
                    data.write_to_file(outfile, consume=True)
            self._assert_intact(data)
            with open(out_path, "wb") as outfile:
                data.write_to_file(outfile, consume=True)
            with open(out_path, "rb") as result:
                self.assertEqual(self._PAYLOAD, result.read())

    @_needs_move
    def test_a_consuming_write_that_fails_later_says_where_the_bytes_are(self) -> None:
        payload, prefix = self._PAYLOAD, b"Prefix"

        class FailsThird(io.FileIO):
            writes = 0

            def write(self, b) -> int:
                # The prefix, then the first chunk moved, then the failure.
                FailsThird.writes += 1
                if FailsThird.writes == 3:
                    raise OSError("disk full")
                return super().write(b)

        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            out_path = os.path.join(directory, "out.bin")
            with mock.patch.object(FileBackedData, "_COPY_CHUNK_SIZE", self._CHUNK):
                with FailsThird(out_path, "wb") as outfile:
                    outfile.write(prefix)
                    with self.assertRaises(OSError) as raised:
                        data.write_to_file(outfile, consume=True)
            # The first chunk moved is the last one, a full chunk.
            left = len(payload) - self._CHUNK
            path = data._path
            del data  # The bytes the note points to outlive the object.
            note = raised.exception.__notes__[0]
            self.assertIn(f"Bytes [0, {left}) are still in {path}", note)
            self.assertIn(f"at offset {len(prefix) + left} of the output", note)
            with open(path, "rb") as source:
                self.assertEqual(payload[:left], source.read())
            with open(out_path, "rb") as result:
                result.seek(len(prefix) + left)
                self.assertEqual(payload[left:], result.read())

    @_needs_move
    def test_an_interrupt_right_after_a_truncation_gets_a_true_note(self) -> None:
        ftruncate = os.ftruncate

        def interrupted_ftruncate(fd: int, length: int) -> None:
            ftruncate(fd, length)
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            with mock.patch.object(FileBackedData, "_COPY_CHUNK_SIZE", self._CHUNK):
                with open(os.path.join(directory, "out.bin"), "wb") as outfile:
                    with mock.patch.object(os, "ftruncate", interrupted_ftruncate):
                        with self.assertRaises(KeyboardInterrupt) as raised:
                            data.write_to_file(outfile, consume=True)
            left = len(self._PAYLOAD) - self._CHUNK
            path = data._path
            del data
            self.assertEqual(left, os.path.getsize(path))
            self.assertIn(f"Bytes [0, {left})", raised.exception.__notes__[0])

    @_needs_move
    def test_a_write_error_reported_by_the_sync_keeps_the_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            with open(os.path.join(directory, "out.bin"), "wb") as outfile:
                with mock.patch.object(os, "fsync", side_effect=OSError("quota")):
                    with self.assertRaises(OSError):
                        data.write_to_file(outfile, consume=True)
            self._assert_intact(data)

    @_needs_move
    def test_a_failed_consuming_cord_write_says_where_the_moved_buffers_are(
        self,
    ) -> None:
        payload, prefix = self._PAYLOAD, b"Prefix"
        second_starts = len(prefix) + len(payload)

        class FailsInSecond(io.FileIO):
            def write(self, b) -> int:
                if self.tell() >= second_starts:
                    raise OSError("disk full")
                return super().write(b)

        with tempfile.TemporaryDirectory() as directory:
            first, second = self._owned(directory), self._owned(directory)
            cord = Cord(prefix)
            cord.append(first)
            cord.append(second)
            out_path = os.path.join(directory, "out.bin")
            with FailsInSecond(out_path, "wb") as outfile:
                with self.assertRaises(OSError) as raised:
                    cord.write_to_file(outfile, consume=True)
            self.assertIn(
                f"{first._path} ({len(payload)} bytes) at offset {len(prefix)}",
                raised.exception.__notes__[-1],
            )
            with open(out_path, "rb") as result:
                self.assertEqual(prefix + payload, result.read(second_starts))
            self._assert_intact(second)

    @_needs_move
    def test_consume_copies_a_symlink_and_keeps_its_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "target.bin")
            with open(target, "wb") as f:
                f.write(self._PAYLOAD)
            link = os.path.join(directory, "link.bin")
            os.symlink(target, link)
            data = FileBackedData.move_from(link)
            with open(os.path.join(directory, "out.bin"), "wb") as outfile:
                data.write_to_file(outfile, consume=True)
            with open(target, "rb") as f:
                self.assertEqual(self._PAYLOAD, f.read())

    @_needs_move
    def test_consume_copies_into_another_file_system(self) -> None:
        fstat = os.fstat
        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            with open(os.path.join(directory, "out.bin"), "wb") as outfile:

                def other_device_fstat(fd: int) -> os.stat_result:
                    st = fstat(fd)
                    if fd != outfile.fileno():
                        return st
                    return os.stat_result((*st[:2], st.st_dev + 1, *st[3:10]))

                with mock.patch.object(os, "fstat", other_device_fstat):
                    data.write_to_file(outfile, consume=True)
            self._assert_intact(data)

    @_needs_move
    def test_another_buffer_on_a_consumed_file_refuses_to_copy_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            second = FileBackedData(data._path)
            with open(os.path.join(directory, "out.bin"), "wb") as outfile:
                data.write_to_file(outfile, consume=True)
                with self.assertRaises(RuntimeError):
                    second.write_to_file(outfile)

    @_needs_move
    def test_a_short_read_stops_the_move_and_keeps_the_data(self) -> None:
        pread = os.pread
        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            with open(os.path.join(directory, "out.bin"), "wb") as outfile:
                with mock.patch.object(
                    os, "pread", lambda fd, n, offset: pread(fd, n - 1, offset)
                ):
                    with self.assertRaises(OSError):
                        data.write_to_file(outfile, consume=True)
            self._assert_intact(data)

    @_needs_move
    def test_a_consuming_write_of_an_emptied_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            second = FileBackedData(data._path, cleanup=True)
            with open(os.path.join(directory, "out.bin"), "wb") as outfile:
                data.write_to_file(outfile, consume=True)
                with self.assertRaises(RuntimeError):
                    second.write_to_file(outfile, consume=True)

    @_needs_move
    def test_consume_copies_data_it_cannot_write_to(self) -> None:
        if os.geteuid() == 0:
            self.skipTest("root can write to any file")
        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            os.chmod(data._path, 0o444)
            with open(os.path.join(directory, "out.bin"), "wb") as outfile:
                data.write_to_file(outfile, consume=True)
            self._assert_intact(data)

    def test_consume_copies_buffers_listed_to_keep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = self._owned(directory)
            with open(os.path.join(directory, "out.bin"), "wb") as outfile:
                Cord(data).write_to_file(outfile, consume=True, keep={data})
            self._assert_intact(data)
